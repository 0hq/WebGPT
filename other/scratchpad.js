const fastRowAddShader = `
  struct BMeta {
    M: u32,
    N: u32,
    ND4: u32,
  }

  @group(1) @binding(0) var<storage,read> array_matrix: array<vec4<f32>>;
  @group(1) @binding(1) var<storage,read> array_bias: array<vec4<f32>>;
  @group(0) @binding(0) var<uniform> bmeta: BMeta;
  @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

  @compute @workgroup_size(8,8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var col: u32 = global_id.x;
    var row: u32 = global_id.y;
    var ND4: u32 = bmeta.ND4;
    var M: u32 = bmeta.M;
    
    if (row >= M || col >= ND4) {
      return;
    }

    array_output[row * ND4 + col] = array_matrix[row * ND4 + col] + array_bias[col];
  }
`;

const slowRowAddShader = `
  struct BMeta {
    M: u32,
    N: u32,
  }

  @group(1) @binding(0) var<storage,read> array_matrix: array<f32>;
  @group(1) @binding(1) var<storage,read> array_bias: array<f32>;
  @group(0) @binding(0) var<uniform> bmeta: BMeta;
  @group(0) @binding(1) var<storage,read_write> array_output: array<f32>;

  @compute @workgroup_size(8,8)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var col: u32 = global_id.x;
    var row: u32 = global_id.y;
    var N: u32 = bmeta.N;
    var M: u32 = bmeta.M;
    
    if (row >= M || col >= N) {
      return;
    }

    array_output[row * N + col] = array_matrix[row * N + col] + array_bias[col];
  }
`;

class TestGPT {
  constructor(folder, type, doAttentionCache = false) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;
    this.doAttentionCache = doAttentionCache;

    this.defaultPrompt;
    this.defaultTopK;
    this.defaultTemperature;
    this.defaultTokens;

    this.bufferDeletionStack = [];
    this.unloadDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    this.initBindGroups();
    await this.initPipelines();

    this.initialized = true;
  }

  async testFastRowAdd(iters = 5, runs = 10, retries = 3) {
    const dimM = 1000;
    const dimN = 25128;
    const matrixA = new Float32Array(dimM * dimN);
    const matrixB = new Float32Array(dimN);
    for (let i = 0; i < dimM * dimN; i++) matrixA[i] = i + 1;
    for (let i = 0; i < dimN; i++) matrixB[i] = i + 1;
    const matrixABuffer = this.initBuffer(["storage", "copy_to"], dimM, dimN, 1, true);
    const matrixBBuffer = this.initBuffer(["storage", "copy_to"], dimN, 1, true);
    this.device.queue.writeBuffer(matrixBBuffer, 0, matrixB);
    this.device.queue.writeBuffer(matrixABuffer, 0, matrixA);

    let times = [];
    for (let i = 0; i < retries; i++) {
      let runTime = 0;
      for (let i = 0; i < runs; i++) {
        const startTime = performance.now();
        const commandEncoder = this.device.createCommandEncoder();
        for (let i = 0; i < iters; i++) {
          this.inlineSlowRowAdd(commandEncoder, matrixABuffer, matrixBBuffer, dimM, dimN);
        }
        this.device.queue.submit([commandEncoder.finish()]);
        this.destroyBuffers();
        const endTime = performance.now();
        runTime += endTime - startTime;
      }

      console.log(`${runs * iters} runs of ${dimM}x${dimN} took ${runTime}ms`);
      console.log(`average time: ${runTime / (runs * iters)}ms`);
      times.push(runTime);
    }
    this.unloadBuffers();

    // Average of times
    let sum = 0;
    for (let i = 0; i < times.length; i++) {
      sum += times[i];
    }
    console.log("Average:", sum / times.length);

    // Median of times
    times.sort((a, b) => a - b);
    console.log("Median:", times[Math.floor(times.length / 2)]);
  }

  inlineFastRowAdd(commandEncoder, inputBuffer, biasBuffer, rows, cols) {
    if (cols % 4 !== 0) throw new Error(`cols must be a multiple of 4, got ${rows}x${cols}`);

    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const bindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer]);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, cols / 4]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fastRowAddPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [inputBuffer, biasBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(cols, 32), wgSize(rows, 8));
    passEncoder.end();

    return resultBuffer;
  }

  inlineSlowRowAdd(commandEncoder, inputBuffer, biasBuffer, rows, cols) {
    if (cols % 4 !== 0) throw new Error(`cols must be a multiple of 4, got ${rows}x${cols}`);

    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const bindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer]);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.slowRowAddPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [inputBuffer, biasBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(rows, 8), wgSize(cols, 8));
    passEncoder.end();

    return resultBuffer;
  }

  initBindGroup(layout, buffers) {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
    });
  }

  initOutputBuffer(commandEncoder, buffer, row, col) {
    const outputBuffer = this.initBuffer(["map_read", "copy_to"], row, col);
    commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, this.bufferSize(row, col));
    return outputBuffer;
  }

  initBuffer(ops, row, col = 1, noDelete = false) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    if (!noDelete) this.bufferDeletionStack.push(buffer);
    else this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  initTensor(data, sizeA, sizeB, ops) {
    const buffer = this.initBuffer([...ops, "copy_to"], sizeA, sizeB, true);
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  bufferSize(dimA, dimB = 1) {
    return Math.ceil((dimA * dimB * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }

  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  async initPipelines() {
    const p = (code, bindGroupLayouts) => {
      return this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts }),
        compute: {
          module: this.device.createShaderModule({ code }),
          entryPoint: "main",
        },
      });
    };

    this.fastRowAddPipeline = await p(fastRowAddShader, [this.u_s_Layout, this.r_r_Layout]);
    this.slowRowAddPipeline = await p(slowRowAddShader, [this.u_s_Layout, this.r_r_Layout]);
  }
}

async function test() {
  const GPU = new TestGPT();
  await GPU.initialize();
  await GPU.testFastRowAdd();
}
