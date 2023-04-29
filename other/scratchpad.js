// Adjusts the input matrix by the mean and standard deviation and gamma and beta parameters.
const fusedSoftmaxShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    M: u32, // row dimension of input matrix
    N: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(1) @binding(1) var<workgroup> shared_sum: array<f32>;

  var<workgroup> sumValues : array<f32, 256>;

  @compute @workgroup_size(256, 1)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col: u32 = global_id.x;
    let row: u32 = global_id.y;
    let N: u32 = DimBuffer.N;
    let M: u32 = DimBuffer.M;

    if (row >= M || col >= N) {
      return;
    }

    // Calculate the exponential of each element in the input matrix
    let exponent: f32 = exp(Input.data[row * N + col]);

    // Store partial sums in shared memory
    partial_sums[local_id.x] = exponent;

    // Synchronize threads in the workgroup
    workgroupBarrier();

    // Perform parallel reduction to compute row-wise sum of exponentials
    for (var offset: u32 = 128u; offset > 0u; offset = offset / 2u) {
      if (local_id.x < offset && local_id.x + offset < N) {
        partial_sums[local_id.x] = partial_sums[local_id.x] + partial_sums[local_id.x + offset];
      }
      workgroupBarrier();
    }

    // Normalize each element by dividing it by the sum of exponentials in its row
    let softmax_val: f32 = exponent / partial_sums[0];

    // Store the result in the output matrix
    Result.data[row * N + col] = softmax_val;
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

  async test() {
    const dimM = 1024;
    const dimN = 1024;
    const matrixA = new Float32Array(dimM * dimN);
    for (let i = 0; i < dimM * dimN; i++) matrixA[i] = i + 1;
    const matrixABuffer = this.initBuffer(["storage", "copy_to"], dimM, dimN, 1);
    this.device.queue.writeBuffer(matrixABuffer, 0, matrixA);

    const commandEncoder = this.device.createCommandEncoder();
    const softmaxOutput = this.maskedInlineSoftmax(commandEncoder, dimM, dimN, matrixABuffer);
    const outputBuffer = this.initOutputBuffer(commandEncoder, softmaxOutput, dimM, dimN);
    this.device.queue.submit([commandEncoder.finish()]);

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const output = outputBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Prevent destruction.
    console.log(outputArray);

    this.destroyBuffers();
  }

  maskedInlineSoftmax(commandEncoder, rows, cols, inputBuffer) {
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const softmaxBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer]);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fusedSoftmaxPipeline);
    passEncoder.setBindGroup(0, softmaxBindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_Layout, [inputBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(cols, 256), wgSize(rows, 1));
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

    this.fusedSoftmaxPipeline = await p(fusedSoftmaxShader, [this.u_s_Layout, this.r_r_Layout]);
  }
}

async function test() {
  const GPU = new TestGPT();
  await GPU.initialize();
  await GPU.test();
}
