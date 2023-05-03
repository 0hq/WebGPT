class OutputBlockClass extends Block {
  constructor() {
    super();
    this.name = "output";
  }

  newInstance(row, col, inputBuffer) {
    const outputBuffer = this.initBuffer(["map_read", "copy_to"], [row, col]);

    const copyCommand = {
      flag: "copy",
      src: inputBuffer,
      srcOffset: 0,
      dst: outputBuffer,
      dstOffset: 0,
      size: this.bufferSize(row, col),
    };

    return {
      resultBuffer: outputBuffer,
      passes: [copyCommand],
    };
  }
}

class CausalMaskBlockClass extends Block {
  constructor() {
    super();
    this.name = "causal_mask";
    this.pipelineCache = new Map();
  }

  getSimpleCausalMaskPipeline() {
    const pipelineCacheKey = `${this.name}_simplecausalmask`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.origCausalMaskShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_CausalMask`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getCausalMaskPipeline() {
    const pipelineCacheKey = `${this.name}_causalmask`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.causalMaskShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_CausalMask`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuffer) {
    const causalMaskPipeline = this.getCausalMaskPipeline();
    const causalMaskUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const causalMaskResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const causalMaskBindGroup = this.initBindGroup(this.u_s_Layout, [causalMaskUniformBuffer, causalMaskResultBuffer], `${this.name}_CausalMaskG`);
    const causalMaskInputBindGroup = this.initBindGroup(this.r_Layout, [inputBuffer], `${this.name}_CausalMaskInputG`);
    this.device.queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([cols, rows])); // Transposes! This is needed for softmax.
    const causalMaskWorkgroups = { x: wgSize(rows, 16), y: wgSize(cols, 16), z: 1 };

    return {
      resultBuffer: causalMaskResultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline: causalMaskPipeline,
          groups: [causalMaskBindGroup, causalMaskInputBindGroup],
          workgroups: causalMaskWorkgroups,
        },
      ],
    };
  }

  simpleCausalMaskShader = `
    struct Matrix {
        data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension of input matrix
      dimX: u32, // col dimension of input matrix
    };

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      let rowMask: u32 = row % dimX;
      if (row >= dimY || col >= dimX) {
        return;
      }

      if (col > rowMask) {
        Result.data[row * dimX + col] = 0.0;
      } else {
        let rowNum: u32 = row / dimX;
        Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];
      }
    }
  `;

  origCausalMaskShader = `
    struct Matrix {
        data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension of input matrix
      dimX: u32, // col dimension of input matrix
    };

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      let rowMask: u32 = row % dimX;
      if (row >= dimY || col > rowMask) {
        return;
      }

      let rowNum: u32 = row / dimX;
      Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];
    }
  `;

  causalMaskShader = `
    struct Matrix {
        data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension of input matrix
      dimX: u32, // col dimension of input matrix
    };

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      let rowMask: u32 = row % dimX;
      let rowNum: u32 = row / dimX;
      let index = row * dimX + col;
      let causalMask: bool = (col <= rowMask);
      Result.data[index] = select(-1e9, Input.data[rowMask * dimY + col + rowNum * dimX], causalMask);
    }
  `;
}

const CausalMaskBlock = new CausalMaskBlockClass();
const OutputBlock = new OutputBlockClass();

operations.push(CausalMaskBlock, OutputBlock);

class TestShader {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;

    this.unloadDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    initializeOperations(this.device);

    this.initialized = true;

    console.log("Model initialized");
  }

  async test() {
    // ---------------- Create Passes ---------------- //
    const { M, N } = { M: 100, N: 300 };
    const input_array = new Float32Array(M * N);
    const weight_array = new Float32Array(M * N).fill(1);

    for (let i = 0; i < M * N; i++) input_array[i] = i;
    console.log(formatAsMatrix(input_array, M, N));

    const inputBuffer = this.initTensor(input_array, [M, N], ["storage"]);
    const weightBuffer = this.initTensor(weight_array, [M, N], ["storage"]);

    this.computePasses = [];
    const push = ({ passes, resultBuffer }) => {
      this.computePasses.push(...passes);
      return resultBuffer;
    };

    let intermediateBuffer = inputBuffer;
    intermediateBuffer = push(CausalMaskBlock.newInstance(M, N, intermediateBuffer)); // Transposes!
    intermediateBuffer = push(SoftmaxBlock.newInstance(N, M, intermediateBuffer));
    intermediateBuffer = push(OutputBlock.newInstance(N, M, intermediateBuffer));
    let resultBuffer = intermediateBuffer;

    // ---------------- Compute Passes ----------------

    const commandEncoder = this.device.createCommandEncoder();
    for (const pass of this.computePasses) {
      if (pass.flag === "compute") {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pass.pipeline);
        for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
        passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
        passEncoder.end();
      } else if (pass.flag === "copy") {
        commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);

    // ---------------- Read Results ----------------

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const output = resultBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.
    console.log(formatAsMatrix(outputArray, N, M));

    // ---------------- Create Passes ---------------- //

    this.computePasses = [];

    intermediateBuffer = inputBuffer;
    intermediateBuffer = push(CausalMaskBlock.newInstance(M, N, intermediateBuffer)); // Transposes!
    intermediateBuffer = push(SoftmaxBlock.newInstance(N, M, intermediateBuffer));
    intermediateBuffer = push(OutputBlock.newInstance(N, M, intermediateBuffer));
    resultBuffer = intermediateBuffer;

    // ---------------- Compute Passes ----------------

    const commandEncoder2 = this.device.createCommandEncoder();
    for (const pass of this.computePasses) {
      if (pass.flag === "compute") {
        const passEncoder = commandEncoder2.beginComputePass();
        passEncoder.setPipeline(pass.pipeline);
        for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
        passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
        passEncoder.end();
      } else if (pass.flag === "copy") {
        commandEncoder2.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
      }
    }
    this.device.queue.submit([commandEncoder2.finish()]);

    // ---------------- Read Results ----------------

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const output2 = resultBuffer.getMappedRange();
    const outputArray2 = new Float32Array(output2).slice(0); // Copy the array, otherwise it'll be destroyed.
    console.log(formatAsMatrix(outputArray2, N, M));

    // ---------------- Compare Results ----------------

    let error = 0;
    for (let i = 0; i < outputArray.length; i++) {
      error += Math.abs(outputArray[i] - outputArray2[i]);
    }
    console.log("Error: ", error);

    // ---------------- Cleanup ----------------

    destroyOperationBuffers();
    this.unloadBuffers();

    return outputArray;
  }

  initTensor(data, dims, ops) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    return Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }
}

async function testInstruction() {
  const testShader = new TestShader();
  await testShader.initialize();
  await testShader.test();
}
