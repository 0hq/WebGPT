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

class TransposeBlockClass extends Block {
  constructor() {
    super();
    this.name = "transpose";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.transposeNewShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuf) {
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_Layout, [inputBuf], `${this.name}_InputG`);
    const workgroups = { x: 100, y: 100, z: 1 };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    return {
      resultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline,
          groups: [opBindGroup, inputBindGroup],
          workgroups,
        },
      ],
    };
  }

  transposeNewShader = `
    struct Meta {
      M: u32,
      N: u32,
    }
    
    @group(1) @binding(0) var<storage, read> input_array: array<f32>;
    
    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;
    
    // Bank conflicts?
    var<workgroup> tile: array<array<f32, 8>, 8>;
    
    @compute @workgroup_size(8, 8)
    fn main (@builtin(workgroup_id) wg_id: vec3<u32>,  @builtin(local_invocation_id) local_id: vec3<u32>) {
      let col: u32 = wg_id.x;
      let row: u32 = wg_id.y;
      let N: u32 = uniforms.N;
      let M: u32 = uniforms.M;

      let tile_col = col * 8u + local_id.x;
      let tile_row = row * 8u + local_id.y;
    
      // Load a tile from input_array to shared memory tile
      if (tile_row < M && tile_col < N) {
        tile[local_id.y][local_id.x] = input_array[tile_row * N + tile_col];
      }
    
      workgroupBarrier(); // Ensure all threads have finished writing to the shared memory before proceeding
    
      // Compute transposed coordinates
      let transposed_col: u32 = row * 8u + local_id.x;
      let transposed_row: u32 = col * 8u + local_id.y;
    
      // Write the transposed tile to result_array
      if (transposed_col < M && transposed_row < N) {
        result_array[transposed_row * M + transposed_col] = tile[local_id.x][local_id.y]; // This line was incorrect
      }
    }
  `;
}

class SplitQBlockClass extends Block {
  constructor() {
    super();
    this.name = "splitq";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.splitQShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, numHeads, inputBuf) {
    if (cols % numHeads !== 0) throw new Error(`cols ${cols} must be divisible by numHeads ${numHeads}`);
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_Layout, [inputBuf], `${this.name}_InputG`);
    const workgroups = { x: 100, y: 100, z: 1 };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, cols / numHeads]));

    return {
      resultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline,
          groups: [opBindGroup, inputBindGroup],
          workgroups,
        },
      ],
    };
  }

  splitQShader = `
    struct Meta {
      M: u32,
      N: u32,
      HSize: u32,
    }

    @group(1) @binding(0) var<storage, read> input_array: array<f32>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;

    var<workgroup> tile: array<array<f32, 8>, 8>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
      let col: u32 = workgroup_id.x * 8 + local_id.x;
      let row: u32 = workgroup_id.y * 8 + local_id.y;
      let N: u32 = uniforms.N;
      let M: u32 = uniforms.M;

      // Load a tile from input_array to shared memory tile
      if (row < M && col < N) {
          tile[local_id.y][local_id.x] = input_array[row * N + col];
      }

      workgroupBarrier(); // Ensure all threads have finished writing to the shared memory before proceeding

      let HSize: u32 = uniforms.HSize;
      let xOffset: u32 = col % HSize;
      let yOffset: u32 = row * HSize + (col / HSize) * HSize * M;

      // Write the tile to result_array
      if (row < M && col < N) {
          result_array[yOffset + xOffset] = tile[local_id.y][local_id.x];
      }
    } 
  `;
}

const CausalMaskBlock = new CausalMaskBlockClass();
const OutputBlock = new OutputBlockClass();
const TransposeBlock = new TransposeBlockClass();
const SplitQBlock = new SplitQBlockClass();

operations.push(CausalMaskBlock, OutputBlock, TransposeBlock, SplitQBlock);

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
    const { M, N } = { M: 4, N: 64 };
    const input_array = new Float32Array(M * N);
    const weight_array = new Float32Array(N * N);
    const seq_length = 10;
    const embed_array = new Float32Array(seq_length * M);
    for (let i = 0; i < seq_length * M; i++) {
      if (i >= seq_length * M - M + 1) {
        embed_array[i] = 1;
      }
    }

    for (let i = 0; i < M * N; i++) input_array[i] = 1;
    for (let i = 0; i < M; i++) weight_array[i * M + i] = 1;
    console.log(formatAsMatrix(input_array, M, N));
    console.log(formatAsMatrix(embed_array, seq_length, M));

    const inputBuffer = this.initTensor(input_array, [M, N], ["storage"]);
    const weightBuffer = this.initTensor(weight_array, [N, N], ["storage"]);
    const embedBuffer = this.initTensor(embed_array, [seq_length, M], ["storage", "copy_from"]);

    this.computePasses = [];
    const push = ({ passes, resultBuffer }) => {
      this.computePasses.push(...passes);
      return resultBuffer;
    };

    const numHeads = 4;

    let intermediateBuffer = inputBuffer;
    intermediateBuffer = push(DeEmbedBlock.newInstance(M, N, N, seq_length, N, embedBuffer, [intermediateBuffer]));
    // intermediateBuffer = push(OutputBlock.newInstance(1, N, intermediateBuffer));
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
    console.log(formatAsMatrix(outputArray, 1, N));

    // ---------------- Create Passes ---------------- //

    // this.computePasses = [];

    // intermediateBuffer = inputBuffer;
    // intermediateBuffer = push(CausalMaskBlock.newInstance(M, N, intermediateBuffer)); // Transposes!
    // intermediateBuffer = push(SoftmaxBlock.newInstance(N, M, intermediateBuffer));
    // intermediateBuffer = push(OutputBlock.newInstance(N, M, intermediateBuffer));
    // resultBuffer = intermediateBuffer;

    // // ---------------- Compute Passes ----------------

    // const commandEncoder2 = this.device.createCommandEncoder();
    // for (const pass of this.computePasses) {
    //   if (pass.flag === "compute") {
    //     const passEncoder = commandEncoder2.beginComputePass();
    //     passEncoder.setPipeline(pass.pipeline);
    //     for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
    //     passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
    //     passEncoder.end();
    //   } else if (pass.flag === "copy") {
    //     commandEncoder2.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
    //   }
    // }
    // this.device.queue.submit([commandEncoder2.finish()]);

    // // ---------------- Read Results ----------------

    // await resultBuffer.mapAsync(GPUMapMode.READ);
    // const output2 = resultBuffer.getMappedRange();
    // const outputArray2 = new Float32Array(output2).slice(0); // Copy the array, otherwise it'll be destroyed.
    // console.log(formatAsMatrix(outputArray2, N, M));

    // // ---------------- Compare Results ----------------

    // let error = 0;
    // for (let i = 0; i < outputArray.length; i++) {
    //   error += Math.abs(outputArray[i] - outputArray2[i]);
    // }
    // console.log("Error: ", error);

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
