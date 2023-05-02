/*

  Softmax will run via rows, every workgroup handling a different row.
  Override constants.

*/

class Block {
  constructor() {
    this.bufferDeletionStack = [];
  }

  initialize(device) {
    this.device = device;
    this.initBindGroups();
  }

  initBindGroup(layout, buffers, label = "") {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
      label,
    });
  }

  initBuffer(ops, dims) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    this.bufferDeletionStack.push(buffer);
    return buffer;
  }

  bufferSize(dimA, dimB = 1) {
    return Math.ceil((dimA * dimB * Float32Array.BYTES_PER_ELEMENT) / 1) * 1;
  }

  // Could be removed with auto bind groups, currently initializing everytime so probably slowing things down.
  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  initPipeline(code, bindGroupLayouts, label = "", constants = {}) {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts }),
      compute: {
        module: this.device.createShaderModule({ code }),
        entryPoint: "main",
        constants,
      },
      label,
    });
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }
}

class FastMatMulBlockClass extends Block {
  constructor() {
    super();
    this.name = "fastMatMul";
    this.pipelineCache = new Map();
  }

  getPipeline(rows) {
    const div4 = rows % 4 === 0;
    const pipelineCacheKey = div4 ? "fastMatMulNoCheck" : "fastMatMul";
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const kernel = div4 ? this.fastMatMulNoCheck : this.fastMatMul;
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_${pipelineCacheKey}`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, bufA, bufB) {
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, bufB], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 64), y: wgSize(rows, 32) };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

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

  fastMatMul = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      if (y * 4u + 0u < M) {
        array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
        array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      }
      if (y * 4u + 1u < M) {
        array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
        array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      }
      if (y * 4u + 2u < M) {
        array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
        array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      }
      if (y * 4u + 3u < M) {
        array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
        array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
      }
    }
  `;

  fastMatMulNoCheck = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
      array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
      array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
      array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
      array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
    }
  `;
}

class FastMLPBlockClass extends Block {
  constructor() {
    super();
    this.name = "fastMLP";
    this.pipelineCache = new Map();
  }

  getPipeline(rows) {
    const div4 = rows % 4 === 0;
    const pipelineCacheKey = div4 ? "fastMLPNoCheck" : "fastMLP";
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const kernel = div4 ? this.fastMLPNoCheck : this.fastMLP;
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_r_Layout], `${this.name}_Pipeline_${pipelineCacheKey}`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, inputBuffer, weightsBuffer, biasBuffer) {
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_r_Layout, [inputBuffer, weightsBuffer, biasBuffer], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 64), y: wgSize(rows, 32) };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

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

  fastMLP = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;
    @group(1) @binding(2) var<storage,read> array_bias: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      var array_bias_1: vec4<f32> = array_bias[x * 2u + 0u];
      sum00 = sum00 + array_bias_1;
      sum01 = sum01 + array_bias_1;
      sum02 = sum02 + array_bias_1;
      sum03 = sum03 + array_bias_1;

      var array_bias_2: vec4<f32> = array_bias[x * 2u + 1u];
      sum10 = sum10 + array_bias_2;
      sum11 = sum11 + array_bias_2;
      sum12 = sum12 + array_bias_2;
      sum13 = sum13 + array_bias_2;

      if (y * 4u + 0u < M) {
        array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
        array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      }
      if (y * 4u + 1u < M) {
        array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
        array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      }
      if (y * 4u + 2u < M) {
        array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
        array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      }
      if (y * 4u + 3u < M) {
        array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
        array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
      }
    }
  `;

  fastMLPNoCheck = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;
    @group(1) @binding(2) var<storage,read> array_bias: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      var array_bias_1: vec4<f32> = array_bias[x * 2u + 0u];
      sum00 = sum00 + array_bias_1;
      sum01 = sum01 + array_bias_1;
      sum02 = sum02 + array_bias_1;
      sum03 = sum03 + array_bias_1;

      var array_bias_2: vec4<f32> = array_bias[x * 2u + 1u];
      sum10 = sum10 + array_bias_2;
      sum11 = sum11 + array_bias_2;
      sum12 = sum12 + array_bias_2;
      sum13 = sum13 + array_bias_2;

      array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
      array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
      array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
      array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
      array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
    }
  `;
}

class ResidualBlockClass extends Block {
  constructor() {
    super();
    this.name = "residual";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.elementWiseAdditionShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, outputBuf, residualBuf) {
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [outputBuf, residualBuf], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 32), y: wgSize(rows, 8), z: 1 };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, Math.ceil(cols / 4)]));

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

  elementWiseAdditionShader = `
    struct Meta {
      M: u32,
      ND4: u32,
    }

    @group(1) @binding(0) var<storage, read> layer_out_array: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read> residual_array: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      var col: u32 = global_id.x;
      var row: u32 = global_id.y;
      var ND4: u32 = uniforms.ND4;
      var M: u32 = uniforms.M;
      
      // Unsure if this is any faster than if return bounds check.
      let mask = (row < M) && (col < ND4);
      let index = row * ND4 + col;
      result_array[index] = select(result_array[index], layer_out_array[index] + residual_array[index], mask);
    }
  `;
}

class NaiveMatMulBlockClass extends Block {
  constructor() {
    super();
    this.name = "naiveMatMul";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.matMulShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, bufA, bufB) {
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OutputG`);
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, bufB], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 16), y: wgSize(rows, 16), z: 1 };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, shared]));

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

  // Experimenting with preloading all weights, not too important just style.
  preloadInstance(cols, shared, bufB) {
    this.cols = cols;
    this.shared = shared;
    this.weightsBuf = bufB;

    return (newPreloadedInstance = (rows, bufA) => {
      const pipeline = this.getPipeline();
      const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
      const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, this.cols]);
      const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OutputG`);
      const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, this.weightsBuf], `${this.name}_InputG`);
      const workgroups = { x: wgSize(this.cols, 16), y: wgSize(rows, 16), z: 1 };
      this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, this.cols, this.shared]));

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
    });
  }

  matMulShader = `
    struct Matrix {
        data: array<f32>,
    }

    struct Uniforms {
      dimY: u32, // row dimension of A and row dimension of C
      dimX: u32, // col dimension of B and col dimension of C
      dimS: u32, // shared dimension of A and B
    };

    @group(1) @binding(0) var<storage, read> A: Matrix;
    @group(1) @binding(1) var<storage, read> B: Matrix;

    @group(0) @binding(1) var<storage, read_write> C: Matrix;
    @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
        let col: u32 = global_id.x;
        let row: u32 = global_id.y;
        let dimX: u32 = dimBuffer.dimX;
        let dimY: u32 = dimBuffer.dimY;
        let dimS: u32 = dimBuffer.dimS;

        if (row >= dimY || col >= dimX) {
          return;
        }

        var sum: f32 = 0.0;
        for (var i: u32 = 0; i < dimS; i = i + 1) {
            sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col];
        }

        C.data[row * dimX + col] = sum;
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
    const pipeline = this.initPipeline(this.transposeShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuf) {
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [inputBuf], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 16), y: wgSize(rows, 16), z: 1 };
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

  transposeShader = `
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
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[col * dimY + row];
    }
  `;
}

class LayerNormBlockClass extends Block {
  constructor() {
    super();
    this.name = "layerNorm";
    this.pipelineCache = new Map();
  }

  getStatsPipeline() {
    const pipelineCacheKey = `${this.name}_stats`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.normStatsShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_Stats`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getNormPipeline() {
    const pipelineCacheKey = `${this.name}_norm`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.normShader, [this.u_s_Layout, this.r_r_r_r_Layout], `${this.name}_Pipeline_Norm`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuffer, gammaBuffer, betaBuffer) {
    const statsPipeline = this.getStatsPipeline();
    const statsUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const statsResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, 2]);
    const statsBindGroup = this.initBindGroup(this.u_s_Layout, [statsUniformBuffer, statsResultBuffer], `${this.name}_BindGroup_stats`);
    const statsInputBindGroup = this.initBindGroup(this.r_Layout, [inputBuffer], `${this.name}_InputG`);
    const statsWorkgroups = { x: wgSize(cols, 16), y: 1, z: 1 };
    this.device.queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

    const normPipeline = this.getNormPipeline();
    const normUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const normResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const normBindGroup = this.initBindGroup(this.u_s_Layout, [normUniformBuffer, normResultBuffer], `${this.name}_BindGroup_norm`);
    const normInputBindGroup = this.initBindGroup(
      this.r_r_r_r_Layout,
      [inputBuffer, gammaBuffer, betaBuffer, statsResultBuffer],
      `${this.name}_InputBindGroup_norm`
    );
    this.device.queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, cols]));
    const normWorkgroups = { x: wgSize(cols, 16), y: wgSize(rows, 16), z: 1 };

    return {
      resultBuffer: normResultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline: statsPipeline,
          groups: [statsBindGroup, statsInputBindGroup],
          workgroups: statsWorkgroups,
        },
        {
          flag: "compute",
          pipeline: normPipeline,
          groups: [normBindGroup, normInputBindGroup],
          workgroups: normWorkgroups,
        },
      ],
    };
  }

  normStatsShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension
      dimX: u32, // col dimension
    };

    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @compute @workgroup_size(16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY) {
        return;
      }

      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < dimX; i = i + 1) {
          sum = sum + Input.data[row * dimX + i];
      }
      var mean: f32 = sum / f32(dimX);

      var variance: f32 = 0.0;
      for (var i: u32 = 0; i < dimX; i = i + 1) {
          variance = variance + (Input.data[row * dimX + i] - mean) * (Input.data[row * dimX + i] - mean);
      }
      variance = variance / f32(dimX);
      var stdev: f32 = sqrt(variance + 1e-5);

      Result.data[row * 2] = mean;
      Result.data[row * 2 + 1] = stdev;
    }
  `;

  normShader = `
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
    @group(1) @binding(1) var<storage, read> Gamma: Matrix;
    @group(1) @binding(2) var<storage, read> Beta: Matrix;
    @group(1) @binding(3) var<storage, read> Stats: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      let mean = Stats.data[row * 2];
      let stdev = Stats.data[row * 2 + 1];
      let output = (Input.data[row * dimX + col] - mean) / stdev;
      let gamma = Gamma.data[col];
      let beta = Beta.data[col];
      let shift = gamma * output + beta;
      Result.data[row * dimX + col] = shift;
    }
  `;
}

class SoftmaxBlockClass extends Block {
  constructor() {
    super();
    this.name = "Softmax";
    this.pipelineCache = new Map();
  }

  getMaxPipeline() {
    const pipelineCacheKey = `${this.name}_max`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.maskedNegMaxShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_Max`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getAddPipeline() {
    const pipelineCacheKey = `${this.name}_add`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.addExpShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_Add`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getSumPipeline() {
    const pipelineCacheKey = `${this.name}_sum`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.sumShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_Sum`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getDivPipeline() {
    const pipelineCacheKey = `${this.name}_div`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.divideShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_Div`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuffer) {
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);

    const maxPipeline = this.getMaxPipeline();
    const maxResultBuffer = this.initBuffer(["storage", "copy_from"], [rows]);
    const maxBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, maxResultBuffer], `${this.name}_BindGroup_Max`);
    const maxInputBindGroup = this.initBindGroup(this.r_Layout, [inputBuffer], `${this.name}_BindGroup_Max_Input`);
    const maxWorkgroups = { x: wgSize(rows, 16), y: 1, z: 1 };

    const addPipeline = this.getAddPipeline();
    const addExpResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const addExpBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, addExpResultBuffer], `${this.name}_BindGroup_Add`);
    const addExpInputBindGroup = this.initBindGroup(this.r_r_Layout, [inputBuffer, maxResultBuffer], `${this.name}_BindGroup_Add_Input`);
    const addExpWorkgroups = { x: wgSize(cols, 16), y: wgSize(rows, 16), z: 1 };

    const sumPipeline = this.getSumPipeline();
    const sumResultBuffer = this.initBuffer(["storage", "copy_from"], [rows]);
    const sumBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, sumResultBuffer], `${this.name}_BindGroup_Sum`);
    const sumInputBindGroup = this.initBindGroup(this.r_Layout, [addExpResultBuffer]);
    const sumWorkgroups = { x: wgSize(rows, 16), y: 1, z: 1 };

    const divResultPipeline = this.getDivPipeline();
    const divResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols], `${this.name}_ResultBuffer_Div`);
    const divBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, divResultBuffer], `${this.name}_BindGroup_Div`);
    const divInputBindGroup = this.initBindGroup(this.r_r_Layout, [addExpResultBuffer, sumResultBuffer], `${this.name}_BindGroup_Div_Input`);
    const divWorkgroups = { x: wgSize(cols, 16), y: wgSize(rows, 16), z: 1 };

    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    return {
      resultBuffer: divResultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline: maxPipeline,
          groups: [maxBindGroup, maxInputBindGroup],
          workgroups: maxWorkgroups,
        },
        {
          flag: "compute",
          pipeline: addPipeline,
          groups: [addExpBindGroup, addExpInputBindGroup],
          workgroups: addExpWorkgroups,
        },
        {
          flag: "compute",
          pipeline: sumPipeline,
          groups: [sumBindGroup, sumInputBindGroup],
          workgroups: sumWorkgroups,
        },
        {
          flag: "compute",
          pipeline: divResultPipeline,
          groups: [divBindGroup, divInputBindGroup],
          workgroups: divWorkgroups,
        },
      ],
    };
  }

  maskedNegMaxShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension
      dimX: u32, // col dimension
    };

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;
    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @compute @workgroup_size(16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY) {
        return;
      }

      let rowMask: u32 = row % dimX;

      var max_buffer: f32 = 0.0;
      for (var i: u32 = 0; i < rowMask; i = i + 1) {
        max_buffer = max(max_buffer, Input.data[row * dimX + i]);
      }

      Result.data[row] = -max_buffer;
    }
  `;

  addExpShader = `
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
    @group(1) @binding(1) var<storage, read> Constants: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      let rowMask: u32 = row % dimX;

      if (row >= dimY || col > rowMask) {
        return;
      }

      Result.data[row * dimX + col] = exp(Input.data[row * dimX + col] + Constants.data[row]);
    }
  `;

  sumShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension
      dimX: u32, // col dimension
    };

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;
    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @compute @workgroup_size(16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY) {
        return;
      }

      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < dimX; i = i + 1) {
          sum = sum + Input.data[row * dimX + i];
      }

      Result.data[row] = sum;
    }
  `;

  divideShader = `
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
    @group(1) @binding(1) var<storage, read> Divisors: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
        let col: u32 = global_id.x;
        let row: u32 = global_id.y;
        let dimX: u32 = DimBuffer.dimX;
        let dimY: u32 = DimBuffer.dimY;

        if (row >= dimY || col >= dimX) {
          return;
        }

        Result.data[row * dimX + col] = Input.data[row * dimX + col] / Divisors.data[row];
      }
  `;
}

class GeluBlockClass extends Block {
  constructor() {
    super();
    this.name = "gelu";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.GELUShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuf) {
    const pipeline = this.getPipeline();
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_Layout, [inputBuf], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 32), y: wgSize(rows, 8), z: 1 };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, Math.ceil(cols / 4)]));

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

  GELUShader = `
    struct Meta {
      M: u32,
      ND4: u32,
    }

    const LOW_THRESHOLD: vec4<f32> = vec4<f32>(-10.0);
    const HIGH_THRESHOLD: vec4<f32> = vec4<f32>(10.0);
    const ZERO: vec4<f32> = vec4<f32>(0.0);
    const HALF: vec4<f32> = vec4<f32>(0.5);
    const SQRPI: vec4<f32> = vec4<f32>(0.7978845608);
    const COEFF: vec4<f32> = vec4<f32>(0.044715);
    fn gelu_vec4(x: vec4<f32>) -> vec4<f32> {
      let x_cubed: vec4<f32> = pow(x, vec4<f32>(3.0));
      let cdf_approx: vec4<f32> = HALF * (vec4<f32>(1.0) + tanh(SQRPI * (x + COEFF * x_cubed)));
  
      let result: vec4<f32> = x * cdf_approx;
  
      let lt_mask: vec4<bool> = x < LOW_THRESHOLD;
      let gt_mask: vec4<bool> = x > HIGH_THRESHOLD;
  
      return select(select(result, ZERO, lt_mask), x, gt_mask);
    }

    @group(1) @binding(0) var<storage,read> array_matrix: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      var col: u32 = global_id.x;
      var row: u32 = global_id.y;
      var ND4: u32 = uniforms.ND4;
      var M: u32 = uniforms.M;
      
      if (row >= M || col >= ND4) {
        return;
      }

      array_output[row * ND4 + col] = gelu_vec4(array_matrix[row * ND4 + col]);
    }
  `;
}

class AttentionBlockClass extends Block {
  constructor() {
    super();
    this.name = "attention";
    this.pipelineCache = new Map();
  }

  getSplitQKVPipeline() {
    const pipelineCacheKey = `${this.name}_splitkqv`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.splitQKVShader, [this.u_s_s_s_Layout, this.r_Layout], `${this.name}_Pipeline_SplitQKV`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getAttentionWeightsPipeline() {
    const pipelineCacheKey = `${this.name}_weights`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.attentionWeightsShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_AttWeights`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getCausalMaskPipeline() {
    const pipelineCacheKey = `${this.name}_causalmask`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.simpleCausalMaskShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_CausalMask`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getAttentionValuesPipeline() {
    const pipelineCacheKey = `${this.name}_values`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.attentionValuesShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_AttValues`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(
    seq_length,
    n_embd,
    attentionDotProductScale,
    n_head,
    inputBuffer,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    FastMLPBlock,
    SoftmaxBlock
  ) {
    const { resultBuffer: qkvMLPResult, passes: qkvMLPPasses } = FastMLPBlock.newInstance(
      seq_length,
      3 * n_embd,
      n_embd,
      inputBuffer,
      qkvWeightsBuffer,
      qkvBiasBuffer
    );

    const splitQKVPipeline = this.getSplitQKVPipeline();
    const splitQKVUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const splitQResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    const splitKResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    const splitVResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    const splitQKVBindGroup = this.initBindGroup(
      this.u_s_s_s_Layout,
      [splitQKVUniformBuffer, splitQResultBuffer, splitKResultBuffer, splitVResultBuffer],
      `${this.name}_SplitQKVG`
    );
    const splitQKVInputBindGroup = this.initBindGroup(this.r_Layout, [qkvMLPResult], `${this.name}_SplitQKVInputG`);
    this.device.queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));
    const splitQKVWorkgroups = { x: wgSize(n_embd, 16), y: wgSize(seq_length, 16), z: 1 };

    const attentionWeightsPipeline = this.getAttentionWeightsPipeline();
    const attentionWeightsUniformBuffer = this.initBuffer(["uniform", "copy_to"], [8]);
    const attentionWeightsResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, seq_length * n_head]);
    const attentionWeightsBindGroup = this.initBindGroup(
      this.u_s_Layout,
      [attentionWeightsUniformBuffer, attentionWeightsResultBuffer],
      `${this.name}_AttentionWeightsG`
    );
    const attentionWeightsInputBindGroup = this.initBindGroup(this.r_r_Layout, [splitQResultBuffer, splitKResultBuffer], `${this.name}_AttentionWeightsInputG`);
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head, seq_length, n_embd / n_head, n_embd]));
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 20, new Float32Array([attentionDotProductScale]));
    const attentionWeightsWorkgroups = { x: wgSize(seq_length * n_head, 16), y: wgSize(seq_length, 16), z: 1 };

    const causalMaskPipeline = this.getCausalMaskPipeline();
    const causalMaskUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const causalMaskResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, seq_length * n_head]);
    const causalMaskBindGroup = this.initBindGroup(this.u_s_Layout, [causalMaskUniformBuffer, causalMaskResultBuffer], `${this.name}_CausalMaskG`);
    const causalMaskInputBindGroup = this.initBindGroup(this.r_Layout, [attentionWeightsResultBuffer], `${this.name}_CausalMaskInputG`);
    this.device.queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length])); // Transposes! This is needed for softmax.
    const causalMaskWorkgroups = { x: wgSize(seq_length, 16), y: wgSize(seq_length * n_head, 16), z: 1 };

    const { resultBuffer: softmaxOutputBuffer, passes: softmaxPasses } = SoftmaxBlock.newInstance(seq_length * n_head, seq_length, causalMaskResultBuffer);

    const attentionValuesPipeline = this.getAttentionValuesPipeline();
    const attentionValuesUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const attentionValuesResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    const attentionValuesBindGroup = this.initBindGroup(this.u_s_Layout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    const attentionValuesInputBindGroup = this.initBindGroup(this.r_r_Layout, [softmaxOutputBuffer, splitVResultBuffer], `${this.name}_AttentionValuesInputG`);
    this.device.queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));
    const attentionValuesWorkgroups = { x: wgSize(n_embd, 16), y: wgSize(seq_length, 16), z: 1 };

    const { resultBuffer: linearMLPResult, passes: linearMLPPasses } = FastMLPBlock.newInstance(
      seq_length,
      n_embd,
      n_embd,
      attentionValuesResultBuffer,
      linearWeightsBuffer,
      linearBiasBuffer
    );

    return {
      resultBuffer: linearMLPResult,
      passes: [
        ...qkvMLPPasses,
        {
          flag: "compute",
          pipeline: splitQKVPipeline,
          groups: [splitQKVBindGroup, splitQKVInputBindGroup],
          workgroups: splitQKVWorkgroups,
        },
        {
          flag: "compute",
          pipeline: attentionWeightsPipeline,
          groups: [attentionWeightsBindGroup, attentionWeightsInputBindGroup],
          workgroups: attentionWeightsWorkgroups,
        },
        {
          flag: "compute",
          pipeline: causalMaskPipeline,
          groups: [causalMaskBindGroup, causalMaskInputBindGroup],
          workgroups: causalMaskWorkgroups,
        },
        ...softmaxPasses,
        {
          flag: "compute",
          pipeline: attentionValuesPipeline,
          groups: [attentionValuesBindGroup, attentionValuesInputBindGroup],
          workgroups: attentionValuesWorkgroups,
        },
        ...linearMLPPasses,
      ],
    };
  }

  splitQKVShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // row dimension of Q, K, V
      dimX: u32, // col dimension of Q, K, V
    };

    @group(1) @binding(0) var<storage, read> Input: Matrix;

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Q: Matrix;
    @group(0) @binding(2) var<storage, read_write> K: Matrix;
    @group(0) @binding(3) var<storage, read_write> V: Matrix;


    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Q.data[row * dimX + col] = Input.data[row * dimX * 3 + col];
      K.data[row * dimX + col] = Input.data[row * dimX * 3 + dimX + col];
      V.data[row * dimX + col] = Input.data[row * dimX * 3 + 2 * dimX + col];

    }
  `;

  attentionWeightsShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // output row dim, Q row dim
      dimX: u32, // output col dim, seq_length * heads
      seqLength: u32, // seq_length or K col dim (Q can be different)
      qkvCols: u32, // head col dim for Q, K or n_embd / n_heads
      embedDim: u32, // n_embd or total Q col dim & K row dim
      attentionScale: f32,
    };

    @group(1) @binding(0) var<storage, read> Queries: Matrix;
    @group(1) @binding(1) var<storage, read> Keys: Matrix;

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimY: u32 = DimBuffer.dimY;
      let dimX: u32 = DimBuffer.dimX;
      let seqLength: u32 = DimBuffer.seqLength;
      let qkvCols: u32 = DimBuffer.qkvCols;
      let embedDim: u32 = DimBuffer.embedDim;

      if (row >= dimY || col >= dimX) {
        return;
      }

      var head: u32 = col / seqLength;
      var col_r: u32 = col % seqLength;
      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < qkvCols; i = i + 1) {
          sum = sum + Queries.data[row * embedDim + i + head * qkvCols] * Keys.data[col_r * embedDim + i + head * qkvCols];
      }

      Result.data[row * dimX + col] = sum * DimBuffer.attentionScale;
    }
  `;

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
      if (row >= dimY || col > rowMask) {
        return;
      }

      let rowNum: u32 = row / dimX;
      Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];

    }
  `;

  attentionValuesShader = `
    struct Matrix {
      data: array<f32>,
    }

    struct Dimensions {
      dimY: u32, // Values row and col dimension, Weights row dimension (context)
      dimX: u32, // Result col dim (n_embd)
      numHeads: u32, // number of heads
      vCols: u32, // col dim of V
    };

    @group(1) @binding(0) var<storage, read> Weights: Matrix;
    @group(1) @binding(1) var<storage, read> Values: Matrix;

    @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
    @group(0) @binding(1) var<storage, read_write> Result: Matrix;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let dimY: u32 = DimBuffer.dimY;
      let dimX: u32 = DimBuffer.dimX;
      let vCols: u32 = DimBuffer.vCols;

      if (row >= dimY || col >= dimX) {
        return;
      }

      var head: u32 = col / vCols;
      var col_r: u32 = col % dimY;
      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < dimY; i = i + 1) {
          sum = sum +  Values.data[i * dimX + col] * Weights.data[row * dimY + i + head * dimY * dimY];
      }

      Result.data[row * dimX + col] = sum;
    }
  `;
}

class EmbedBlockClass extends Block {
  constructor() {
    super();
    this.name = "embed";
    this.pipelineCache = new Map();
  }

  newInstance(idx, seq_length, n_embd, embdBuffer, posEmbdBuffer, ResidualBlock) {
    const embdOutputBuffer = this.initBuffer(["storage", "copy_to"], [seq_length, n_embd]);
    const posEmbdOutputBuffer = this.initBuffer(["storage", "copy_to"], [seq_length, n_embd]);

    // Can build a cache later.
    const embdCopyCommands = Array(seq_length)
      .fill()
      .map((_, i) => {
        return {
          flag: "copy",
          src: embdBuffer,
          srcOffset: this.bufferSize(n_embd) * idx[i],
          dst: embdOutputBuffer,
          dstOffset: this.bufferSize(n_embd) * i,
          size: this.bufferSize(n_embd),
        };
      });

    // Also can be cached.
    const posCopyCommand = {
      flag: "copy",
      src: posEmbdBuffer,
      srcOffset: 0,
      dst: posEmbdOutputBuffer,
      dstOffset: 0,
      size: this.bufferSize(seq_length, n_embd),
    };

    const { resultBuffer: residualResult, passes: residualPasses } = ResidualBlock.newInstance(seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer);

    return {
      resultBuffer: residualResult,
      passes: [...embdCopyCommands, posCopyCommand, ...residualPasses],
    };
  }
}

class DeEmbedBlockClass extends Block {
  constructor() {
    super();
    this.name = "deembed";
    this.pipelineCache = new Map();
  }

  getPipeline() {
    const pipelineCacheKey = this.name; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.deEmbedShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(vocab_size, n_embd, seq_length, vocab_chunk_size, embedBuffer, deEmbeddingsBuffers) {
    const deEmbedPipeline = this.getPipeline();
    const slicedEmbedOutputBuffer = this.initBuffer(["storage", "copy_to"], [n_embd]);
    const deEmbedOutputBuffer = this.initBuffer(["map_read", "copy_to"], [vocab_size]);
    console.log(vocab_size, n_embd, seq_length, vocab_chunk_size, embedBuffer, deEmbeddingsBuffers);

    const sliceEmbedCopyCommand = {
      flag: "copy",
      src: embedBuffer,
      srcOffset: this.bufferSize(seq_length - 1, n_embd),
      dst: slicedEmbedOutputBuffer,
      dstOffset: 0,
      size: this.bufferSize(1, n_embd),
    };

    const deEmbedPasses = deEmbeddingsBuffers.flatMap((embdBuffer, i) => {
      // Some future optimizations where we can assume that vocab_size is consistent.
      // Each chunk must be divisible by 4.
      const padded_vocab_chunk_size = Math.ceil(vocab_chunk_size / 4) * 4;
      const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
      const resultBuffer = this.initBuffer(["storage", "copy_from"], [padded_vocab_chunk_size]);
      const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
      const inputBindGroup = this.initBindGroup(this.r_r_Layout, [slicedEmbedOutputBuffer, embdBuffer], `${this.name}_InputG`);
      const workgroups = { x: wgSize(padded_vocab_chunk_size, 64), y: 1, z: 1 };
      this.device.queue.writeBuffer(
        uniformBuffer,
        0,
        new Uint32Array([padded_vocab_chunk_size, Math.ceil(n_embd / 4), Math.ceil(padded_vocab_chunk_size / 4)])
      );

      return [
        {
          flag: "compute",
          pipeline: deEmbedPipeline,
          groups: [opBindGroup, inputBindGroup],
          workgroups,
        },
        {
          flag: "copy",
          src: resultBuffer,
          srcOffset: 0,
          dst: deEmbedOutputBuffer,
          dstOffset: i * this.bufferSize(vocab_chunk_size),
          size: this.bufferSize(vocab_chunk_size),
        },
      ];
    });

    console.log([sliceEmbedCopyCommand, ...deEmbedPasses]);

    return {
      resultBuffer: deEmbedOutputBuffer,
      passes: [sliceEmbedCopyCommand, ...deEmbedPasses],
    };
  }

  deEmbedShader = `
    struct uniforms {
      N: u32,
      MD4: u32,
      ND4: u32,
    }

    @group(1) @binding(0) var<storage,read> embed_vector: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_matrix: array<vec4<f32>>;
    @group(0) @binding(0) var<uniform> uniforms: uniforms;
    @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

    @compute @workgroup_size(4)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var colD16: u32 = global_id.x;
      var MD4: u32 = uniforms.MD4;
      var ND4: u32 = uniforms.ND4;
      var N: u32 = uniforms.N;
      
      // Prevent out of bounds access + uneven matrix N dim sizes.
      if (colD16 * 16 >= N) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>(0.0);
      var sum01: vec4<f32> = vec4<f32>(0.0);
      var sum02: vec4<f32> = vec4<f32>(0.0);
      var sum03: vec4<f32> = vec4<f32>(1.0);

      for (var i: u32 = 0u; i < MD4; i = i + 1u) {
        var embedChunk = embed_vector[i];
 
        sum00 = sum00 + vec4<f32>(embedChunk.x) * array_matrix[(i * 4u + 0u) * ND4 + colD16 * 4 + 0u];
        sum01 = sum01 + vec4<f32>(embedChunk.x) * array_matrix[(i * 4u + 0u) * ND4 + colD16 * 4 + 1u];
        sum02 = sum02 + vec4<f32>(embedChunk.x) * array_matrix[(i * 4u + 0u) * ND4 + colD16 * 4 + 2u];
        sum03 = sum03 + vec4<f32>(embedChunk.x) * array_matrix[(i * 4u + 0u) * ND4 + colD16 * 4 + 3u];

        sum00 = sum00 + vec4<f32>(embedChunk.y) * array_matrix[(i * 4u + 1u) * ND4 + colD16 * 4 + 0u];
        sum01 = sum01 + vec4<f32>(embedChunk.y) * array_matrix[(i * 4u + 1u) * ND4 + colD16 * 4 + 1u];
        sum02 = sum02 + vec4<f32>(embedChunk.y) * array_matrix[(i * 4u + 1u) * ND4 + colD16 * 4 + 2u];
        sum03 = sum03 + vec4<f32>(embedChunk.y) * array_matrix[(i * 4u + 1u) * ND4 + colD16 * 4 + 3u];

        sum00 = sum00 + vec4<f32>(embedChunk.z) * array_matrix[(i * 4u + 2u) * ND4 + colD16 * 4 + 0u];
        sum01 = sum01 + vec4<f32>(embedChunk.z) * array_matrix[(i * 4u + 2u) * ND4 + colD16 * 4 + 1u];
        sum02 = sum02 + vec4<f32>(embedChunk.z) * array_matrix[(i * 4u + 2u) * ND4 + colD16 * 4 + 2u];
        sum03 = sum03 + vec4<f32>(embedChunk.z) * array_matrix[(i * 4u + 2u) * ND4 + colD16 * 4 + 3u];

        sum00 = sum00 + vec4<f32>(embedChunk.w) * array_matrix[(i * 4u + 3u) * ND4 + colD16 * 4 + 0u];
        sum01 = sum01 + vec4<f32>(embedChunk.w) * array_matrix[(i * 4u + 3u) * ND4 + colD16 * 4 + 1u];
        sum02 = sum02 + vec4<f32>(embedChunk.w) * array_matrix[(i * 4u + 3u) * ND4 + colD16 * 4 + 2u];
        sum03 = sum03 + vec4<f32>(embedChunk.w) * array_matrix[(i * 4u + 3u) * ND4 + colD16 * 4 + 3u];
      }

      if (colD16 * 4u + 0u < N) {
        array_output[colD16 * 4u + 0u] = sum00;
      }
      if (colD16 * 4u + 1u < N) {
        array_output[colD16 * 4u + 1u] = sum01;
      }
      if (colD16 * 4u + 2u < N) {
        array_output[colD16 * 4u + 2u] = sum02;
      }
      if (colD16 * 4u + 3u < N) {
        array_output[colD16 * 4u + 3u] = sum03;
      }
    }
  `;
}

class OldDeEmbedBlockClass extends Block {
  constructor() {
    super();
    this.name = "deembed";
    this.pipelineCache = new Map();
  }

  newInstance(vocab_size, n_embd, seq_length, embedBuffer, embeddingWeightsBuffer, NaiveMatMulBlock) {
    const slicedEmbedOutputBuffer = this.initBuffer(["storage", "copy_to"], [n_embd]);
    const deEmbedOutputBuffer = this.initBuffer(["map_read", "copy_to"], [vocab_size]);

    // Assumes that vocab_size has a decent least prime factor.
    const maxStorageBufferSize = this.device.limits.maxStorageBufferBindingSize;
    const totalElements = this.bufferSize(vocab_size, n_embd);
    var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
    if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
    var vocabChunkSize = vocab_size / numInstances;

    const chunkBuffers = Array(numInstances)
      .fill()
      .map((_, i) => {
        return this.initBuffer(["storage", "copy_to"], [n_embd, vocabChunkSize]);
      });

    const sliceEmbedCopyCommand = {
      flag: "copy",
      src: embedBuffer,
      srcOffset: this.bufferSize(seq_length - 1, n_embd),
      dst: slicedEmbedOutputBuffer,
      dstOffset: 0,
      size: this.bufferSize(1, n_embd),
    };

    const deEmbedPasses = chunkBuffers.flatMap((buffer, i) => {
      const { resultBuffer: matmulResult, passes: matmulPasses } = NaiveMatMulBlock.newInstance(vocabChunkSize, 1, n_embd, buffer, slicedEmbedOutputBuffer);
      // We're doing some buffer tricks here. Since slicedEmbedOutputBuffer is a row matrix, we can just pretend it's a column matrix without any changes to the way it's stored. We then multiply it by the transposed embeddingWeights chunk, resulting in a column vector which, once again, we can pretend is a row vector.

      return [
        {
          flag: "copy",
          src: embeddingWeightsBuffer,
          srcOffset: i * this.bufferSize(n_embd * vocabChunkSize),
          dst: buffer,
          dstOffset: 0,
          size: this.bufferSize(n_embd, vocabChunkSize),
        },
        ...matmulPasses,
        {
          flag: "copy",
          src: matmulResult,
          srcOffset: 0,
          dst: deEmbedOutputBuffer,
          dstOffset: i * this.bufferSize(vocabChunkSize),
          size: this.bufferSize(vocabChunkSize),
        },
      ];
    });

    return {
      resultBuffer: deEmbedOutputBuffer,
      passes: [sliceEmbedCopyCommand, ...deEmbedPasses],
    };
  }
}
