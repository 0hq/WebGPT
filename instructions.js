// --------------------- Instructions --------------------- //

// All have been vectorized, but not all have been optimized with shared memory.
// Soon: quantize to int8, possible int4.
// Tuning needed and more detailed benchmarking, but decent for now.

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
    const settings = rows % 4 !== 0 ? "withCheck" : "noCheck";
    const pipelineCacheKey = `${this.name}_${settings}}`;
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const kernel = this.fastMatMul(settings);
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_r_Layout], `${this.name}_Pipeline_${pipelineCacheKey}`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, inputBuffer, weightsBuffer, biasBuffer) {
    if (cols % 8 !== 0) throw new Error("Cols must be divisible by 16."); // Is this not 8? Or is it 16?
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_r_Layout, [inputBuffer, weightsBuffer, biasBuffer], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 8 * 8), y: wgSize(rows, 4 * 8) };
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

  fastMatMul(flag) {
    console.log(flag);
    const outputCode = {
      withCheck: `
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
    `,
      noCheck: `
      array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
      array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
      array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
      array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
      array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
    `,
    };

    return `
    struct Meta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;
    @group(1) @binding(2) var<storage,read> array_bias: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = uniforms.M;
      var N: u32 = uniforms.N;
      var ND4: u32 = uniforms.ND4;
      var KD4: u32 = uniforms.KD4;
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

      ${outputCode[flag]}
    }
  `;
  }
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
    if (cols % 4 !== 0) throw new Error("Cols must be divisible by 4.");
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

      if (row >= M || col >= ND4) {
        return;
      }
      
      let index = row * ND4 + col;
      result_array[index] =  layer_out_array[index] + residual_array[index];
    }
  `;
}

class LayerNormBlockClass extends Block {
  constructor() {
    super();
    this.name = "layerNorm";
    this.pipelineCache = new Map();
  }

  getStatsPipeline(workgroups) {
    const pipelineCacheKey = `${this.name}_stats_${workgroups}`;
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.normStatsShader(workgroups), [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_Stats`);
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
    if (cols % 4 !== 0) throw new Error("Cols must be divisible by 4.");

    const workgroupsX = cols > 4096 ? 256 : 64;
    const statsPipeline = this.getStatsPipeline(workgroupsX);
    const statsUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const statsResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, 2]);
    const statsBindGroup = this.initBindGroup(this.u_s_Layout, [statsUniformBuffer, statsResultBuffer], `${this.name}_BindGroup_stats`);
    const statsInputBindGroup = this.initBindGroup(this.r_Layout, [inputBuffer], `${this.name}_InputG`);
    const statsWorkgroups = { x: 1, y: wgSize(rows, 1), z: 1 };
    this.device.queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([cols]));

    const normPipeline = this.getNormPipeline();
    const normUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const normResultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const normBindGroup = this.initBindGroup(this.u_s_Layout, [normUniformBuffer, normResultBuffer], `${this.name}_BindGroup_norm`);
    const normInputBindGroup = this.initBindGroup(
      this.r_r_r_r_Layout,
      [inputBuffer, gammaBuffer, betaBuffer, statsResultBuffer],
      `${this.name}_InputBindGroup_norm`
    );
    this.device.queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, Math.ceil(cols / 4)]));
    const normWorkgroups = { x: wgSize(cols, 32), y: wgSize(rows, 8), z: 1 };

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

  // I'm pretty sure this breaks for col < workgroupSize? Needs testing.
  normStatsShader = (wg_size) => `
    struct Meta {
      N: u32, 
    };

    const wg_size: u32 = ${wg_size};

    @group(1) @binding(0) var<storage, read> input_array: array<f32>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;

    var<workgroup> row_mean: f32;
    var<workgroup> op_buffer: array<f32, ${wg_size}>;

    @compute @workgroup_size(${wg_size})
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let N: u32 = uniforms.N;

      // Condense.
      var threadSum: f32 = 0.0;
      for (var i: u32 = col; i < N; i = i + wg_size) {
        threadSum = threadSum + input_array[row * N + i];
      }
      op_buffer[col] = threadSum;
      workgroupBarrier();
      
      // Reduce to one value sum. Optimize with bit shifts.
      for (var i: u32 = wg_size >> 1; i > 0; i = i >> 1) {
        if (col < i) {
          op_buffer[col] = op_buffer[col] + op_buffer[col + i];
        }
        workgroupBarrier();
      }
      
      if (col == 0) {
        row_mean = op_buffer[0] / f32(N);
      }
      workgroupBarrier();

      // Condense.
      var threadVariance: f32 = 0.0;
      for (var i: u32 = col; i < N; i = i + wg_size) {
        threadVariance = threadVariance + pow(input_array[row * N + i] - row_mean, 2);
      }
      op_buffer[col] = threadVariance / f32(N);
      workgroupBarrier();
      
      // Reduce to one value sum. Optimize with bit shifts.
      for (var i: u32 = wg_size >> 1; i > 0; i = i >> 1) {
        if (col < i) {
          op_buffer[col] = op_buffer[col] + op_buffer[col + i];
        }
        workgroupBarrier();
      }
      
      var stdev: f32 = sqrt(op_buffer[0] + 1e-5);

      if (col == 0) {
        result_array[row * 2] = row_mean;
        result_array[row * 2 + 1] = stdev;
      }
    }
  `;

  normShader = `
    struct Meta {
      M: u32,
      ND4: u32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<vec4<f32>>;

    @group(1) @binding(0) var<storage, read> input_array: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read> gamma_param: array<vec4<f32>>;
    @group(1) @binding(2) var<storage, read> beta_param: array<vec4<f32>>;
    @group(1) @binding(3) var<storage, read> stats_param: array<f32>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      var col: u32 = global_id.x;
      var row: u32 = global_id.y;
      var ND4: u32 = uniforms.ND4;
      var M: u32 = uniforms.M;

      if (row >= M || col >= ND4) {
        return;
      }

      let mean = stats_param[row * 2];
      let stdev = stats_param[row * 2 + 1];
      let output = (input_array[row * ND4 + col] - mean) / stdev;
      let gamma = gamma_param[col];
      let beta = beta_param[col];
      let shift = gamma * output + beta;
      result_array[row * ND4 + col] = shift;
    }
  `;
}

class SoftmaxBlockClass extends Block {
  constructor() {
    super();
    this.name = "softmax";
    this.pipelineCache = new Map();
  }

  getFusedPipeline(workgroups, transpose) {
    const pipelineCacheKey = `${this.name}_fused_${workgroups}_${transpose}`;
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.fusedShader(workgroups, transpose), [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_Div`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, inputBuffer, transpose = false) {
    const workgroupsX = cols > 4096 ? 256 : 64;
    const fusedPipeline = this.getFusedPipeline(workgroupsX, transpose);

    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols], `${this.name}_ResultBuffer_`);
    const bindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_BindGroup_`);
    const inputBindGroup = this.initBindGroup(this.r_Layout, [inputBuffer], `${this.name}_BindGroup__Input`);
    const workgroups = { x: 1, y: wgSize(rows, 1), z: 1 };

    return {
      resultBuffer: resultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline: fusedPipeline,
          groups: [bindGroup, inputBindGroup],
          workgroups: workgroups,
        },
      ],
    };
  }

  /*
    Possible improvements: Vectorization? Modify input buffer to coalesce better?
  */
  fusedShader(wg_size, transpose) {
    const outputIndex = transpose ? "i * uniforms.M + row" : "row * N + i";
    return `
    struct Meta {
      M: u32,
      N: u32, 
    };

    const minFloat: f32 = -3.402823e+38f;
    const wg_size: u32 = ${wg_size};

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;
    @group(1) @binding(0) var<storage, read> input_array: array<f32>;

    var<workgroup> max_row: f32;
    var<workgroup> sum_row: f32;
    var<workgroup> op_buffer: array<f32, ${wg_size}>;

    @compute @workgroup_size(${wg_size})
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let N: u32 = uniforms.N;

      // Condense into 256 col max.
      var thread_max = minFloat;
      for (var i: u32 = col; i < N; i = i + wg_size) {
        thread_max = max(thread_max, input_array[row * N + i]);
      }
      if (col < N) {
        op_buffer[col] = thread_max;
      }
      workgroupBarrier();
      
      // Reduce to one value max. Optimize with bit shifts.
      var reductionSize: u32 = min(N, wg_size);
      for (var i: u32 = reductionSize >> 1; i > 0; i = reductionSize >> 1) {
        reductionSize = i + (reductionSize & 1); // Ensure odd numbers are rounded up.
        if (col < i) {
          op_buffer[col] = max(op_buffer[col], op_buffer[col + reductionSize]);
        }
        workgroupBarrier();
      }
      if (col == 0) {
        max_row = op_buffer[0];
      }
      workgroupBarrier();

      var threadSum: f32 = 0.0;
      for (var i: u32 = col; i < N; i = i + wg_size) {
        threadSum = threadSum + exp(input_array[row * N + i] - max_row);
      }
      op_buffer[col] = threadSum;
      workgroupBarrier();
      
      // Reduce to one value sum. Optimize with bit shifts.
      for (var i: u32 = wg_size >> 1; i > 0; i = i >> 1) {
        if (col < i) {
          op_buffer[col] = op_buffer[col] + op_buffer[col + i];
        }
        workgroupBarrier();
      }
      
      if (col == 0) {
        sum_row = op_buffer[0];
      }
      workgroupBarrier();

      for (var i: u32 = col; i < N; i = i + wg_size) {
        result_array[${outputIndex}] = exp(input_array[row * N + i] - max_row) / sum_row;
      }
    }
  `;
  }
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
    if (cols % 4 !== 0) throw new Error("Cols must be divisible by 4.");
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

class EmbedBlockClass extends Block {
  constructor() {
    super();
    this.name = "embed";
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

  newInstance(n_embd, vocab_size, padded_vocab_size, seq_length, vocab_chunk_size, embedBuffer, deEmbeddingsBuffers) {
    const deEmbedPipeline = this.getPipeline();
    const slicedEmbedOutputBuffer = this.initBuffer(["storage", "copy_to"], [n_embd]);
    const deEmbedOutputBuffer = this.initBuffer(["map_read", "copy_to"], [vocab_size]);

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
      const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
      const resultBuffer = this.initBuffer(["storage", "copy_from"], [vocab_chunk_size]);
      const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
      const inputBindGroup = this.initBindGroup(this.r_r_Layout, [slicedEmbedOutputBuffer, embdBuffer], `${this.name}_InputG`);
      const workgroups = { x: wgSize(vocab_chunk_size, 32), y: 1, z: 1 };
      this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([vocab_chunk_size, Math.ceil(vocab_chunk_size / 4), Math.ceil(n_embd / 4)]));

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
          size: i == deEmbeddingsBuffers.length - 1 ? this.bufferSize(vocab_chunk_size - (padded_vocab_size - vocab_size)) : this.bufferSize(vocab_chunk_size),
        },
      ];
    });

    return {
      resultBuffer: deEmbedOutputBuffer,
      passes: [sliceEmbedCopyCommand, ...deEmbedPasses],
    };
  }

  deEmbedShader = `
    struct Meta {
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> embed_vector: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> deembed_matrix: array<vec4<f32>>;
    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

    @compute @workgroup_size(4)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var N: u32 = uniforms.N;
      var ND4: u32 = uniforms.ND4;
      var KD4: u32 = uniforms.KD4;
      var colD8: u32 = global_id.x;

      if (colD8 * 8 >= N) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = embed_vector[k];
        var brow: vec4<f32>;

        brow = deembed_matrix[(k * 4u + 0u) * ND4 + colD8 * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;

        brow = deembed_matrix[(k * 4u + 0u) * ND4 + colD8 * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
       
        brow = deembed_matrix[(k * 4u + 1u) * ND4 + colD8 * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
       
        brow = deembed_matrix[(k * 4u + 1u) * ND4 + colD8 * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
       
        brow = deembed_matrix[(k * 4u + 2u) * ND4 + colD8 * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        
        brow = deembed_matrix[(k * 4u + 2u) * ND4 + colD8 * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
       
        brow = deembed_matrix[(k * 4u + 3u) * ND4 + colD8 * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;

        brow = deembed_matrix[(k * 4u + 3u) * ND4 + colD8 * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
      }

      if (colD8 * 8u + 0u < N) {
        array_output[colD8 * 2u + 0u] = sum00;
      }
      if (colD8 * 8u + 4u < N) {
        array_output[colD8 * 2u + 1u] = sum10;
      }
    }
  `;
}

// ------------------ Needs Optimization ------------------ //

class AttentionBlockClass extends Block {
  constructor() {
    super();
    this.name = "attention";
    this.pipelineCache = new Map();
  }

  getNewAttentionWeightsPipeline() {
    const pipelineCacheKey = `${this.name}_weights_fused`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.fusedWeightsShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_AttWeights`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getNewAttentionValuesPipeline() {
    const pipelineCacheKey = `${this.name}_values_fused`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.newAttentionValuesShader, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_AttValues`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  getFormatQPipeline() {
    const pipelineCacheKey = `${this.name}_format_q`; // No param optimization.
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const pipeline = this.initPipeline(this.formatQShader, [this.u_s_Layout, this.r_Layout], `${this.name}_Pipeline_formatQ`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newFusedInstance(
    seq_length,
    n_embd,
    attentionDotProductScale,
    n_head,
    head_size,
    inputBuffer,
    qWeightsBuffer,
    qBiasBuffer,
    kWeightsBuffer,
    kBiasBuffer,
    vWeightsBuffer,
    vBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    FastMatMulBlock,
    SoftmaxBlock
  ) {
    const { resultBuffer: QResultBuffer, passes: QMLPPasses } = FastMatMulBlock.newInstance(
      seq_length,
      n_embd,
      n_embd,
      inputBuffer,
      qWeightsBuffer,
      qBiasBuffer
    );
    const { resultBuffer: KResultBuffer, passes: KMLPPasses } = FastMatMulBlock.newInstance(
      seq_length,
      n_embd,
      n_embd,
      inputBuffer,
      kWeightsBuffer,
      kBiasBuffer
    );
    const { resultBuffer: VResultBuffer, passes: VMLPPasses } = FastMatMulBlock.newInstance(
      seq_length,
      n_embd,
      n_embd,
      inputBuffer,
      vWeightsBuffer,
      vBiasBuffer
    );

    const formatQPipeline = this.getFormatQPipeline();
    const formatQUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const formatQResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length * n_head, head_size]);
    const formatQBindGroup = this.initBindGroup(this.u_s_Layout, [formatQUniformBuffer, formatQResultBuffer]);
    const formatQInputBindGroup = this.initBindGroup(this.r_Layout, [QResultBuffer], `${this.name}_formatQInputG`);
    this.device.queue.writeBuffer(formatQUniformBuffer, 0, new Uint32Array([seq_length, n_embd, head_size]));
    const formatQWorkgroups = { x: wgSize(n_embd, 8), y: wgSize(seq_length, 8), z: 1 };

    // const transposePipeline = this.getTransposePipeline();
    // const transposeUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    // const transposeResultBuffer = this.initBuffer(["storage", "copy_from"], [n_embd, seq_length]);
    // const transposeBindGroup = this.initBindGroup(this.u_s_Layout, [transposeUniformBuffer, transposeResultBuffer]);
    // const transposeInputBindGroup = this.initBindGroup(this.r_Layout, [KResultBuffer], `${this.name}_transposeInputG`);
    // this.device.queue.writeBuffer(transposeUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));
    // const transposeWorkgroups = { x: wgSize(n_embd, 8), y: wgSize(seq_length, 8), z: 1 };

    const attentionWeightsPipeline = this.getNewAttentionWeightsPipeline();
    const attentionWeightsUniformBuffer = this.initBuffer(["uniform", "copy_to"], [8]);
    const attentionWeightsResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length * n_head, seq_length]);
    const attentionWeightsBindGroup = this.initBindGroup(
      this.u_s_Layout,
      [attentionWeightsUniformBuffer, attentionWeightsResultBuffer],
      `${this.name}_AttentionWeightsG`
    );
    const attentionWeightsInputBindGroup = this.initBindGroup(this.r_r_Layout, [formatQResultBuffer, KResultBuffer], `${this.name}_AttentionWeightsInputG`);
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length, n_embd / 4, head_size / 4]));
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 16, new Float32Array([attentionDotProductScale]));
    const attentionWeightsWorkgroups = { x: wgSize(seq_length, 8), y: wgSize(seq_length * n_head, 8), z: 1 };

    const { resultBuffer: softmaxOutputBuffer, passes: softmaxPasses } = SoftmaxBlock.newInstance(
      seq_length * n_head,
      seq_length,
      attentionWeightsResultBuffer
    );

    // const attentionValuesPipeline = this.getAttentionValuesPipeline();
    // const attentionValuesUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    // const attentionValuesResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    // const attentionValuesBindGroup = this.initBindGroup(this.u_s_Layout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    // const attentionValuesInputBindGroup = this.initBindGroup(this.r_r_Layout, [softmaxOutputBuffer, VResultBuffer], `${this.name}_AttentionValuesInputG`);
    // this.device.queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, head_size]));
    // const attentionValuesWorkgroups = { x: wgSize(n_embd, 16), y: wgSize(seq_length, 16), z: 1 };

    const attentionValuesPipeline = this.getNewAttentionValuesPipeline();
    const attentionValuesUniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const attentionValuesResultBuffer = this.initBuffer(["storage", "copy_from"], [seq_length, n_embd]);
    const attentionValuesBindGroup = this.initBindGroup(this.u_s_Layout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    const attentionValuesInputBindGroup = this.initBindGroup(this.r_r_Layout, [softmaxOutputBuffer, VResultBuffer], `${this.name}_AttentionValuesInputG`);
    this.device.queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd / 4, head_size / 4]));
    const attentionValuesWorkgroups = { x: wgSize(n_embd, 32), y: wgSize(seq_length, 8), z: 1 };

    const { resultBuffer: linearMLPResult, passes: linearMLPPasses } = FastMatMulBlock.newInstance(
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
        ...QMLPPasses,
        ...KMLPPasses,
        ...VMLPPasses,
        {
          flag: "compute",
          pipeline: formatQPipeline,
          groups: [formatQBindGroup, formatQInputBindGroup],
          workgroups: formatQWorkgroups,
        },
        {
          flag: "compute",
          pipeline: attentionWeightsPipeline,
          groups: [attentionWeightsBindGroup, attentionWeightsInputBindGroup],
          workgroups: attentionWeightsWorkgroups,
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

  formatQShader = `
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

  transposeShader = `
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
        
      // Write the transposed tile to result_array. Flips dims.
      if (tile_row < M && tile_col < N) {
        result_array[tile_col * M + tile_row] = tile[local_id.x][local_id.y]; 
      }
    }
  `;

  // Sequence length invariant, no padding needed.
  fusedWeightsShader = `
    struct Meta {
      M: u32, // seq_length * n_heads
      N: u32, // seq_length
      ED4: u32, // hsize * n_heads
      HD4: u32,
      attentionScale: f32,
    };

    @group(1) @binding(0) var<storage, read> query_array: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read> key_array: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let col: u32 = global_id.x;
      let row: u32 = global_id.y;
      let N: u32 = uniforms.N;
      let HD4: u32 = uniforms.HD4;

      if (row >= uniforms.M || col >= N) {
        return;
      }

      let head: u32 = row / N;
      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < HD4; i = i + 1) {
        sum = sum + dot(query_array[row * HD4 + i], key_array[col * uniforms.ED4 + i + head * HD4]);
      }

      // Causal attention step.
      let rowMask: u32 = row % N;
      let causalMask: bool = (col <= rowMask);
      result_array[row * N + col] = select(-1e9, sum * uniforms.attentionScale, causalMask);
    }
  `;

  newAttentionValuesShader = `
    struct Meta {
      M: u32,
      ND4: u32,
      HD4: u32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<vec4<f32>>;

    @group(1) @binding(0) var<storage, read> weights_array: array<f32>;
    @group(1) @binding(1) var<storage, read> values_array: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      var col: u32 = global_id.x;
      var row: u32 = global_id.y;
      var M: u32 = uniforms.M;
      var ND4: u32 = uniforms.ND4;
      var HD4: u32 = uniforms.HD4;

      if (row >= M || col >= ND4) {
        return;
      }

      let head: u32 = col / HD4;
      var sum: vec4<f32> = vec4<f32>(0.0);
      for (var i: u32 = 0; i < M; i = i + 1) {
        var weight = weights_array[row * M + i + head * M * M]; // weights is M * M
        sum = sum + values_array[i * ND4 + col] * weight;
      }

      result_array[row * ND4 + col] = sum;
    }
  `;
}
