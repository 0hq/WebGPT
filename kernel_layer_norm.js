const createNormStatsShader = () => `
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

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY || col >= 1) {
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

const createNormShaderInline = () => `
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
  @group(2) @binding(0) var<storage, read> Stats: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
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

function inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, inputBuffer, gammaBuffer, betaBuffer) {
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_stats_Y = 16; // Dictated by shader.
  const workgroup_norm_X = 16; // Dictated by shader.
  const workgroup_norm_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // STATS pipeline, will be reused.
  const statsBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const statsPipeline = createPipeline(device, createNormStatsShader(), [statsBindGroupLayout, inputBufferBindGroupLayout]);
  // NORM pipeline, will be reused.
  const normInputBindGroupLayout = createBindGroupLayout(device, ["read-only-storage", "read-only-storage", "read-only-storage"]);
  const normBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const normPipeline = createPipeline(device, createNormShaderInline(), [normBindGroupLayout, normInputBindGroupLayout, inputBufferBindGroupLayout]);

  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, normBindGroupLayout, [normUniformBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const passEncoder_stats = commandEncoder.beginComputePass();
  passEncoder_stats.setPipeline(statsPipeline);
  passEncoder_stats.setBindGroup(0, statsBindGroup);
  passEncoder_stats.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_stats.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_stats_Y));
  passEncoder_stats.end();

  const passEncoder_norm = commandEncoder.beginComputePass();
  passEncoder_norm.setPipeline(normPipeline);
  passEncoder_norm.setBindGroup(0, normBindGroup);
  passEncoder_norm.setBindGroup(1, createBindGroup(device, normInputBindGroupLayout, [inputBuffer, gammaBuffer, betaBuffer]));
  passEncoder_norm.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [statsResultBuffer]));
  passEncoder_norm.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_norm_Y), workgroupCalc(n_embd, workgroup_norm_X));
  passEncoder_norm.end();

  return normResultBuffer;
}
