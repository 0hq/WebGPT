const createNormStatsShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
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
      data: array<f32>, // runtime-sized array
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
      let gamma = Gamma.data[row * 2];
      let beta = Beta.data[row * 2];
      let shift = gamma * output + beta;
      Result.data[row * dimX + col] = shift;
    } 
  `;

const createNormShader = () => `
  struct Matrix {
      data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read> Params: Matrix;
  @group(0) @binding(2) var<storage, read_write> Result: Matrix;
  
  @group(1) @binding(0) var<storage, read> Input: Matrix;
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
      let gamma = Params.data[row * 2];
      let beta = Params.data[row * 2 + 1];
      let shift = gamma * output + beta;
      Result.data[row * dimX + col] = shift;
    } 
  `;

async function layerNorm(rows, cols, input, gamma, beta) {
  const { device, queue } = await initializeWebGPU();
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
  const normBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "storage"]);
  const normPipeline = createPipeline(device, createNormShader(), [normBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normParamsBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, normBindGroupLayout, [normUniformBuffer, normParamsBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, cols]));
  queue.writeBuffer(normParamsBuffer, 0, new Float32Array(gamma.flatMap((gamma, i) => [gamma, beta[i]])));

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder_stats = commandEncoder.beginComputePass();
  passEncoder_stats.setPipeline(statsPipeline);
  passEncoder_stats.setBindGroup(0, statsBindGroup);
  passEncoder_stats.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_stats.dispatchWorkgroups(workgroupCalc(rows, workgroup_stats_Y));
  passEncoder_stats.end();

  const passEncoder_norm = commandEncoder.beginComputePass();
  passEncoder_norm.setPipeline(normPipeline);
  passEncoder_norm.setBindGroup(0, normBindGroup);
  passEncoder_norm.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_norm.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [statsResultBuffer]));
  passEncoder_norm.dispatchWorkgroups(workgroupCalc(rows, workgroup_norm_Y), workgroupCalc(cols, workgroup_norm_X));
  passEncoder_norm.end();

  const outputBufferSize = bufferSizeCalc(rows, cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(normResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  return readBuffer.getMappedRange();
}

function inlineLayerNorm(device, queue, commandEncoder, rows, cols, inputBuffer, gammaBuffer, betaBuffer) {
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
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, normBindGroupLayout, [normUniformBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, cols]));

  const passEncoder_stats = commandEncoder.beginComputePass();
  passEncoder_stats.setPipeline(statsPipeline);
  passEncoder_stats.setBindGroup(0, statsBindGroup);
  passEncoder_stats.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_stats.dispatchWorkgroups(workgroupCalc(rows, workgroup_stats_Y));
  passEncoder_stats.end();

  const passEncoder_norm = commandEncoder.beginComputePass();
  passEncoder_norm.setPipeline(normPipeline);
  passEncoder_norm.setBindGroup(0, normBindGroup);
  passEncoder_norm.setBindGroup(1, createBindGroup(device, normInputBindGroupLayout, [inputBuffer, gammaBuffer, betaBuffer]));
  passEncoder_norm.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [statsResultBuffer]));
  passEncoder_norm.dispatchWorkgroups(workgroupCalc(rows, workgroup_norm_Y), workgroupCalc(cols, workgroup_norm_X));
  passEncoder_norm.end();

  return normResultBuffer;
}

// (async () => {
//   const row = 10;
//   const col = 20;
//   const input = new Float32Array(row * col);
//   for (let i = 0; i < row * col; i++) input[i] = i;
//   const gamma = new Array(row).fill(1);
//   const beta = new Array(row).fill(0);

//   printMatrix(row, col, input);

//   const result = await layerNorm(row, col, input, gamma, beta);

//   const mat = printMatrix(row, col, new Float32Array(result));
//   // for (const row of mat) {
//   //   console.log(row.reduce((a, b) => a + b) / row.length);
//   //   console.log(getStandardDeviation(row));
//   // }
// })();

function printMatrix(rows, cols, array) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(Array.from(array.slice(i * cols, (i + 1) * cols)));
  }
  console.log(matrix);
  return matrix;
}

function getStandardDeviation(array) {
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
}
