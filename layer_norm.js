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

const createNormShader = () => `
  struct Matrix {
      data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
    gamma: f32, // network parameter
    beta: f32, // network parameter
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  
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
      let shift = DimBuffer.gamma * output + DimBuffer.beta;
      Result.data[row * dimX + col] = shift;
    } 
  `;

async function layerNorm(rows, cols, input, gamma = 1, beta = 0) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_stats_X = 16; // Dictated by shader.
  const workgroup_norm_X = 16; // Dictated by shader.
  const workgroup_norm_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // STATS pipeline, will be reused.
  const statsBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const statsPipeline = createPipeline(device, createNormStatsShader(), [statsBindGroupLayout, inputBufferBindGroupLayout]);
  // NORM pipeline, will be reused.
  const normBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const normPipeline = createPipeline(device, createNormShader(), [normBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, normBindGroupLayout, [normUniformBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, cols]));
  queue.writeBuffer(normUniformBuffer, 8, new Float32Array([gamma, beta]));

  /*

  This throws an error because rows and cols are both uint32 but gamma and beta are float32.

  */

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder1 = commandEncoder.beginComputePass();
  passEncoder1.setPipeline(statsPipeline);
  passEncoder1.setBindGroup(0, statsBindGroup);
  passEncoder1.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder1.dispatchWorkgroups(workgroupCalc(rows, workgroup_stats_X));
  passEncoder1.end();

  const passEncoder2 = commandEncoder.beginComputePass();
  passEncoder2.setPipeline(normPipeline);
  passEncoder2.setBindGroup(0, normBindGroup);
  passEncoder2.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder2.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [statsResultBuffer]));
  passEncoder2.dispatchWorkgroups(workgroupCalc(rows, workgroup_norm_Y), workgroupCalc(cols, workgroup_norm_X));
  passEncoder2.end();

  const outputBufferSize = bufferSizeCalc(rows, cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(normResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  return readBuffer.getMappedRange();
}

(async () => {
  const row = 10;
  const col = 20;
  const input = new Float32Array(row * col);
  for (let i = 0; i < row * col; i++) {
    input[i] = i * 2 + 10;
  }
  printMatrix(row, col, input);
  const result = await layerNorm(row, col, input);

  printMatrix(row, col, new Float32Array(result));
})();

function printMatrix(rows, cols, array) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(Array.from(array.slice(i * cols, (i + 1) * cols)));
  }
  console.log(matrix);
  for (const row of matrix) {
    console.log(row.reduce((a, b) => a + b) / row.length);
    console.log(getStandardDeviation(row));
  }
  return matrix;
}

function getStandardDeviation(array) {
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
}
