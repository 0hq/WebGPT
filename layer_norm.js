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

  @compute @workgroup_size(16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY) {
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
    var stdeviation: f32 = sqrt(variance);

    Result.data[row * 2] = mean;
    Result.data[row * 2 + 1] = stdeviation;
  }
  `;

async function layerNorm(rows, cols, input) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_size_X = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // STATS pipeline, will be reused.
  const statsBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const statsPipeline = createPipeline(device, createNormStatsShader(), [statsBindGroupLayout, inputBufferBindGroupLayout]);
  // NORM pipeline, will be reused.
  //   const normBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  //   const normPipeline = createPipeline(device, createNormShader(), [normBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

  const pipeline = statsPipeline;

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, statsBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_size_X));
  passEncoder.end();

  const outputBufferSize = bufferSizeCalc(rows, 2);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(statsResultBuffer, 0, readBuffer, 0, outputBufferSize);

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

  printMatrix(row, 2, new Float32Array(result));
})();

function printMatrix(rows, cols, array) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(Array.from(array.slice(i * cols, (i + 1) * cols)));
  }
  console.log(matrix);
  for (const row of matrix) {
    console.log(row);
  }
  return matrix;
}
