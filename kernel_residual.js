// Obviously super inefficient but i'll be optimizing later, just trying to get this working for now.

const createElementWiseAdditionShader = () => `
  struct Matrix {
      data: array<f32>, // runtime-sized array
  }

  struct Uniforms {
    dimY: u32, 
    dimX: u32, 
  };

  @group(2) @binding(0) var<storage, read> LayerOutput: Matrix;
  @group(1) @binding(0) var<storage, read> Residual: Matrix;

  @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = dimBuffer.dimX;
    let dimY: u32 = dimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Result.data[row * dimX + col] = LayerOutput.data[row * dimX + col] + Residual.data[row * dimX + col];
  } 
`;
async function layerResidual(rows, cols, layerOutput, residual) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // ADD pipeline, will be reused.
  const residualBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const residualPipeline = createPipeline(device, createElementWiseAdditionShader(), [
    residualBindGroupLayout,
    inputBufferBindGroupLayout,
    inputBufferBindGroupLayout,
  ]);

  console.log("Starting network");
  const layerOutputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(layerOutputBuffer, 0, layerOutput);
  const residualBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(residualBuffer, 0, residual);

  const residualUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const residualResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const residualBindGroup = createBindGroup(device, residualBindGroupLayout, [residualUniformBuffer, residualResultBuffer]);
  queue.writeBuffer(residualUniformBuffer, 0, new Uint32Array([rows, cols]));

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(residualPipeline);
  passEncoder.setBindGroup(0, residualBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [residualBuffer]));
  passEncoder.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [layerOutputBuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  const outputBufferSize = bufferSizeCalc(rows, cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(residualResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  return readBuffer.getMappedRange();
}

// (async () => {
//   const row = 10;
//   const col = 20;
//   const layerOutput = new Float32Array(row * col);
//   const residual = new Float32Array(row * col);
//   for (let i = 0; i < row * col; i++) {
//     layerOutput[i] = i;
//     residual[i] = i;
//   }

//   printMatrix(row, col, layerOutput);
//   printMatrix(row, col, residual);

//   const result = await layerResidual(row, col, layerOutput, residual);

//   const mat = printMatrix(row, col, new Float32Array(result));
//   // for (const row of mat) {
//   //   console.log(row.reduce((a, b) => a + b) / row.length);
//   //   console.log(getStandardDeviation(row));
//   // }
// })();

function getStandardDeviation(array) {
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
}

function inlineResidual(device, queue, commandEncoder, rows, cols, layerOutputBuffer, residualBuffer) {
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // ADD pipeline, will be reused.
  const residualBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const residualPipeline = createPipeline(device, createElementWiseAdditionShader(), [
    residualBindGroupLayout,
    inputBufferBindGroupLayout,
    inputBufferBindGroupLayout,
  ]);

  const residualUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const residualResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const residualBindGroup = createBindGroup(device, residualBindGroupLayout, [residualUniformBuffer, residualResultBuffer]);
  queue.writeBuffer(residualUniformBuffer, 0, new Uint32Array([rows, cols]));

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(residualPipeline);
  passEncoder.setBindGroup(0, residualBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [residualBuffer]));
  passEncoder.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [layerOutputBuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  return residualResultBuffer;
}
