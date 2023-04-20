async function initializeWebGPU() {
  if (!navigator.gpu) return console.error("WebGPU is not supported");
  return await (await navigator.gpu.requestAdapter()).requestDevice();
}

async function createMatMulPipeline(device) {
  const shaderModule = device.createShaderModule({ code: matMulKernel });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: Array(4)
      .fill(null)
      .map((_, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: i < 3 ? "storage" : "uniform" },
      })),
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" },
  });
}

const matMulKernel = `
  struct Matrix { data: array<f32>, }
  struct Uniforms { dimY: u32, dimX: u32, dimS: u32, };
  @group(0) @binding(0) var<storage, read_write> A: Matrix;
  @group(0) @binding(1) var<storage, read_write> B: Matrix;
  @group(0) @binding(2) var<storage, read_write> C: Matrix;
  @group(0) @binding(3) var<uniform> dimBuffer: Uniforms;
  @compute @workgroup_size(8, 8)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = dimBuffer.dimX;
      let dimY: u32 = dimBuffer.dimY;
      if (row >= dimY || col >= dimX) { return; }
      let dimS: u32 = dimBuffer.dimS;
      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < dimS; i = i + 1) { sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col]; }
      C.data[row * dimX + col] = sum;
  } 
`;

async function runMatMul(device, pipeline, A, B, verbose = false) {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const align = (size) => Math.ceil(size / device.limits.minStorageBufferOffsetAlignment) * device.limits.minStorageBufferOffsetAlignment;
  const bufferSize = (matrix) => align(matrix.length * matrix[0].length * Float32Array.BYTES_PER_ELEMENT);
  if (A[0].length !== B.length) throw new Error("Invalid matrix dimensions");
  const buffers = [
    { size: bufferSize(A), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { size: bufferSize(B), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { size: bufferSize([[...A], [...B]]), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC },
    { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
  ].map((desc) => device.createBuffer(desc));
  const flatten = (matrix) => matrix.reduce((acc, row) => acc.concat(row), []);
  device.queue.writeBuffer(buffers[0], 0, new Float32Array(flatten(A)));
  device.queue.writeBuffer(buffers[1], 0, new Float32Array(flatten(B)));
  device.queue.writeBuffer(buffers[3], 0, new Uint32Array([A.length, B[0].length, B.length]));
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  const [numWorkgroupsX, numWorkgroupsY] = [A.length, B[0].length].map((dim) => Math.min(Math.ceil(dim / 8), 256));
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
  passEncoder.end();
  const readBuffer = device.createBuffer({ size: buffers[2].size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  commandEncoder.copyBufferToBuffer(buffers[2], 0, readBuffer, 0, buffers[2].size);
  device.queue.submit([commandEncoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  return new Float32Array(readBuffer.getMappedRange());
}
