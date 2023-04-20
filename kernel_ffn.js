// There's tons of obvious ineffiencies here but I'm pushing them to after this is working.

const createGELUShader = () => `
  struct Matrix {
      data: array<f32>, 
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  const SQRPI: f32 = 0.7978845608;
  fn gelu(x: f32) -> f32 {
    if (x < -10.0) {
      return 0.0;
    } else if (x > 10.0) {
      return x;
    } else {
      let cdf_approx: f32 = 0.5 * (1.0 + tanh(SQRPI * (x + 0.044715 * pow(x, 3))));
      return x * cdf_approx;
    }
  }

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = gelu(Input.data[row * dimX + col]);
    } 
  `;

function inlineFFN(
  device,
  queue,
  commandEncoder,
  context,
  n_embed,
  hidden_size,
  inputBuffer,
  firstLayerWeightsBuffer,
  firstLayerBiasBuffer,
  secondLayerWeightsBuffer,
  secondLayerBiasBuffer
) {
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // FFN pipeline, will be reused.
  const ffnBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  const FFNpipeline = createPipeline(device, createFFNShader(), [ffnBindGroupLayout, inputBufferBindGroupLayout]);
  // GELU pipeline, will be reused.
  const geluBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const GELUpipeline = createPipeline(device, createGELUShader(), [geluBindGroupLayout, inputBufferBindGroupLayout]);

  const firstLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const firstLayerResultBuffer = createBuffer(device, bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const firstLayerBindGroup = createBindGroup(device, ffnBindGroupLayout, [
    firstLayerUniformBuffer,
    firstLayerBiasBuffer,
    firstLayerWeightsBuffer,
    firstLayerResultBuffer,
  ]);
  queue.writeBuffer(firstLayerUniformBuffer, 0, new Uint32Array([context, hidden_size, n_embed]));

  const geluUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const geluResultBuffer = createBuffer(device, bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const geluBindGroup = createBindGroup(device, geluBindGroupLayout, [geluUniformBuffer, geluResultBuffer]);
  queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([context, hidden_size]));

  const secondLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const secondLayerResultBuffer = createBuffer(device, bufferSizeCalc(context, n_embed), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const secondLayerBindGroup = createBindGroup(device, ffnBindGroupLayout, [
    secondLayerUniformBuffer,
    secondLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerResultBuffer,
  ]);
  queue.writeBuffer(secondLayerUniformBuffer, 0, new Uint32Array([context, n_embed, hidden_size]));

  const passEncoder_first = commandEncoder.beginComputePass();
  passEncoder_first.setPipeline(FFNpipeline);
  passEncoder_first.setBindGroup(0, firstLayerBindGroup);
  passEncoder_first.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_first.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
  passEncoder_first.end();

  const passEncoder_gelu = commandEncoder.beginComputePass();
  passEncoder_gelu.setPipeline(GELUpipeline);
  passEncoder_gelu.setBindGroup(0, geluBindGroup);
  passEncoder_gelu.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [firstLayerResultBuffer]));
  passEncoder_gelu.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
  passEncoder_gelu.end();

  const passEncoder_second = commandEncoder.beginComputePass();
  passEncoder_second.setPipeline(FFNpipeline);
  passEncoder_second.setBindGroup(0, secondLayerBindGroup);
  passEncoder_second.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [geluResultBuffer]));
  passEncoder_second.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(n_embed, workgroup_X));
  passEncoder_second.end();

  return secondLayerResultBuffer;
}
