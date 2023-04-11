/*
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}

*/

const createFFNShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension of A and row dimension of C
    dimX: u32, // col dimension of B and col dimension of C
    dimS: u32, // shared dimension of A and B
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read> Bias: Matrix;
  @group(0) @binding(2) var<storage, read> Weight: Matrix;
  @group(0) @binding(3) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;
    let dimS: u32 = DimBuffer.dimS;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimS; i = i + 1) {
        sum = sum + Input.data[row * dimS + i] * Weight.data[i * dimX + col];
    }

    Result.data[row * dimX + col] = sum + Bias.data[col];
  }
  `;

// There's tons of obvious ineffiencies here but I'm pushing them to after this is working.

const createGELUShader = () => `
  struct Matrix {
      data: array<f32>, // runtime-sized array
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

async function runEntireFFN() {
  const n_embd = 768;
  const contextSize = 1024;
  const inputLayerSize = n_embd;
  const hiddenLayerSize = n_embd * 4;
  const outputLayerSize = n_embd;
  const workgroupSizeX = 16;
  const workgroupSizeY = 16;
  const workgroupCalc = (dim, size) => Math.min(Math.ceil(dim / size), 256);
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  // This is all represented as matmuls.
  // Input as a matrix (contextSize, n_embd)
  // Hidden layer as a transformation matrix (n_embd, n_embd * 4) -> (contextSize, n_embd * 4)
  // Output layer as a transformation matrix (n_embd * 4, n_embd) -> (contextSize, n_embd)

  const inputLayer = {
    data: new Float32Array(contextSize * inputLayerSize).fill(1),
  };
  const hiddenLayer = {
    weights: new Float32Array(inputLayerSize * hiddenLayerSize).fill(1),
    bias: new Float32Array(hiddenLayerSize).fill(0),
  };
  const outputLayer = {
    weights: new Float32Array(hiddenLayerSize * outputLayerSize).fill(1),
    bias: new Float32Array(outputLayerSize).fill(0),
  };

  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;

  const inputBuffer = createBuffer(device, bufferSizeCalc(contextSize, inputLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  const hiddenLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const hiddenLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(inputLayerSize, hiddenLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const hiddenLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hiddenLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const hiddenLayerResultBuffer = createBuffer(device, bufferSizeCalc(contextSize, hiddenLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const geluUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const geluResultBuffer = createBuffer(device, bufferSizeCalc(contextSize, hiddenLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const outputLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const outputLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hiddenLayerSize, outputLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputLayerBiasBuffer = createBuffer(device, bufferSizeCalc(outputLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputLayerResultBuffer = createBuffer(device, bufferSizeCalc(contextSize, outputLayerSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Generic bind group for input buffer.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);

  // FFN pipeline
  const ffnBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  const FFNpipeline = createPipeline(device, createFFNShader(), [ffnBindGroupLayout, inputBufferBindGroupLayout]);

  // GELU pipeline
  const geluBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const GELUpipeline = createPipeline(device, createGELUShader(), [geluBindGroupLayout, inputBufferBindGroupLayout]);

  // Bind groups for use in FFN pipeline with shared buffers.
  const inputBufferBindGroup = createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]);
  const hiddenLayerBindGroup = createBindGroup(device, ffnBindGroupLayout, [
    hiddenLayerUniformBuffer,
    hiddenLayerWeightsBuffer,
    hiddenLayerBiasBuffer,
    hiddenLayerResultBuffer,
  ]);

  // Bind groups for use in GELU pipeline with shared buffers.
  const hiddenResultBufferBindGroup = createBindGroup(device, inputBufferBindGroupLayout, [hiddenLayerResultBuffer]);
  const geluBindGroup = createBindGroup(device, geluBindGroupLayout, [geluUniformBuffer, geluResultBuffer]);

  // Bind groups for use in FFN pipeline with shared buffers.
  const geluResultBufferBindGroup = createBindGroup(device, inputBufferBindGroupLayout, [geluResultBuffer]);
  const outputLayerBindGroup = createBindGroup(device, ffnBindGroupLayout, [
    outputLayerUniformBuffer,
    outputLayerWeightsBuffer,
    outputLayerBiasBuffer,
    outputLayerResultBuffer,
  ]);

  queue.writeBuffer(inputBuffer, 0, inputLayer.data);

  queue.writeBuffer(hiddenLayerUniformBuffer, 0, new Uint32Array([contextSize, hiddenLayerSize, inputLayerSize]));
  queue.writeBuffer(hiddenLayerWeightsBuffer, 0, hiddenLayer.weights);
  queue.writeBuffer(hiddenLayerBiasBuffer, 0, hiddenLayer.bias);

  queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([contextSize, hiddenLayerSize]));

  queue.writeBuffer(outputLayerUniformBuffer, 0, new Uint32Array([contextSize, outputLayerSize, hiddenLayerSize]));
  queue.writeBuffer(outputLayerWeightsBuffer, 0, outputLayer.weights);
  queue.writeBuffer(outputLayerBiasBuffer, 0, outputLayer.bias);

  const commandEncoder = device.createCommandEncoder();

  // First linear transformation (expansion from n_embed to hidden_size)
  const passEncoder1 = commandEncoder.beginComputePass();
  passEncoder1.setPipeline(FFNpipeline);
  passEncoder1.setBindGroup(0, hiddenLayerBindGroup);
  passEncoder1.setBindGroup(1, inputBufferBindGroup);
  passEncoder1.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(hiddenLayerSize, workgroupSizeY));
  passEncoder1.end();

  // Apply GELU activation
  const passEncoder2 = commandEncoder.beginComputePass();
  passEncoder2.setPipeline(GELUpipeline);
  passEncoder2.setBindGroup(0, geluBindGroup); // Reuse the same bind group as input
  passEncoder2.setBindGroup(1, hiddenResultBufferBindGroup); // Use the result from the first linear transformation as input
  passEncoder2.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(hiddenLayerSize, workgroupSizeY));
  passEncoder2.end();

  // Second linear transformation (contraction back down to n_embed)
  const passEncoder3 = commandEncoder.beginComputePass();
  passEncoder3.setPipeline(FFNpipeline);
  passEncoder3.setBindGroup(0, outputLayerBindGroup);
  passEncoder3.setBindGroup(1, geluResultBufferBindGroup); // Use the result from GELU activation as input
  passEncoder3.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(outputLayerSize, workgroupSizeY));
  passEncoder3.end();

  const readBuffer = createBuffer(device, bufferSizeCalc(contextSize, outputLayerSize), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(outputLayerResultBuffer, 0, readBuffer, 0, bufferSizeCalc(contextSize, outputLayerSize));

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readBuffer.getMappedRange();
  const resultArray = new Float32Array(arrayBuffer);
  console.log("arrayBuffer", arrayBuffer);
  const resultMatrix = [];
  for (let i = 0; i < contextSize; i++) {
    resultMatrix.push(resultArray.slice(i * outputLayerSize, (i + 1) * outputLayerSize));
  }
  console.log("resultMatrix", resultMatrix);
  console.log("resultMatrix (row 0, elem 0)", resultMatrix[0][0]);
}

(async () => {
  runEntireFFN();
})();
