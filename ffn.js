/*
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}

*/

// You forgot to write the shaders at all so you gotta write those then set the binding gropus after.

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
  const { device, queue } = await initializeWebGPU();

  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
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

  // Randomize the input data
  const inputLayer = {
    data: new Float32Array(contextSize * inputLayerSize),
    weights: new Float32Array(inputLayerSize * hiddenLayerSize),
    bias: new Float32Array(hiddenLayerSize),
  };
  const hiddenLayer = {
    weights: new Float32Array(hiddenLayerSize * outputLayerSize),
    bias: new Float32Array(outputLayerSize),
  };
  for (let i = 0; i < contextSize * inputLayerSize; i++) {
    inputLayer.data[i] = Math.random();
  }
  for (let i = 0; i < inputLayerSize * hiddenLayerSize; i++) {
    inputLayer.weights[i] = Math.random();
  }
  for (let i = 0; i < hiddenLayerSize; i++) {
    inputLayer.bias[i] = Math.random();
  }
  for (let i = 0; i < hiddenLayerSize * outputLayerSize; i++) {
    hiddenLayer.weights[i] = Math.random();
  }
  for (let i = 0; i < outputLayerSize; i++) {
    hiddenLayer.bias[i] = Math.random();
  }

  const inputUniformBuffer = device.createBuffer({
    size: 16, // must be a multiple of 16
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const inputBuffer = device.createBuffer({
    size: bufferSizeCalc(contextSize, inputLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const inputLayerWeightsBuffer = device.createBuffer({
    size: bufferSizeCalc(inputLayerSize, hiddenLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const inputLayerBiasBuffer = device.createBuffer({
    size: bufferSizeCalc(hiddenLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const inputLayerResultBuffer = device.createBuffer({
    size: bufferSizeCalc(contextSize, hiddenLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const geluUniformBuffer = device.createBuffer({
    size: 16, // must be a multiple of 16
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const geluResultBuffer = device.createBuffer({
    size: bufferSizeCalc(contextSize, hiddenLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const hiddenLayerUniformBuffer = device.createBuffer({
    size: 16, // must be a multiple of 16
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const hiddenLayerWeightsBuffer = device.createBuffer({
    size: bufferSizeCalc(hiddenLayerSize, outputLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const hiddenLayerBiasBuffer = device.createBuffer({
    size: bufferSizeCalc(outputLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const hiddenLayerResultBuffer = device.createBuffer({
    size: bufferSizeCalc(contextSize, outputLayerSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  queue.writeBuffer(inputBuffer, 0, inputLayer.data);

  queue.writeBuffer(inputUniformBuffer, 0, new Uint32Array([contextSize, hiddenLayerSize, inputLayerSize]));
  queue.writeBuffer(inputLayerWeightsBuffer, 0, inputLayer.weights);
  queue.writeBuffer(inputLayerBiasBuffer, 0, inputLayer.bias);

  queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([contextSize, hiddenLayerSize]));

  queue.writeBuffer(hiddenLayerUniformBuffer, 0, new Uint32Array([contextSize, outputLayerSize, hiddenLayerSize]));
  queue.writeBuffer(hiddenLayerWeightsBuffer, 0, hiddenLayer.weights);
  queue.writeBuffer(hiddenLayerBiasBuffer, 0, hiddenLayer.bias);

  const inputBufferBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  const ffnBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const ffnShader = createFFNShader();
  const shaderModule = device.createShaderModule({
    code: ffnShader,
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ffnBindGroupLayout, inputBufferBindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // Bind groups for the feedforward step
  const hiddenLayerBindGroup = device.createBindGroup({
    layout: ffnBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputUniformBuffer } },
      { binding: 1, resource: { buffer: inputLayerWeightsBuffer } },
      { binding: 2, resource: { buffer: inputLayerBiasBuffer } },
      { binding: 3, resource: { buffer: inputLayerResultBuffer } },
    ],
  });

  const outputLayerBindGroup = device.createBindGroup({
    layout: ffnBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: hiddenLayerUniformBuffer } },
      { binding: 1, resource: { buffer: hiddenLayerWeightsBuffer } },
      { binding: 2, resource: { buffer: hiddenLayerBiasBuffer } },
      { binding: 3, resource: { buffer: hiddenLayerResultBuffer } },
    ],
  });

  // GELU activation function

  const geluShader = createGELUShader();
  const geluShaderModule = device.createShaderModule({
    code: geluShader,
  });
  const hiddenResultBufferBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  const geluBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const geluPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [geluBindGroupLayout, hiddenResultBufferBindGroupLayout],
  });
  const geluPipeline = device.createComputePipeline({
    layout: geluPipelineLayout,
    compute: {
      module: geluShaderModule,
      entryPoint: "main",
    },
  });
  const geluBindGroup = device.createBindGroup({
    layout: geluBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: geluUniformBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: geluResultBuffer,
        },
      },
    ],
  });

  const inputBufferBindGroup = device.createBindGroup({
    layout: inputBufferBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: inputBuffer } }],
  });

  const hiddenResultBufferBindGroup = device.createBindGroup({
    layout: hiddenResultBufferBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: inputLayerResultBuffer } }],
  });

  // const geluResultBufferBindGroupLayout = device.createBindGroupLayout({
  //   entries: [
  //     {
  //       binding: 0,
  //       visibility: GPUShaderStage.COMPUTE,
  //       buffer: { type: "read-only-storage" },
  //     },
  //   ],
  // });

  const geluResultBufferBindGroup = device.createBindGroup({
    layout: inputBufferBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: geluResultBuffer } }],
  });

  // const commandEncoder = device.createCommandEncoder();
  // const passEncoder = commandEncoder.beginComputePass();

  const readBuffer = device.createBuffer({
    size: bufferSizeCalc(contextSize, outputLayerSize),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder1 = commandEncoder.beginComputePass();
  const passEncoder2 = commandEncoder.beginComputePass();
  const passEncoder3 = commandEncoder.beginComputePass();

  // First linear transformation (expansion from n_embed to hidden_size)
  passEncoder1.setPipeline(pipeline);
  passEncoder1.setBindGroup(0, hiddenLayerBindGroup);
  passEncoder1.setBindGroup(1, inputBufferBindGroup);
  passEncoder1.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(hiddenLayerSize, workgroupSizeY));
  passEncoder1.end();

  // Apply GELU activation
  passEncoder2.setPipeline(geluPipeline);
  passEncoder2.setBindGroup(0, geluBindGroup); // Reuse the same bind group as input
  passEncoder2.setBindGroup(1, hiddenResultBufferBindGroup); // Use the result from the first linear transformation as input
  passEncoder2.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(hiddenLayerSize, workgroupSizeY));
  passEncoder2.end();

  // Second linear transformation (contraction back down to n_embed)
  passEncoder3.setPipeline(pipeline);
  passEncoder3.setBindGroup(0, outputLayerBindGroup);
  passEncoder3.setBindGroup(1, geluResultBufferBindGroup); // Use the result from GELU activation as input
  passEncoder3.dispatchWorkgroups(workgroupCalc(contextSize, workgroupSizeX), workgroupCalc(outputLayerSize, workgroupSizeY));
  passEncoder3.end();

  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(hiddenLayerResultBuffer, 0, readBuffer, 0, bufferSizeCalc(contextSize, outputLayerSize));

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readBuffer.getMappedRange();
  console.log("arrayBuffer", arrayBuffer);
}

(async () => {
  runEntireFFN();
})();
