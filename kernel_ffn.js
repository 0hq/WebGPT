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

async function callDynamicNetwork() {
  // const inputMatrix = [
  //   [1, 2, 3, 4, 5],
  //   [6, 7, 8, 9, 10],
  //   [11, 12, 13, 14, 15],
  //   [16, 17, 18, 19, 20],
  //   [21, 22, 23, 24, 25],
  // ];
  // const firstLayerWeightsMatrix = [
  //   [0.1, 0.2, 0.3, 0.4, 0.5],
  //   [0.6, 0.7, 0.8, 0.9, 0.1],
  //   [0.11, 0.12, 0.13, 0.14, 0.15],
  //   [0.16, 0.17, 0.18, 0.19, 0.2],
  //   [0.21, 0.22, 0.23, 0.24, 0.25],
  // ];
  // make identity matrix
  const firstLayerWeightsMatrix = Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => (i === j ? 1 : 0)));
  const firstLayerBiasMatrix = [[0, 0, 0, 0, 0]];

  const layers = [
    {
      layer_type: "FFN",
      rowDim: 1024,
      sharedDim: 768,
      colDim: 3072,
      weights: new Float32Array(768 * 3072).fill(0.1),
      bias: new Float32Array(3072).fill(0),
    },
    {
      layer_type: "GELU",
      rowDim: 1024,
      colDim: 3072,
    },
    {
      layer_type: "FFN",
      rowDim: 1024,
      sharedDim: 3072,
      colDim: 768,
      weights: new Float32Array(3072 * 768).fill(0.1),
      bias: new Float32Array(768).fill(0),
    },
  ];
  const workgroupX = 16;
  const workgroupY = 16;

  const inputMatrix = new Float32Array(1024 * 768).fill(0.1);
  const result = await dynamicFFNNetwork(layers, workgroupX, workgroupY, inputMatrix);

  console.log("Done with network:", result);
  const { rowDim, colDim } = layers[layers.length - 1];
  const resultArray = new Float32Array(result);
  const resultMatrix = [];
  for (let i = 0; i < rowDim; i++) {
    resultMatrix.push(Array.from(resultArray.slice(i * colDim, (i + 1) * colDim)));
  }
  console.log("Resulting matrix:", resultMatrix);
  // console.log("Matrix value: (row 0)", Array.from(resultMatrix[0]));
  // console.log("Matrix value: (row 0, elem 0)", resultMatrix[0][0]);
  // for (const row of resultMatrix) {
  //   console.log(row);
  // }
}

async function dynamicFFNNetwork(layers, workgroupX, workgroupY, input) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // FFN pipeline, will be reused.
  const ffnBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  const FFNpipeline = createPipeline(device, createFFNShader(), [ffnBindGroupLayout, inputBufferBindGroupLayout]);
  // GELU pipeline, will be reused.
  const geluBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const GELUpipeline = createPipeline(device, createGELUShader(), [geluBindGroupLayout, inputBufferBindGroupLayout]);

  const passes = [];
  let lastResultBuffer = null;

  console.log("Starting network");
  const inputBuffer = createBuffer(
    device,
    bufferSizeCalc(layers[0].rowDim, layers[0].sharedDim || layers[0].colDim),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  );
  queue.writeBuffer(inputBuffer, 0, input);
  lastResultBuffer = inputBuffer;

  for (let i = 0; i < layers.length; i++) {
    const { layer_type, rowDim, colDim, sharedDim, weights, bias } = layers[i];
    if (layer_type === "FFN") {
      const ffnUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const ffnWeightsBuffer = createBuffer(device, bufferSizeCalc(sharedDim, colDim), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const ffnBiasBuffer = createBuffer(device, bufferSizeCalc(colDim), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const ffnResultBuffer = createBuffer(device, bufferSizeCalc(rowDim, colDim), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const ffnBindGroup = createBindGroup(device, ffnBindGroupLayout, [ffnUniformBuffer, ffnBiasBuffer, ffnWeightsBuffer, ffnResultBuffer]);
      queue.writeBuffer(ffnUniformBuffer, 0, new Uint32Array([rowDim, colDim, sharedDim]));
      queue.writeBuffer(ffnWeightsBuffer, 0, weights);
      queue.writeBuffer(ffnBiasBuffer, 0, bias);

      passes.push({
        pipeline: FFNpipeline,
        bindGroups: [ffnBindGroup, createBindGroup(device, inputBufferBindGroupLayout, [lastResultBuffer])],
        rowDim: rowDim,
        colDim: colDim,
      });
      lastResultBuffer = ffnResultBuffer;
    } else if (layer_type === "GELU") {
      console.log("GELU", layers[i]);
      const geluUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const geluResultBuffer = createBuffer(device, bufferSizeCalc(rowDim, colDim), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const geluBindGroup = createBindGroup(device, geluBindGroupLayout, [geluUniformBuffer, geluResultBuffer]);
      queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([rowDim, colDim]));

      passes.push({
        pipeline: GELUpipeline,
        bindGroups: [geluBindGroup, createBindGroup(device, inputBufferBindGroupLayout, [lastResultBuffer])],
        rowDim: rowDim,
        colDim: colDim,
      });
      lastResultBuffer = geluResultBuffer;
    }
  }

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();
  for (let i = 0; i < passes.length; i++) {
    const pass = passes[i];
    console.log("Pass", i, pass);
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pass.pipeline);
    for (let j = 0; j < pass.bindGroups.length; j++) passEncoder.setBindGroup(j, pass.bindGroups[j]);
    passEncoder.dispatchWorkgroups(workgroupCalc(pass.rowDim, workgroupY), workgroupCalc(pass.colDim, workgroupX));
    passEncoder.end();
  }

  const outputCols = passes[passes.length - 1].colDim;
  const outputRows = passes[passes.length - 1].rowDim;
  const outputBufferSize = bufferSizeCalc(outputCols, outputRows);
  console.log("Output buffer size:", outputBufferSize, outputCols, outputRows);

  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(lastResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  return readBuffer.getMappedRange();
}

// (async () => {
//   callDynamicNetwork();
// })();
