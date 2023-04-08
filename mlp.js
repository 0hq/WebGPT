async function initializeWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU is not supported");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  return device;
}

async function createMatMulPipeline(device) {
  const shader = createMatMulShader(device);

  const shaderModule = device.createShaderModule({
    code: shader,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  return pipeline;
}

function createMatMulShader(device) {
  return `
    struct Matrix {
        data: array<f32>, // runtime-sized array
    }

    struct Uniforms {
      dimY: u32, // row dimension of A and row dimension of C
      dimX: u32, // col dimension of B and col dimension of C
      dimS: u32, // shared dimension of A and B
    };

    @group(0) @binding(0) var<storage, read_write> A: Matrix;
    @group(0) @binding(1) var<storage, read_write> B: Matrix;
    @group(0) @binding(2) var<storage, read_write> C: Matrix;
    @group(0) @binding(3) var<uniform> dimBuffer: Uniforms;

    @compute @workgroup_size(64)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row: u32 = global_id.x;
        let col: u32 = global_id.y;
        let dimX: u32 = dimBuffer.dimX;
        let dimY: u32 = dimBuffer.dimY;
        let dimS: u32 = dimBuffer.dimS;

        var sum: f32 = 0.0;
        for (var i: u32 = 0; i < dimS; i = i + 1) {
            sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col];
        }

        if (row < dimY && col < dimX) {
          C.data[row * dimX + col] = sum;
        }
      } 
  `;
}

async function runMatMul(device, pipeline, A, B, verbose = false) {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  // [row][col]
  const bufferSizeA = A.length * A[0].length * Float32Array.BYTES_PER_ELEMENT;
  const bufferSizeB = B.length * B[0].length * Float32Array.BYTES_PER_ELEMENT;
  const bufferSizeC = B[0].length * A.length * Float32Array.BYTES_PER_ELEMENT;

  // The col dimension of A must match the row dimension of B
  // Or A[0].length === B.length
  if (A[0].length !== B.length) throw new Error("Invalid matrix dimensions");
  const dim = B.length; // or B[0].length
  const masterDimA = A.length;
  const masterDimB = B[0].length;

  const bufferA = device.createBuffer({
    size: bufferSizeA,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferB = device.createBuffer({
    size: bufferSizeB,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bufferC = device.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniformBuffer = device.createBuffer({
    size: 16, // number of bytes, mult of 16
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const queue = device.queue;

  const flatten = (matrix) => {
    return matrix.reduce((acc, row) => acc.concat(row), []);
  };

  const flatA = new Float32Array(flatten(A));
  const flatB = new Float32Array(flatten(B));

  queue.writeBuffer(bufferA, 0, flatA);
  queue.writeBuffer(bufferB, 0, flatB);
  queue.writeBuffer(uniformBuffer, 0, new Uint32Array([masterDimA, masterDimB, dim]));

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferC } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const workgroupSizeX = 8;
  const workgroupSizeY = 8;
  const numWorkgroupsX = Math.ceil(masterDimA / workgroupSizeX);
  const numWorkgroupsY = Math.ceil(masterDimB / workgroupSizeY);

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(masterDimA, masterDimB, 1);
  passEncoder.end();

  const readBuffer = device.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, bufferSizeC);

  queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readBuffer.getMappedRange();
  // console.log("arrayBuffer", arrayBuffer);
  const resultArray = new Float32Array(arrayBuffer);

  if (verbose) {
    console.log("A", `(${A.length}x${A[0].length})`, A);
    console.log("B", `(${B.length}x${B[0].length})`, B);
    console.log("C (output)", `(${A.length}x${B[0].length})`);
    console.log("dim or dimS", dim);
    console.log("masterDimA or dimY", masterDimA);
    console.log("masterDimB or dimX", masterDimB);
    console.log("flatA", flatA);
    console.log("flatB", flatB);
    // console.log("arrayBuffer int", new Int32Array(arrayBuffer));
    const resultMatrix = [];
    for (let i = 0; i < A.length; i++) {
      resultMatrix.push(resultArray.slice(i * B[0].length, (i + 1) * B[0].length));
    }
    console.log("resultMatrix", resultMatrix);
  }

  return resultArray;
}

async function runMLP(input, weight) {
  const device = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);

  // let layerOutput = input;
  // for (let i = 0; i < weights.length; i++) {
  //   const weight = weights[i];
  //   // const bias = biases[i];

  //   // Perform matrix multiplication
  //   layerOutput = await runMatMul(device, pipeline, layerOutput, weight);

  //   // Add biases
  //   // for (let j = 0; j < layerOutput.length; j++) {
  //   //   layerOutput[j] += bias[j % bias.length];
  //   // }

  //   // Apply the activation function
  //   // if (activation === "relu") {
  //   //   layerOutput = layerOutput.map((value) => Math.max(0, value));
  //   // } else if (activation === "sigmoid") {
  //   //   layerOutput = layerOutput.map((value) => 1 / (1 + Math.exp(-value)));
  //   // }
  // }

  return await runMatMul(device, pipeline, input, weight);
}

(async () => {
  // const first = [[0, 1]];
  // const second = [
  //   [0, 1],
  //   [0, 0],
  // ];
  // const output = await runMLP(first, second);
  // console.log("MLP Output:", output);

  const device = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);
  testMatrixMultiplication(device, pipeline);
})();

function generateRandomMatrix(rows, cols) {
  const matrix = new Array(rows);
  for (let i = 0; i < rows; i++) {
    matrix[i] = new Array(cols);
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = Math.random() * 2 - 1;
    }
  }
  return matrix;
}

function matMulCPU(A, B) {
  const result = new Array(A.length);
  for (let i = 0; i < A.length; i++) {
    result[i] = new Array(B[0].length).fill(0);
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < A[0].length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

async function testMatrixMultiplication(device, pipeline) {
  const numTests = 100;
  const maxDimension = 200;

  for (let t = 0; t < numTests; t++) {
    const rowsA = Math.ceil(Math.random() * maxDimension);
    const colsA = Math.ceil(Math.random() * maxDimension);
    const rowsB = colsA;
    const colsB = Math.ceil(Math.random() * maxDimension);

    const A = generateRandomMatrix(rowsA, colsA);
    const B = generateRandomMatrix(rowsB, colsB);

    const gpuResult = await runMatMul(device, pipeline, A, B);
    const cpuResult = matMulCPU(A, B);

    const epsilon = 1e-3;
    let success = true;
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        if (Math.abs(cpuResult[i][j] - gpuResult[i * colsB + j]) > epsilon) {
          success = false;
          break;
        }
      }
      if (!success) break;
    }

    console.log(`Test ${t + 1}: ${success ? "PASSED" : "FAILED"}`);
    if (!success) {
      const gpuResult = await runMatMul(device, pipeline, A, B, true);

      console.log("CPU Result", cpuResult);
      console.log("GPU Result", formatAsMatrix(gpuResult, rowsA, colsB));
      break;
    }
  }
}

function formatAsMatrix(floatArray, dimA, dimB) {
  const numArray = Array.from(floatArray);
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(numArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}
