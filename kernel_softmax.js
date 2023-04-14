const createNegMaxShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    var max_buffer: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
      max_buffer = max(max_buffer, Input.data[row * dimX + i]);
    }

    Result.data[row] = -max_buffer;
  }
  `;

const createAddShader = () => `
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
  @group(2) @binding(0) var<storage, read> Constants: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] + Constants.data[row];
    } 
  `;

const createExpShader = () => `
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

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = exp(Input.data[row * dimX + col]);
    }
  `;

const createSumShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

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

    Result.data[row] = sum;
  }
  `;

const createDivideShader = () => `
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
  @group(2) @binding(0) var<storage, read> Divisors: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] / Divisors.data[row];
    } 
  `;

async function softmax(rows, cols, input) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  const operationBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  // MAX pipeline, will be reused.
  const maxPipeline = createPipeline(device, createNegMaxShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // ADD pipeline, will be reused.
  const addPipeline = createPipeline(device, createAddShader(), [operationBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);
  // EXP pipeline, will be reused.
  const expPipeline = createPipeline(device, createExpShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // SUM pipeline, will be reused.
  const sumPipeline = createPipeline(device, createSumShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // DIV pipeline, will be reused.
  const dividePipeline = createPipeline(device, createDivideShader(), [operationBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const dimUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

  const maxResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const maxBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, maxResultBuffer]);

  const addResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const addBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, addResultBuffer]);

  const expResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const expBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, expResultBuffer]);

  const sumResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const sumBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, sumResultBuffer]);

  const divResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const divBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, divResultBuffer]);

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder_max = commandEncoder.beginComputePass();
  passEncoder_max.setPipeline(maxPipeline);
  passEncoder_max.setBindGroup(0, maxBindGroup);
  passEncoder_max.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_max.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_max.end();

  const passEncoder_add = commandEncoder.beginComputePass();
  passEncoder_add.setPipeline(addPipeline);
  passEncoder_add.setBindGroup(0, addBindGroup);
  passEncoder_add.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_add.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [maxResultBuffer]));
  passEncoder_add.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_add.end();

  const passEncoder_exp = commandEncoder.beginComputePass();
  passEncoder_exp.setPipeline(expPipeline);
  passEncoder_exp.setBindGroup(0, expBindGroup);
  passEncoder_exp.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [addResultBuffer]));
  passEncoder_exp.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_exp.end();

  const passEncoder_sum = commandEncoder.beginComputePass();
  passEncoder_sum.setPipeline(sumPipeline);
  passEncoder_sum.setBindGroup(0, sumBindGroup);
  passEncoder_sum.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [expResultBuffer]));
  passEncoder_sum.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_sum.end();

  const passEncoder_div = commandEncoder.beginComputePass();
  passEncoder_div.setPipeline(dividePipeline);
  passEncoder_div.setBindGroup(0, divBindGroup);
  passEncoder_div.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [expResultBuffer]));
  passEncoder_div.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [sumResultBuffer]));
  passEncoder_div.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_div.end();

  const outputBufferSize = bufferSizeCalc(rows, cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(divResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  return readBuffer.getMappedRange();
}

function inlineSoftmax(device, queue, commandEncoder, rows, cols, inputBuffer) {
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  const operationBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  // MAX pipeline, will be reused.
  const maxPipeline = createPipeline(device, createNegMaxShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // ADD pipeline, will be reused.
  const addPipeline = createPipeline(device, createAddShader(), [operationBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);
  // EXP pipeline, will be reused.
  const expPipeline = createPipeline(device, createExpShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // SUM pipeline, will be reused.
  const sumPipeline = createPipeline(device, createSumShader(), [operationBindGroupLayout, inputBufferBindGroupLayout]);
  // DIV pipeline, will be reused.
  const dividePipeline = createPipeline(device, createDivideShader(), [operationBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);

  const dimUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

  const maxResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const maxBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, maxResultBuffer]);

  const addResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const addBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, addResultBuffer]);

  const expResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const expBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, expResultBuffer]);

  const sumResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const sumBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, sumResultBuffer]);

  const divResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const divBindGroup = createBindGroup(device, operationBindGroupLayout, [dimUniformBuffer, divResultBuffer]);

  console.log("Starting passes");

  const passEncoder_max = commandEncoder.beginComputePass();
  passEncoder_max.setPipeline(maxPipeline);
  passEncoder_max.setBindGroup(0, maxBindGroup);
  passEncoder_max.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_max.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_max.end();

  const passEncoder_add = commandEncoder.beginComputePass();
  passEncoder_add.setPipeline(addPipeline);
  passEncoder_add.setBindGroup(0, addBindGroup);
  passEncoder_add.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_add.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [maxResultBuffer]));
  passEncoder_add.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_add.end();

  const passEncoder_exp = commandEncoder.beginComputePass();
  passEncoder_exp.setPipeline(expPipeline);
  passEncoder_exp.setBindGroup(0, expBindGroup);
  passEncoder_exp.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [addResultBuffer]));
  passEncoder_exp.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_exp.end();

  const passEncoder_sum = commandEncoder.beginComputePass();
  passEncoder_sum.setPipeline(sumPipeline);
  passEncoder_sum.setBindGroup(0, sumBindGroup);
  passEncoder_sum.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [expResultBuffer]));
  passEncoder_sum.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_sum.end();

  const passEncoder_div = commandEncoder.beginComputePass();
  passEncoder_div.setPipeline(dividePipeline);
  passEncoder_div.setBindGroup(0, divBindGroup);
  passEncoder_div.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [expResultBuffer]));
  passEncoder_div.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [sumResultBuffer]));
  passEncoder_div.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_div.end();

  return divResultBuffer;
}

// (async () => {
//   const row = 10;
//   const col = 10;
//   const input = new Float32Array(row * col);
//   for (let i = 0; i < row * col; i++) input[i] = i;

//   printMatrix(row, col, input);

//   const result = await softmax(row, col, input);

//   const mat = printMatrix(row, col, new Float32Array(result));
//   for (const row of mat) {
//     console.log(row.reduce((a, b) => a + b));
//     // console.log(getStandardDeviation(row));
//   }
// })();

function printMatrix(rows, cols, array) {
  // console.log(array);
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
