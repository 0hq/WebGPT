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

// currently also transposes the matrix for copying
const createCausalMaskShader = () => `
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

      let rowMask: u32 = row % dimX;
      let rowNum: u32 = row / dimX;
      if (col > rowMask) {
        Result.data[row * dimX + col] = -1e9;
      } else {
        Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];
      }

    } 
  `;

const createSplitQKVShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension of Q, K, V
    dimX: u32, // col dimension of Q, K, V
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Q: Matrix;
  @group(0) @binding(2) var<storage, read_write> K: Matrix;
  @group(0) @binding(3) var<storage, read_write> V: Matrix;


  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;    

    if (row >= dimY || col >= dimX) {
      return;
    }

    Q.data[row * dimX + col] = Input.data[row * dimX * 3 + col];
    K.data[row * dimX + col] = Input.data[row * dimX * 3 + dimX + col];
    V.data[row * dimX + col] = Input.data[row * dimX * 3 + 2 * dimX + col];

  }
  `;

const createAttentionWeightsShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // output row and col dimension, Q & K row dimension (context)
    dimX: u32, // context * heads
    qkvCols: u32, // col dim of Q, K heads
    embedDim: u32, // embedding dimension
  };

  @group(1) @binding(0) var<storage, read> Queries: Matrix;
  @group(1) @binding(1) var<storage, read> Keys: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let qkvCols: u32 = DimBuffer.qkvCols;
    let embedDim: u32 = DimBuffer.embedDim;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / dimY;
    var col_r: u32 = col % dimY;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < qkvCols; i = i + 1) {
        sum = sum + Queries.data[row * embedDim + i + head * qkvCols] * Keys.data[col_r * embedDim + i + head * qkvCols];
    }

    Result.data[row * dimX + col] = sum;
  }
  `;

const createAttentionValuesShader = () => `
  struct Matrix {
    data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // Values row and col dimension, Weights row dimension (context)
    dimX: u32, // Result col dim (n_embd)
    numHeads: u32, // number of heads
    vCols: u32, // col dim of V
  };

  @group(1) @binding(0) var<storage, read> Weights: Matrix;
  @group(1) @binding(1) var<storage, read> Values: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let vCols: u32 = DimBuffer.vCols;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / vCols;
    var col_r: u32 = col % dimY;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimY; i = i + 1) {
        sum = sum +  Values.data[i * dimX + col] * Weights.data[row * dimY + i + head * dimY * dimY];
    }

    Result.data[row * dimX + col] = sum;
  }
  `;

async function attention(rows, cols, input, n_heads, qkv_weights, qkv_bias, linear_weights, linear_bias) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  if (cols % n_heads != 0) {
    throw new Error("cols must be divisible by n_heads");
  }

  // Generic bind group for input buffer, can be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);

  // FFN pipeline, can be reused.
  const ffnBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  const FFNpipeline = createPipeline(device, createFFNShader(), [ffnBindGroupLayout, inputBufferBindGroupLayout]);

  // Split QKV pipeline, can be reused.
  const splitQKVBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage", "storage", "storage"]);
  const splitQKVpipeline = createPipeline(device, createSplitQKVShader(), [splitQKVBindGroupLayout, inputBufferBindGroupLayout]);

  // Attention weights pipeline, can be reused.
  const attentionInputBindGroupLayout = createBindGroupLayout(device, ["read-only-storage", "read-only-storage"]);
  const attentionBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const attentionWeightsPipeline = createPipeline(device, createAttentionWeightsShader(), [attentionBindGroupLayout, attentionInputBindGroupLayout]);
  const attentionValuesPipeline = createPipeline(device, createAttentionValuesShader(), [attentionBindGroupLayout, attentionInputBindGroupLayout]);

  // Causal mask pipeline, can be reused.
  const causalMaskBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const causalMaskPipeline = createPipeline(device, createCausalMaskShader(), [causalMaskBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const qkvUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(cols, 3 * cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvResultBuffer = createBuffer(device, bufferSizeCalc(rows, 3 * cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const qkvBindGroup = createBindGroup(device, ffnBindGroupLayout, [qkvUniformBuffer, qkvBiasBuffer, qkvWeightsBuffer, qkvResultBuffer]);
  queue.writeBuffer(qkvUniformBuffer, 0, new Uint32Array([rows, 3 * cols, cols]));
  queue.writeBuffer(qkvWeightsBuffer, 0, qkv_weights);
  queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

  const splitQKVUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const splitQResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitKResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitVResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitQKVBindGroup = createBindGroup(device, splitQKVBindGroupLayout, [
    splitQKVUniformBuffer,
    splitQResultBuffer,
    splitKResultBuffer,
    splitVResultBuffer,
  ]);
  queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([rows, cols]));

  const attentionWeightsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionWeightsResultBuffer = createBuffer(device, bufferSizeCalc(rows, rows * n_heads), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionWeightsBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
  queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([rows, rows * n_heads, cols / n_heads, cols]));

  // TODO: Add divide the magic number before mask fill

  const causalMaskUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const causalMaskResultBuffer = createBuffer(device, bufferSizeCalc(rows, rows * n_heads), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const causalMaskBindGroup = createBindGroup(device, causalMaskBindGroupLayout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
  queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([rows * n_heads, rows])); // Transposes! This is needed for softmax.

  const attentionValuesUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionValuesResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionValuesBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
  queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([rows, cols, n_heads, cols / n_heads]));

  const linearUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(cols, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const linearBiasBuffer = createBuffer(device, bufferSizeCalc(cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const linearResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const linearBindGroup = createBindGroup(device, ffnBindGroupLayout, [linearUniformBuffer, linearBiasBuffer, linearWeightsBuffer, linearResultBuffer]);
  queue.writeBuffer(linearUniformBuffer, 0, new Uint32Array([rows, cols, cols]));
  queue.writeBuffer(linearWeightsBuffer, 0, linear_weights);
  queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder_qkv = commandEncoder.beginComputePass();
  passEncoder_qkv.setPipeline(FFNpipeline);
  passEncoder_qkv.setBindGroup(0, qkvBindGroup);
  passEncoder_qkv.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_qkv.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(3 * cols, workgroup_X));
  passEncoder_qkv.end();

  const passEncoder_splitQKV = commandEncoder.beginComputePass();
  passEncoder_splitQKV.setPipeline(splitQKVpipeline);
  passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
  passEncoder_splitQKV.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [qkvResultBuffer]));
  passEncoder_splitQKV.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_splitQKV.end();

  const passEncoder_attentionWeights = commandEncoder.beginComputePass();
  passEncoder_attentionWeights.setPipeline(attentionWeightsPipeline);
  passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
  passEncoder_attentionWeights.setBindGroup(1, createBindGroup(device, attentionInputBindGroupLayout, [splitQResultBuffer, splitKResultBuffer]));
  passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(rows * n_heads, workgroup_X));
  passEncoder_attentionWeights.end();

  const passEncoder_causalMask = commandEncoder.beginComputePass();
  passEncoder_causalMask.setPipeline(causalMaskPipeline);
  passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
  passEncoder_causalMask.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [attentionWeightsResultBuffer]));
  passEncoder_causalMask.dispatchWorkgroups(workgroupCalc(rows * n_heads, workgroup_Y), workgroupCalc(rows, workgroup_X));
  passEncoder_causalMask.end();

  const softmaxOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(rows, rows * n_heads),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  for (let i = 0; i < n_heads; i++) {
    const softmaxInputBuffer = createBuffer(device, bufferSizeCalc(rows, rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    commandEncoder.copyBufferToBuffer(causalMaskResultBuffer, i * bufferSizeCalc(rows, rows), softmaxInputBuffer, 0, bufferSizeCalc(rows, rows));
    const softMaxResultBuffer = inlineSoftmax(device, queue, commandEncoder, rows, rows, softmaxInputBuffer);
    commandEncoder.copyBufferToBuffer(softMaxResultBuffer, 0, softmaxOutputBuffer, i * bufferSizeCalc(rows, rows), bufferSizeCalc(rows, rows));
  }

  const passEncoder_attentionValues = commandEncoder.beginComputePass();
  passEncoder_attentionValues.setPipeline(attentionValuesPipeline);
  passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
  passEncoder_attentionValues.setBindGroup(1, createBindGroup(device, attentionInputBindGroupLayout, [softmaxOutputBuffer, splitVResultBuffer]));
  passEncoder_attentionValues.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_attentionValues.end();

  const passEncoder_linear = commandEncoder.beginComputePass();
  passEncoder_linear.setPipeline(FFNpipeline);
  passEncoder_linear.setBindGroup(0, linearBindGroup);
  passEncoder_linear.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [attentionValuesResultBuffer]));
  passEncoder_linear.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_linear.end();

  const output_rows = rows;
  const output_cols = cols;
  const outputBufferSize = bufferSizeCalc(output_rows, output_cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const otherBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const thirdBuffer = createBuffer(device, bufferSizeCalc(rows * n_heads, rows), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const VBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const QBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const KBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(attentionValuesResultBuffer, 0, readBuffer, 0, outputBufferSize);
  copyCommandEncoder.copyBufferToBuffer(linearResultBuffer, 0, otherBuffer, 0, bufferSizeCalc(rows, cols));
  copyCommandEncoder.copyBufferToBuffer(softmaxOutputBuffer, 0, thirdBuffer, 0, bufferSizeCalc(rows * n_heads, rows));
  copyCommandEncoder.copyBufferToBuffer(splitVResultBuffer, 0, VBuffer, 0, bufferSizeCalc(rows, cols));
  copyCommandEncoder.copyBufferToBuffer(splitQResultBuffer, 0, QBuffer, 0, bufferSizeCalc(rows, cols));
  copyCommandEncoder.copyBufferToBuffer(splitKResultBuffer, 0, KBuffer, 0, bufferSizeCalc(rows, cols));

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  await otherBuffer.mapAsync(GPUMapMode.READ);
  await thirdBuffer.mapAsync(GPUMapMode.READ);
  await QBuffer.mapAsync(GPUMapMode.READ);
  await KBuffer.mapAsync(GPUMapMode.READ);
  await VBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  const result = readBuffer.getMappedRange();
  const other = otherBuffer.getMappedRange();
  const third = thirdBuffer.getMappedRange();
  const Q = QBuffer.getMappedRange();
  const K = KBuffer.getMappedRange();
  const V = VBuffer.getMappedRange();
  printMatrix(output_rows, output_cols, new Float32Array(result));
  // printMatrix(rows, cols, new Float32Array(other));
  printMatrix(rows * n_heads, rows, new Float32Array(third));
  // printMatrix(rows, cols, new Float32Array(Q));
  // printMatrix(rows, cols, new Float32Array(K));
  printMatrix(rows, cols, new Float32Array(V));
  return result;
}

(async () => {
  const row = 12;
  const col = 24;
  const input = new Float32Array(row * col);
  for (let y = 0; y < row; y++) {
    for (let x = 0; x < col; x++) {
      input[y * col + x] = 0;
    }
  }
  const n_heads = 4;

  const qkv_bias = new Float32Array(col * 3);
  const qkv_weights = new Float32Array(col * 3 * col);
  for (let y = 0; y < col; y++) {
    for (let x = 0; x < col * 3; x++) {
      qkv_bias[x] = Math.floor((x * 2) / col);
      qkv_weights[y * col * 3 + x] = x * y;
    }
  }

  const linear_bias = new Float32Array(col).fill(0);
  const linear_weights = new Float32Array(col * col);
  for (let y = 0; y < col; y++) {
    for (let x = 0; x < col; x++) {
      if (x === y) linear_weights[y * col + x] = 1;
      else linear_weights[y * col + x] = 0;
    }
  }

  printMatrix(row, col, input);
  printMatrix(col, col * 3, qkv_weights);

  const result = await attention(row, col, input, n_heads, qkv_weights, qkv_bias, linear_weights, linear_bias);

  // for (let i = 0; i < n_heads; i++) {
  //   const sliced = result.slice(i * row * col * 3, (i + 1) * row * col * 3);
  //   const mat = printMatrix(row, col / n_heads, new Float32Array(sliced));
  // }
  // for (const row of mat) {
  //   console.log(row.reduce((a, b) => a + b));
  // console.log(getStandardDeviation(row));
  // }
})();

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
