const createFFNShader = () => `
  struct Matrix {
    data: array<f32>, 
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
      data: array<f32>, 
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
    data: array<f32>, 
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
    data: array<f32>, 
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
    data: array<f32>, 
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

const createMultiplyShader = () => `
  struct Matrix {
      data: array<f32>, 
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
    attentionScale: f32, 
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] * DimBuffer.attentionScale;
    } 
  `;

function inlineAttention(
  device,
  queue,
  commandEncoder,
  seq_length,
  n_embd,
  attentionDotProductScale,
  inputBuffer,
  n_head,
  qkvWeightsBuffer,
  qkvBiasBuffer,
  linearWeightsBuffer,
  linearBiasBuffer
) {
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  if (n_embd % n_head != 0) {
    throw new Error("cols must be divisible by n_head");
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

  // Multiply pipeline, can be reused.
  const multiplyBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const multiplyPipeline = createPipeline(device, createMultiplyShader(), [multiplyBindGroupLayout, inputBufferBindGroupLayout]);

  // Causal mask pipeline, can be reused.
  const causalMaskBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const causalMaskPipeline = createPipeline(device, createCausalMaskShader(), [causalMaskBindGroupLayout, inputBufferBindGroupLayout]);

  const qkvUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const qkvResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const qkvBindGroup = createBindGroup(device, ffnBindGroupLayout, [qkvUniformBuffer, qkvBiasBuffer, qkvWeightsBuffer, qkvResultBuffer]);
  queue.writeBuffer(qkvUniformBuffer, 0, new Uint32Array([seq_length, 3 * n_embd, n_embd]));

  const splitQKVUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const splitQResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitKResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitVResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitQKVBindGroup = createBindGroup(device, splitQKVBindGroupLayout, [
    splitQKVUniformBuffer,
    splitQResultBuffer,
    splitKResultBuffer,
    splitVResultBuffer,
  ]);
  queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const attentionWeightsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionWeightsResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionWeightsBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
  queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head, n_embd / n_head, n_embd]));

  const multiplyUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const multiplyResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const multiplyBindGroup = createBindGroup(device, multiplyBindGroupLayout, [multiplyUniformBuffer, multiplyResultBuffer]);
  queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head]));
  queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

  const causalMaskUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const causalMaskResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const causalMaskBindGroup = createBindGroup(device, causalMaskBindGroupLayout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
  queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length])); // Transposes! This is needed for softmax.

  const attentionValuesUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionValuesResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionValuesBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
  queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));

  const linearUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const linearResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const linearBindGroup = createBindGroup(device, ffnBindGroupLayout, [linearUniformBuffer, linearBiasBuffer, linearWeightsBuffer, linearResultBuffer]);
  queue.writeBuffer(linearUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_embd]));

  const passEncoder_qkv = commandEncoder.beginComputePass();
  passEncoder_qkv.setPipeline(FFNpipeline);
  passEncoder_qkv.setBindGroup(0, qkvBindGroup);
  passEncoder_qkv.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_qkv.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(3 * n_embd, workgroup_X));
  passEncoder_qkv.end();

  const passEncoder_splitQKV = commandEncoder.beginComputePass();
  passEncoder_splitQKV.setPipeline(splitQKVpipeline);
  passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
  passEncoder_splitQKV.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [qkvResultBuffer]));
  passEncoder_splitQKV.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_splitQKV.end();

  const passEncoder_attentionWeights = commandEncoder.beginComputePass();
  passEncoder_attentionWeights.setPipeline(attentionWeightsPipeline);
  passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
  passEncoder_attentionWeights.setBindGroup(1, createBindGroup(device, attentionInputBindGroupLayout, [splitQResultBuffer, splitKResultBuffer]));
  passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
  passEncoder_attentionWeights.end();

  const passEncoder_multiply = commandEncoder.beginComputePass();
  passEncoder_multiply.setPipeline(multiplyPipeline);
  passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
  passEncoder_multiply.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [attentionWeightsResultBuffer]));
  passEncoder_multiply.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
  passEncoder_multiply.end();

  const passEncoder_causalMask = commandEncoder.beginComputePass();
  passEncoder_causalMask.setPipeline(causalMaskPipeline);
  passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
  passEncoder_causalMask.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [multiplyResultBuffer]));
  passEncoder_causalMask.dispatchWorkgroups(workgroupCalc(seq_length * n_head, workgroup_Y), workgroupCalc(seq_length, workgroup_X));
  passEncoder_causalMask.end();

  // This is a sloppy-ish solution to the casual mask buffer being processed with every head at once. Obviously, this could be fixed if we just did this in a smarter way but I only realized you could do this at the end. Still learning WebGPU!
  const softmaxOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(seq_length, seq_length * n_head),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  for (let i = 0; i < n_head; i++) {
    const softmaxInputBuffer = createBuffer(
      device,
      bufferSizeCalc(seq_length, seq_length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    commandEncoder.copyBufferToBuffer(
      causalMaskResultBuffer,
      i * bufferSizeCalc(seq_length, seq_length),
      softmaxInputBuffer,
      0,
      bufferSizeCalc(seq_length, seq_length)
    );
    const softMaxResultBuffer = inlineSoftmax(device, queue, commandEncoder, seq_length, seq_length, softmaxInputBuffer);
    commandEncoder.copyBufferToBuffer(
      softMaxResultBuffer,
      0,
      softmaxOutputBuffer,
      i * bufferSizeCalc(seq_length, seq_length),
      bufferSizeCalc(seq_length, seq_length)
    );
  }

  const passEncoder_attentionValues = commandEncoder.beginComputePass();
  passEncoder_attentionValues.setPipeline(attentionValuesPipeline);
  passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
  passEncoder_attentionValues.setBindGroup(1, createBindGroup(device, attentionInputBindGroupLayout, [softmaxOutputBuffer, splitVResultBuffer]));
  passEncoder_attentionValues.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_attentionValues.end();

  const passEncoder_linear = commandEncoder.beginComputePass();
  passEncoder_linear.setPipeline(FFNpipeline);
  passEncoder_linear.setBindGroup(0, linearBindGroup);
  passEncoder_linear.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [attentionValuesResultBuffer]));
  passEncoder_linear.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_linear.end();

  return linearResultBuffer;
}
