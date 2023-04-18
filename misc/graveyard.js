async function benchmarkMatrixMultiplication(device, pipeline) {
  const numTests = 100;
  const maxDimension = 1000;
  const rowsA = 1;
  const colsA = maxDimension;
  const rowsB = colsA;
  const colsB = 1;

  const A = generateRandomMatrix(rowsA, colsA);
  const B = generateRandomMatrix(rowsB, colsB);

  console.log("Running matrix multiplication benchmark...");
  console.log(`Matrix A: ${rowsA}x${colsA}`);
  console.log(`Matrix B: ${rowsB}x${colsB}`);

  const start = performance.now();
  for (let t = 0; t < numTests; t++) {
    const gpuResult = await runMatMul(device, pipeline, A, B);

    // console.log(`Run ${t + 1}: DONE`);
  }
  console.log("DONE");

  const end = performance.now();
  console.log(`Time taken: ${end - start} ms`);
  console.log(`Time per run: ${(end - start) / numTests} ms`);
  console.log(`Runs per second: ${numTests / ((end - start) / 1000)}`);
  console.log(`Ops per second: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000)}`);
  console.log(`GFLOPS: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000) / 1e9}`);
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

/*


  transformer.wte.weight: torch.Size([65, 128])
transformer.wpe.weight: torch.Size([64, 128])
transformer.h.0.ln_1.weight: torch.Size([128])
transformer.h.0.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.0.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.0.ln_2.weight: torch.Size([128])
transformer.h.0.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.0.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.1.ln_1.weight: torch.Size([128])
transformer.h.1.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.1.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.1.ln_2.weight: torch.Size([128])
transformer.h.1.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.1.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.2.ln_1.weight: torch.Size([128])
transformer.h.2.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.2.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.2.ln_2.weight: torch.Size([128])
transformer.h.2.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.2.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.3.ln_1.weight: torch.Size([128])
transformer.h.3.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.3.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.3.ln_2.weight: torch.Size([128])
transformer.h.3.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.3.mlp.c_proj.weight: torch.Size([128, 512])
transformer.ln_f.weight: torch.Size([128])
lm_head.weight: torch.Size([65, 128])

With bias:

transformer.wte.weight: torch.Size([65, 64])
transformer.wpe.weight: torch.Size([64, 64])

transformer.h.0.ln_1.weight: torch.Size([64])
transformer.h.0.ln_1.bias: torch.Size([64])
transformer.h.0.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.0.attn.c_attn.bias: torch.Size([192])
transformer.h.0.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.0.attn.c_proj.bias: torch.Size([64])
transformer.h.0.ln_2.weight: torch.Size([64])
transformer.h.0.ln_2.bias: torch.Size([64])
transformer.h.0.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.0.mlp.c_fc.bias: torch.Size([256])
transformer.h.0.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.0.mlp.c_proj.bias: torch.Size([64])

transformer.h.1.ln_1.weight: torch.Size([64])
transformer.h.1.ln_1.bias: torch.Size([64])
transformer.h.1.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.1.attn.c_attn.bias: torch.Size([192])
transformer.h.1.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.1.attn.c_proj.bias: torch.Size([64])
transformer.h.1.ln_2.weight: torch.Size([64])
transformer.h.1.ln_2.bias: torch.Size([64])
transformer.h.1.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.1.mlp.c_fc.bias: torch.Size([256])
transformer.h.1.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.1.mlp.c_proj.bias: torch.Size([64])

transformer.ln_f.weight: torch.Size([64])
transformer.ln_f.bias: torch.Size([64])
lm_head.weight: torch.Size([65, 64])

    */

async function runGPTOld(
  device,
  queue,
  seq_length,
  vocab_size,
  n_embd,
  n_heads,
  n_layers,
  attentionDotProductScale,
  bufferSizeCalc,
  inputBuffer,
  embdBuffer,
  posEmbdBuffer,
  normAttentionGammaBuffer,
  normAttentionBetaBuffer,
  qkvWeightsBuffer,
  qkvBiasBuffer,
  linearWeightsBuffer,
  linearBiasBuffer,
  normLinearGammaBuffer,
  normLinearBetaBuffer,
  firstLayerWeightsBuffer,
  firstLayerBiasBuffer,
  secondLayerWeightsBuffer,
  secondLayerBiasBuffer,
  normGammaBuffer,
  normBetaBuffer,
  deEmbedBuffer
) {
  const commandEncoder = device.createCommandEncoder();
  let layerBuffer = inputBuffer;

  // COMPUTE

  const embdOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerBuffer, embdBuffer, seq_length, n_embd, vocab_size);

  // Crop the position embeddings to the correct size.
  const posEmbdOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(seq_length, n_embd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  commandEncoder.copyBufferToBuffer(
    posEmbdBuffer, // Source buffer (original position embeddings)
    0, // Source offset (starting from the beginning of the buffer)
    posEmbdOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(seq_length, n_embd) // Number of bytes to copy
  );

  // Residual connection is just elementwise addition, can be used for combining embedding and position embedding.
  const embeddedInputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer);
  layerBuffer = embeddedInputBuffer;

  for (let i = 0; i < n_layers; i++) {
    const blockOutputBuffer = transformerBlock(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      n_heads,
      attentionDotProductScale,
      layerBuffer,
      normAttentionGammaBuffer,
      normAttentionBetaBuffer,
      qkvWeightsBuffer,
      qkvBiasBuffer,
      linearWeightsBuffer,
      linearBiasBuffer,
      normLinearGammaBuffer,
      normLinearBetaBuffer,
      firstLayerWeightsBuffer,
      firstLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerBiasBuffer
    );
    layerBuffer = blockOutputBuffer;
  }

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

  const deEmbedOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerNormOutputBuffer, deEmbedBuffer, seq_length, vocab_size, n_embd);

  const outputBufferSize = bufferSizeCalc(seq_length, vocab_size);
  const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  commandEncoder.copyBufferToBuffer(deEmbedOutputBuffer, 0, outputBuffer, 0, outputBufferSize);
  queue.submit([commandEncoder.finish()]);

  await outputBuffer.mapAsync(GPUMapMode.READ);

  return outputBuffer.getMappedRange();
}

async function timeGPTOld() {
  const { device, queue } = await initializeWebGPU();
  const context_size = 1024;
  const seq_length = 24;
  const vocab_size = 50304;
  const n_embd = 768 / 2;
  const n_heads = 4;
  const n_layers = 12;
  const inputMatrix = new Float32Array(seq_length * vocab_size).fill(1);
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  const inputBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, inputMatrix);

  const embeddings = new Float32Array(vocab_size * n_embd).fill(-1);
  const embdBuffer = createBuffer(device, bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdBuffer, 0, embeddings);

  const posEmbeddings = new Float32Array(context_size * n_embd).fill(-1);
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  // Transformer Block Weights
  const layerNormAttentionGamma = new Array(n_embd).fill(1);
  const layerNormAttentionBeta = new Array(n_embd).fill(1);
  const qkv_bias = new Float32Array(n_embd * 3);
  const qkv_weights = new Float32Array(n_embd * 3 * n_embd);
  for (let y = 0; y < n_embd; y++) {
    for (let x = 0; x < n_embd * 3; x++) {
      qkv_bias[x] = 0.1;
      qkv_weights[y * n_embd * 3 + x] = 0.1;
    }
  }
  const linear_bias = new Float32Array(n_embd).fill(0);
  const linear_weights = new Float32Array(n_embd * n_embd);
  for (let y = 0; y < n_embd; y++) {
    for (let x = 0; x < n_embd; x++) {
      if (x === y) linear_weights[y * n_embd + x] = 0.1;
      else linear_weights[y * n_embd + x] = 0;
    }
  }
  const layerNormLinearGamma = new Float32Array(n_embd).fill(1);
  const layerNormLinearBeta = new Float32Array(n_embd).fill(0);
  const firstLayerWeights = new Float32Array(hidden_size * n_embd).fill(1);
  const firstLayerBias = new Float32Array(hidden_size).fill(1);
  const secondLayerWeights = new Float32Array(hidden_size * n_embd).fill(1);
  const secondLayerBias = new Float32Array(hidden_size).fill(1);

  const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normAttentionGammaBuffer, 0, new Float32Array(layerNormAttentionGamma));
  queue.writeBuffer(normAttentionBetaBuffer, 0, new Float32Array(layerNormAttentionBeta));

  const qkvWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(qkvWeightsBuffer, 0, qkv_weights);
  queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

  const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  queue.writeBuffer(linearWeightsBuffer, 0, linear_weights);
  queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

  const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
  queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

  const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(firstLayerWeightsBuffer, 0, firstLayerWeights);
  queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

  const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(secondLayerWeightsBuffer, 0, secondLayerWeights);
  queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

  const layerNormGamma = new Float32Array(seq_length).fill(1);
  const layerNormBeta = new Float32Array(seq_length).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(seq_length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  const deEmbeddings = new Float32Array(n_embd * vocab_size).fill(-1);
  const deEmbedBuffer = createBuffer(device, bufferSizeCalc(n_embd, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(deEmbedBuffer, 0, deEmbeddings);

  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  const startTime = performance.now();
  const result = await runGPTOld(
    device,
    queue,
    seq_length,
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    attentionDotProductScale,
    bufferSizeCalc,
    inputBuffer,
    embdBuffer,
    posEmbdBuffer,
    normAttentionGammaBuffer,
    normAttentionBetaBuffer,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    normLinearGammaBuffer,
    normLinearBetaBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer,
    normGammaBuffer,
    normBetaBuffer,
    deEmbedBuffer
  );
  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  printMatrix(seq_length, vocab_size, new Float32Array(result));
}

// timeGPTOld();

const createNormShader = () => `
  struct Matrix {
      data: array<f32>, // runtime-sized array
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read> Params: Matrix;
  @group(0) @binding(2) var<storage, read_write> Result: Matrix;
  
  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(2) @binding(0) var<storage, read> Stats: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      let mean = Stats.data[row * 2];
      let stdev = Stats.data[row * 2 + 1];
      let output = (Input.data[row * dimX + col] - mean) / stdev;
      let gamma = Params.data[row * 2];
      let beta = Params.data[row * 2 + 1];
      let shift = gamma * output + beta;
      Result.data[row * dimX + col] = shift;
    } 
  `;

async function layerNorm(rows, cols, input, gamma, beta) {
  const { device, queue } = await initializeWebGPU();
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_stats_Y = 16; // Dictated by shader.
  const workgroup_norm_X = 16; // Dictated by shader.
  const workgroup_norm_Y = 16; // Dictated by shader.

  // Generic bind group for input buffer, will be reused.
  const inputBufferBindGroupLayout = createBindGroupLayout(device, ["read-only-storage"]);
  // STATS pipeline, will be reused.
  const statsBindGroupLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  const statsPipeline = createPipeline(device, createNormStatsShader(), [statsBindGroupLayout, inputBufferBindGroupLayout]);
  // NORM pipeline, will be reused.
  const normBindGroupLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "storage"]);
  const normPipeline = createPipeline(device, createNormShader(), [normBindGroupLayout, inputBufferBindGroupLayout, inputBufferBindGroupLayout]);

  console.log("Starting network");
  const inputBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, input);

  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, statsBindGroupLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([rows, cols]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normParamsBuffer = createBuffer(device, bufferSizeCalc(rows, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, normBindGroupLayout, [normUniformBuffer, normParamsBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([rows, cols]));
  queue.writeBuffer(normParamsBuffer, 0, new Float32Array(gamma.flatMap((gamma, i) => [gamma, beta[i]])));

  console.log("Starting passes");
  const commandEncoder = device.createCommandEncoder();

  const passEncoder_stats = commandEncoder.beginComputePass();
  passEncoder_stats.setPipeline(statsPipeline);
  passEncoder_stats.setBindGroup(0, statsBindGroup);
  passEncoder_stats.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_stats.dispatchWorkgroups(workgroupCalc(rows, workgroup_stats_Y));
  passEncoder_stats.end();

  const passEncoder_norm = commandEncoder.beginComputePass();
  passEncoder_norm.setPipeline(normPipeline);
  passEncoder_norm.setBindGroup(0, normBindGroup);
  passEncoder_norm.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [inputBuffer]));
  passEncoder_norm.setBindGroup(2, createBindGroup(device, inputBufferBindGroupLayout, [statsResultBuffer]));
  passEncoder_norm.dispatchWorkgroups(workgroupCalc(rows, workgroup_norm_Y), workgroupCalc(cols, workgroup_norm_X));
  passEncoder_norm.end();

  const outputBufferSize = bufferSizeCalc(rows, cols);
  const readBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const copyCommandEncoder = device.createCommandEncoder();
  copyCommandEncoder.copyBufferToBuffer(normResultBuffer, 0, readBuffer, 0, outputBufferSize);

  queue.submit([commandEncoder.finish(), copyCommandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  console.log("Done!");
  return readBuffer.getMappedRange();
}

// (async () => {
//   const row = 10;
//   const col = 20;
//   const input = new Float32Array(row * col);
//   for (let i = 0; i < row * col; i++) input[i] = i;
//   const gamma = new Array(row).fill(1);
//   const beta = new Array(row).fill(0);

//   printMatrix(row, col, input);

//   const result = await layerNorm(row, col, input, gamma, beta);

//   const mat = printMatrix(row, col, new Float32Array(result));
//   // for (const row of mat) {
//   //   console.log(row.reduce((a, b) => a + b) / row.length);
//   //   console.log(getStandardDeviation(row));
//   // }
// })();

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

// (async () => {
//   const row = 12;
//   const col = 24;
//   const input = new Float32Array(row * col);
//   for (let y = 0; y < row; y++) {
//     for (let x = 0; x < col; x++) {
//       input[y * col + x] = 0;
//     }
//   }
//   const n_heads = 4;

//   const qkv_bias = new Float32Array(col * 3);
//   const qkv_weights = new Float32Array(col * 3 * col);
//   for (let y = 0; y < col; y++) {
//     for (let x = 0; x < col * 3; x++) {
//       qkv_bias[x] = Math.floor((x * 2) / col);
//       qkv_weights[y * col * 3 + x] = x * y;
//     }
//   }

//   const linear_bias = new Float32Array(col).fill(0);
//   const linear_weights = new Float32Array(col * col);
//   for (let y = 0; y < col; y++) {
//     for (let x = 0; x < col; x++) {
//       if (x === y) linear_weights[y * col + x] = 1;
//       else linear_weights[y * col + x] = 0;
//     }
//   }

//   printMatrix(row, col, input);
//   printMatrix(col, col * 3, qkv_weights);

//   const result = await attention(row, col, input, n_heads, qkv_weights, qkv_bias, linear_weights, linear_bias);

//   // for (let i = 0; i < n_heads; i++) {
//   //   const sliced = result.slice(i * row * col * 3, (i + 1) * row * col * 3);
//   //   const mat = printMatrix(row, col / n_heads, new Float32Array(sliced));
//   // }
//   // for (const row of mat) {
//   //   console.log(row.reduce((a, b) => a + b));
//   // console.log(getStandardDeviation(row));
//   // }
// })();

function cpuTest() {
  const inputEmbeddings = validateModel[tokenIndex].tok_pos_emb.data[0];

  const biasEnabled = modelParams.params.biasEnabled;

  const prefix = `transformer.h.${0}.`;

  const layerNormAttentionGamma = rawModel[`${prefix}ln_1.weight`].values.flat().map(parseFloat);
  const layerNormAttentionBeta = biasEnabled ? rawModel[`${prefix}ln_1.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

  console.log(inputEmbeddings, layerNormAttentionBeta, layerNormAttentionGamma);

  // Calculate stats

  // inputEmbeddings dim is 57 x 128

  const average = (array) => array.reduce((a, b) => a + b) / array.length;
  const averageInputEmbeddings = inputEmbeddings.map((row) => average(row));

  console.log("avg", averageInputEmbeddings);

  const variance = (array) => {
    const avg = average(array);
    return average(array.map((a) => (a - avg) ** 2)) + 1e-5;
  };
  const varianceInputEmbeddings = inputEmbeddings.map((row) => variance(row));

  console.log("var", varianceInputEmbeddings);

  const stdevInputEmbeddings = varianceInputEmbeddings.map(Math.sqrt);

  console.log("stdev", stdevInputEmbeddings);

  // let mean = Stats.data[row * 2];
  // let stdev = Stats.data[row * 2 + 1];
  // let output = (Input.data[row * dimX + col] - mean) / stdev;
  // let gamma = Gamma.data[row * 2];
  // let beta = Beta.data[row * 2];
  // let shift = gamma * output + beta;
  // Result.data[row * dimX + col] = shift;

  const output = inputEmbeddings.map((row, rowIdx) => {
    return row.map((col, colIdx) => {
      const mean = averageInputEmbeddings[rowIdx];
      const stdev = stdevInputEmbeddings[rowIdx];
      const gamma = layerNormAttentionGamma[colIdx];
      const beta = layerNormAttentionBeta[colIdx];
      return ((col - mean) / stdev) * gamma + beta;
    });
  });

  console.log("output", output);

  const expectedDict = validateModel[tokenIndex][`block${0}_ln1`];
  const expected = expectedDict.data[0];
  console.log("expected output", expected);
}

function cpuTest2() {
  const inputEmbeddings = validateModel[0].tok_pos_emb.data[0];

  const biasEnabled = modelParams.params.biasEnabled;
  const n_embd = modelParams.params.n_embd;

  const prefix = `transformer.h.${0}.`;

  const layerNormAttentionGamma = rawModel[`${prefix}ln_1.weight`].values.flat().map(parseFloat);
  const layerNormAttentionBeta = biasEnabled ? rawModel[`${prefix}ln_1.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

  // const layerNormAttentionGamma = new Array(n_embd).fill(1);
  // const layerNormAttentionBeta = new Array(n_embd).fill(0);

  console.log(inputEmbeddings, layerNormAttentionBeta, layerNormAttentionGamma);

  const average = (array) => array.reduce((a, b) => a + b) / array.length;

  // inputEmbeddings dim is 57 x 128
  // Calculate column averages of of 1 x 128
  const averageInputEmbeddings = inputEmbeddings[0].map((_, colIdx) => average(inputEmbeddings.map((row) => row[colIdx])));

  // console.log("avg", averageInputEmbeddings);

  // Calculate column variances of of 1 x 128
  const variance = (array) => {
    const avg = average(array);
    return average(array.map((a) => (a - avg) ** 2)) + 1e-5;
  };
  const varianceInputEmbeddings = inputEmbeddings[0].map((_, colIdx) => variance(inputEmbeddings.map((row) => row[colIdx])));

  // console.log("var", varianceInputEmbeddings);

  // Calculate column stdevs of of 1 x 128
  const stdevInputEmbeddings = varianceInputEmbeddings.map(Math.sqrt);

  // console.log("stdev", stdevInputEmbeddings);

  // Calculate output of 57 x 128
  const output = inputEmbeddings.map((row, rowIdx) => {
    return row.map((col, colIdx) => {
      const mean = averageInputEmbeddings[colIdx];
      const stdev = stdevInputEmbeddings[colIdx];
      const gamma = layerNormAttentionGamma[colIdx];
      const beta = layerNormAttentionBeta[colIdx];
      return ((col - mean) / stdev) * gamma + beta;
    });
  });
  console.log("output", output);

  // Verify that every column is gaussian
  // verifyLayerNorm(output);

  const averageOutput = output[0].map((_, colIdx) => average(output.map((row) => row[colIdx])));
  console.log(averageOutput);

  const stdevOutput = output[0].map((_, colIdx) => variance(output.map((row) => row[colIdx]))).map(Math.sqrt);
  console.log(stdevOutput);

  const expectedDict = validateModel[tokenIndex][`block${0}_ln1`];
  const expected = expectedDict.data[0];
  console.log("expected output", expected);

  const averageExpectedOutput = expected[0].map((_, colIdx) => average(expected.map((row) => row[colIdx])));
  console.log(averageExpectedOutput);

  const stdevExpectedOutput = expected[0].map((_, colIdx) => variance(expected.map((row) => row[colIdx]))).map(Math.sqrt);
  console.log(stdevExpectedOutput);

  const averageExpectedOutputRow = expected.map((row) => average(row));
  console.log(averageExpectedOutputRow);

  const stdevExpectedOutputRow = expected.map((row) => variance(row)).map(Math.sqrt);
  console.log(stdevExpectedOutputRow);
}

function verifyLayerNorm(output, epsilon = 1e-2) {
  const columnMean = (array, colIdx) => array.reduce((a, b) => a + b[colIdx], 0) / array.length;

  const columnStdev = (array, colIdx) => {
    const mean = columnMean(array, colIdx);
    const variance = array.reduce((a, b) => a + (b[colIdx] - mean) ** 2, 0) / array.length;
    return Math.sqrt(variance + 1e-5);
  };

  const numColumns = output[0].length;
  for (let colIdx = 0; colIdx < numColumns; colIdx++) {
    const mean = columnMean(output, colIdx);
    const stdev = columnStdev(output, colIdx);
    if (Math.abs(mean) > epsilon || Math.abs(stdev - 1) > epsilon) {
      console.log(`Column ${colIdx} does not meet the criteria: mean = ${mean}, stdev = ${stdev}`);
      return false;
    }
  }
  console.log("All columns meet the criteria");
  return true;
}

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

function devInlineAttention(
  device,
  queue,
  commandEncoder,
  seq_length,
  n_embd,
  attentionDotProductScale,
  inputBuffer,
  n_heads,
  qkvWeightsBuffer,
  qkvBiasBuffer,
  linearWeightsBuffer,
  linearBiasBuffer
) {
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const workgroup_X = 16; // Dictated by shader.
  const workgroup_Y = 16; // Dictated by shader.

  if (n_embd % n_heads != 0) {
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
  const attentionWeightsResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_heads), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionWeightsBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
  queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_heads, n_embd / n_heads, n_embd]));

  const multiplyUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const multiplyResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_heads), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const multiplyBindGroup = createBindGroup(device, multiplyBindGroupLayout, [multiplyUniformBuffer, multiplyResultBuffer]);
  queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_heads]));
  queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

  const causalMaskUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const causalMaskResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_heads), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const causalMaskBindGroup = createBindGroup(device, causalMaskBindGroupLayout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
  queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_heads, seq_length])); // Transposes! This is needed for softmax.

  const attentionValuesUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionValuesResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionValuesBindGroup = createBindGroup(device, attentionBindGroupLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
  queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_heads, n_embd / n_heads]));

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
  passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_heads, workgroup_X));
  passEncoder_attentionWeights.end();

  const passEncoder_multiply = commandEncoder.beginComputePass();
  passEncoder_multiply.setPipeline(multiplyPipeline);
  passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
  passEncoder_multiply.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [attentionWeightsResultBuffer]));
  passEncoder_multiply.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_heads, workgroup_X));
  passEncoder_multiply.end();

  const passEncoder_causalMask = commandEncoder.beginComputePass();
  passEncoder_causalMask.setPipeline(causalMaskPipeline);
  passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
  passEncoder_causalMask.setBindGroup(1, createBindGroup(device, inputBufferBindGroupLayout, [multiplyResultBuffer]));
  passEncoder_causalMask.dispatchWorkgroups(workgroupCalc(seq_length * n_heads, workgroup_Y), workgroupCalc(seq_length, workgroup_X));
  passEncoder_causalMask.end();

  const softmaxOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(seq_length, seq_length * n_heads),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  for (let i = 0; i < n_heads; i++) {
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

  return {
    qkvResultBuffer,
    splitQResultBuffer,
    splitKResultBuffer,
    splitVResultBuffer,
    attentionWeightsResultBuffer,
    multiplyResultBuffer,
    causalMaskResultBuffer,
    attentionValuesResultBuffer,
    linearResultBuffer,
  };
}

async function runMatMulDynamic(device, queue, pipeline, A, B, verbose = false) {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;

  // [row][col]
  const bufferSizeA = alignedSize(A.length * A[0].length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const bufferSizeB = alignedSize(B.length * B[0].length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const bufferSizeC = alignedSize(B[0].length * A.length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

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

  const workgroupSizeX = 16;
  const workgroupSizeY = 16;
  const numWorkgroupsX = Math.min(Math.ceil(masterDimA / workgroupSizeX), 256);
  const numWorkgroupsY = Math.min(Math.ceil(masterDimB / workgroupSizeY), 256);

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
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
    console.log("resultMatrix (row 0, elem 0)", resultMatrix[0][0]);
  }

  return resultArray;
}

async function runMatMulDiscrete(
  device,
  queue,
  pipeline,
  A,
  B,
  bufferA,
  bufferB,
  bufferC,
  uniformBuffer,
  dim,
  masterDimA,
  masterDimB,
  bindGroup,
  numWorkgroupsX,
  numWorkgroupsY,
  bufferSizeC
) {
  const flatA = flatten(A);
  const flatB = flatten(B);

  queue.writeBuffer(bufferA, 0, flatA);
  queue.writeBuffer(bufferB, 0, flatB);
  queue.writeBuffer(uniformBuffer, 0, new Uint32Array([masterDimA, masterDimB, dim]));

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
  passEncoder.end();

  const readBuffer = device.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, bufferSizeC);

  queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readBuffer.getMappedRange();
  const resultArray = new Float32Array(arrayBuffer);

  return resultArray;
}

async function runMatMulSameMatrix(device, queue, pipeline, bufferC, bindGroup, numWorkgroupsX, numWorkgroupsY, bufferSizeC) {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY, 1);
  passEncoder.end();

  const readBuffer = device.createBuffer({
    size: bufferSizeC,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, bufferSizeC);

  queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = readBuffer.getMappedRange();
  const resultArray = new Float32Array(arrayBuffer);

  return resultArray;
}

async function preMatMulDiscrete(device, queue, pipeline, A, B) {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const minStorageBufferOffsetAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferSizeA = alignedSize(A.length * A[0].length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const bufferSizeB = alignedSize(B.length * B[0].length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const bufferSizeC = alignedSize(B[0].length * A.length * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
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

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferC } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const workgroupSizeX = 16;
  const workgroupSizeY = 16;
  const numWorkgroupsX = Math.min(Math.ceil(masterDimA / workgroupSizeX), 256);
  const numWorkgroupsY = Math.min(Math.ceil(masterDimB / workgroupSizeY), 256);

  const flatA = flatten(A);
  const flatB = flatten(B);

  queue.writeBuffer(bufferA, 0, flatA);
  queue.writeBuffer(bufferB, 0, flatB);
  queue.writeBuffer(uniformBuffer, 0, new Uint32Array([masterDimA, masterDimB, dim]));

  return {
    bufferA,
    bufferB,
    bufferC,
    uniformBuffer,
    dim,
    masterDimA,
    masterDimB,
    bindGroup,
    numWorkgroupsX,
    numWorkgroupsY,
    bufferSizeC,
  };
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
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
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
