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
