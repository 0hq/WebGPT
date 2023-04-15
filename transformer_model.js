async function runGPT(
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

async function timeGPT() {
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
  const result = await runGPT(
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

// timeGPT();
