async function runGPT(
  device,
  queue,
  seq_length,
  vocab_size,
  n_embd,
  n_heads,
  n_layers,
  attentionDotProductScale,
  embdOutputBuffer,
  posEmbdBuffer,
  layer_buffers,
  normGammaBuffer,
  normBetaBuffer
) {
  const commandEncoder = device.createCommandEncoder();

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
  let layerBuffer = embeddedInputBuffer;

  // Used for validation.
  const buffers = [];

  for (let i = 0; i < n_layers; i++) {
    // console.log(`Processing block ${i}...`);
    const layer_params = layer_buffers[i];
    const {
      layerNormAttentionOutputBuffer,
      attentionOutputBuffer,
      residualAttentionOutputBuffer,
      layerNormLinearOutputBuffer,
      linearOutputBuffer,
      residualLinearOutputBuffer,
    } = transformerBlock(device, queue, commandEncoder, seq_length, n_embd, n_heads, attentionDotProductScale, layerBuffer, ...layer_params);
    buffers.push({
      layerNormAttentionOutputBuffer,
      attentionOutputBuffer,
      residualAttentionOutputBuffer,
      layerNormLinearOutputBuffer,
      linearOutputBuffer,
      residualLinearOutputBuffer,
    });
    layerBuffer = residualLinearOutputBuffer;
  }

  // console.log("Normalizing output...");
  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

  const outputBuffer = createOutputBuffer(device, commandEncoder, layerNormOutputBuffer, seq_length, n_embd);

  queue.submit([commandEncoder.finish()]);

  await outputBuffer.mapAsync(GPUMapMode.READ);

  // You can't read twice from mapped range.
  const output = outputBuffer.getMappedRange();

  return deEmbedCPU(output, seq_length, n_embd, vocab_size);
}

function transformerBlock(
  device,
  queue,
  commandEncoder,
  seq_length,
  n_embd,
  n_heads,
  attentionDotProductScale,
  inputBuffer,
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
) {
  const minStorageBufferOffsetAlignment = 1; // device.limits.minStorageBufferOffsetAlignment; // This was breaking things. Probably should check later.
  const bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.

  // console.log("\tNormalizing block input...");

  const layerNormAttentionOutputBuffer = inlineLayerNorm(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    inputBuffer,
    normAttentionGammaBuffer,
    normAttentionBetaBuffer
  );

  // console.log("\tRunning attention block...");

  const attentionOutputBuffer = inlineAttention(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    layerNormAttentionOutputBuffer,
    n_heads,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer
  );

  // console.log("\tResidual connections from attention and input...");

  const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, inputBuffer);

  // console.log("\tNormalizing attention output...");

  const layerNormLinearOutputBuffer = inlineLayerNorm(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    residualAttentionOutputBuffer,
    normLinearGammaBuffer,
    normLinearBetaBuffer
  );

  // console.log("\tRunning MLP...");

  const linearOutputBuffer = inlineFFN(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    hidden_size,
    layerNormLinearOutputBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer
  );

  // console.log("\tResidual connections from MLP and attention...");

  const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

  return {
    layerNormAttentionOutputBuffer,
    attentionOutputBuffer,
    residualAttentionOutputBuffer,
    layerNormLinearOutputBuffer,
    linearOutputBuffer,
    residualLinearOutputBuffer,
  };
}

function deEmbedCPU(embeddings, seq_length, n_embd, vocab_size) {
  // console.log("De-embedding output... (CPU)");

  const predictionEmbeddings = new Float32Array(embeddings).slice((seq_length - 1) * n_embd);

  // const vocabToEmbeddings = transposeArray(embeddingWeights, vocab_size, n_embd);
  const vocabToEmbeddings = embeddingWeights;

  const logits = [];
  for (let i = 0; i < vocab_size; i++) {
    let dotProduct = 0;
    for (let j = 0; j < n_embd; j++) {
      dotProduct += vocabToEmbeddings[i * n_embd + j] * predictionEmbeddings[j];
    }
    logits.push(dotProduct);
  }

  return logits;
}
