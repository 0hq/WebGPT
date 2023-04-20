async function runGPT(idx) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeights } = modelParams;
  const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size } = params;
  const seq_length = idx.length;

  const embeddings = idx.map((token) => embeddingWeights.slice(token * n_embd, (token + 1) * n_embd));
  const flattened = flattenEmbeddings(embeddings, n_embd, seq_length);
  const embdOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdOutputBuffer, 0, flattened);

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
  let layerOutputBuffer = embeddedInputBuffer;

  for (let i = 0; i < n_layer; i++) {
    const buffers = layer_buffers[i];

    const layerNormAttentionOutputBuffer = inlineLayerNorm(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      layerOutputBuffer,
      buffers.normAttentionGammaBuffer,
      buffers.normAttentionBetaBuffer
    );

    const attentionOutputBuffer = inlineAttention(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      attentionDotProductScale,
      layerNormAttentionOutputBuffer,
      n_head,
      buffers.qkvWeightsBuffer,
      buffers.qkvBiasBuffer,
      buffers.linearWeightsBuffer,
      buffers.linearBiasBuffer
    );

    const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, layerOutputBuffer);

    const layerNormLinearOutputBuffer = inlineLayerNorm(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      residualAttentionOutputBuffer,
      buffers.normLinearGammaBuffer,
      buffers.normLinearBetaBuffer
    );

    const linearOutputBuffer = inlineFFN(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      hidden_size,
      layerNormLinearOutputBuffer,
      buffers.firstLayerWeightsBuffer,
      buffers.firstLayerBiasBuffer,
      buffers.secondLayerWeightsBuffer,
      buffers.secondLayerBiasBuffer
    );

    const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

    layerOutputBuffer = residualLinearOutputBuffer;
  }

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerOutputBuffer, normGammaBuffer, normBetaBuffer);

  const slicedlayerNormOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(1, n_embd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  commandEncoder.copyBufferToBuffer(
    layerNormOutputBuffer, // Source buffer (original position embeddings)
    bufferSizeCalc((seq_length - 1) * n_embd, n_embd), // Source offset (starting from the beginning of the buffer)
    slicedlayerNormOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(1, n_embd) // Number of bytes to copy
  );

  const embeddingWeightsBuffer = createBuffer(
    device,
    bufferSizeCalc(vocab_size, n_embd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  queue.writeBuffer(embeddingWeightsBuffer, 0, transposeArray(embeddingWeights, vocab_size, n_embd));

  const maxStorageBufferSize = device.limits.maxStorageBufferBindingSize;

  // AFAIK, this should run in parallel because the encoder understands that they have no dependencies. Double check that deEmbed
  const deEmbedOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const embedSize = vocab_size * n_embd;
  const maxVocabChunkSize = Math.floor(maxStorageBufferSize / n_embd);
  const numInstances = Math.ceil(vocab_size / maxVocabChunkSize);
  const vocabChunkSize = maxStorageBufferSize;
  console.log(embedSize, maxStorageBufferSize, numInstances, vocabChunkSize);
  for (let i = 0; i < numInstances; i++) {
    const deEmbedInputBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, vocabChunkSize),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    commandEncoder.copyBufferToBuffer(
      embeddingWeightsBuffer,
      i * bufferSizeCalc(seq_length, vocabChunkSize),
      deEmbedInputBuffer,
      0,
      bufferSizeCalc(n_embd, vocabChunkSize)
    );
    const softMaxResultBuffer = inlineMatMul(
      device,
      queue,
      commandEncoder,
      slicedlayerNormOutputBuffer,
      deEmbedInputBuffer,
      seq_length,
      vocabChunkSize,
      n_embd
    );

    const copySize = i === numInstances - 1 ? bufferSizeCalc(seq_length, vocab_size - vocabChunkSize * i) : bufferSizeCalc(seq_length, vocabChunkSize);
    commandEncoder.copyBufferToBuffer(softMaxResultBuffer, 0, deEmbedOutputBuffer, i * bufferSizeCalc(seq_length, vocabChunkSize), copySize);
  }

  queue.submit([commandEncoder.finish()]);

  await deEmbedOutputBuffer.mapAsync(GPUMapMode.READ);
  const output = deEmbedOutputBuffer.getMappedRange();
  const deEmbed = deEmbedCPU(output, embeddingWeights, seq_length, n_embd, vocab_size);

  return new Float32Array(deEmbed);
}
