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

  /* 
    Compute Pass Splitting: Possible Approaches.

    The limiting factor running larger models is often the maxStorageBufferBindingSize, which prevents you from doing giant matrix multiplications. As far as I know, the best way to solve this is to generate multiple compute passes and split the calculation into smaller sub-matrices. You can simply write back to the buffer with the proper offset and byte size and nothing changes. As long as these operations don't have inter-dependencies, WebGPU should recognize that they can be run in parallel and you shouldn't experience significant performance losses. This needs to be verified!

    The question then is to how to divide the operation properly for efficiency. I'm still figuring out how everything works, so I'm unsure what the most efficient way to do this is.

    The first thought is that if you have a matrix of elements maxStorageBufferBindingSize * 2, it's straightforward to chop it down the middle. However for non-evenly divisible matrix sizes, you might run into serious memory inefficiencies if you divide by the minimum number of sub-matrices. 

    I've implemented/planned a couple different solutions.

    (1) Start with the minimum number of groups and calculate wasted memory, then increase # of groups and record the most efficient sizing up to some maximum group number. This calculation can be done when the model is loaded.

    (2) Divide by some standard matrix size (maybe a square matrix of rows * rows) and then add one final "overflow matrix" of some irregular size. I really don't know if this is more efficient, still learning, but my gut tells me this might be result in too many groups when fewer could do better.
    
    (3) Assume that the matrix has some decently small factor that fits perfectly and use that. This is the simplest solution, and given that I have 0 clue which option is best until I test, I'm going with this for now.

  */

  const slicedEmbedOutputBuffer = createBuffer(device, bufferSizeCalc(1, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  commandEncoder.copyBufferToBuffer(
    layerNormOutputBuffer, // Source buffer (original position embeddings)
    bufferSizeCalc((seq_length - 1) * n_embd, n_embd), // Source offset (starting from the beginning of the buffer)
    slicedEmbedOutputBuffer, // Destination buffer (cropped buffer)
    0, // Destination offset (starting from the beginning of the cropped buffer)
    bufferSizeCalc(1, n_embd) // Number of bytes to copy
  );

  const embeddingWeightsBuffer = createBuffer(
    device,
    bufferSizeCalc(vocab_size, n_embd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  queue.writeBuffer(embeddingWeightsBuffer, 0, transposeArray(embeddingWeights, vocab_size, n_embd));

  const deEmbedOutputBuffer = createBuffer(device, bufferSizeCalc(1, vocab_size), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  // Assumes that vocab_size has a decent least prime factor.
  const maxStorageBufferSize = device.limits.maxStorageBufferBindingSize;
  const totalElements = n_embd * vocab_size;
  var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
  if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
  var vocabChunkSize = vocab_size / numInstances;

  for (let i = 0; i < numInstances; i++) {
    const deEmbedChunkInputBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, vocabChunkSize),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    // Remember that embeddingWeightsBuffer is transposed.
    commandEncoder.copyBufferToBuffer(
      embeddingWeightsBuffer,
      i * bufferSizeCalc(n_embd * vocabChunkSize),
      deEmbedChunkInputBuffer,
      0,
      bufferSizeCalc(n_embd, vocabChunkSize)
    );
    // We're doing some buffer tricks here. Since slicedlayerNormOutputBuffer is a row matrix, we can just pretend it's a column matrix without any changes to the way it's stored. We then multiply it by the transposed embeddingWeights chunk, resulting in a column vector which, once again, we can pretend is a row vector.
    const deEmbedChunkResultBuffer = inlineMatMul(
      device,
      queue,
      commandEncoder,
      deEmbedChunkInputBuffer,
      slicedlayerNormOutputBuffer,
      vocabChunkSize,
      1,
      n_embd
    );
    commandEncoder.copyBufferToBuffer(deEmbedChunkResultBuffer, 0, deEmbedOutputBuffer, i * bufferSizeCalc(vocabChunkSize), bufferSizeCalc(vocabChunkSize));
  }

  queue.submit([commandEncoder.finish()]);

  await deEmbedOutputBuffer.mapAsync(GPUMapMode.READ);
  const output = deEmbedOutputBuffer.getMappedRange();

  return new Float32Array(output);
}
