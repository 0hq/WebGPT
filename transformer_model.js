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

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

  const outputBuffer = createOutputBuffer(device, commandEncoder, layerNormOutputBuffer, seq_length, n_embd);

  queue.submit([commandEncoder.finish()]);

  await outputBuffer.mapAsync(GPUMapMode.READ);

  // You can't read twice from mapped range.
  const output = outputBuffer.getMappedRange();

  return deEmbedCPU(output, seq_length, n_embd, vocab_size);
}

function deEmbedCPU(embeddings, seq_length, n_embd, vocab_size) {
  const predictionEmbeddings = new Float32Array(embeddings).slice((seq_length - 1) * n_embd);

  const vocabToEmbeddings = transposeArray(embeddingWeights, vocab_size, n_embd);

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

function validateResult(result, validate, verbose = false) {
  const resultArray = formatAsMatrix(result, validate.shape[1], validate.shape[2]);
  const validateArray = validate.data[0]; // Unpack from batch of 1

  const equal = checkAlmostEqualMatrices(resultArray, validateArray);

  if (!equal) {
    // console.log("Result:", result);
    // console.log("Validate:", validate);
    console.log("Result mat:", resultArray);
    console.log("Validate mat:", validateArray);

    // Calculate the difference
    const diff = subtractMatrices(resultArray, validateArray);
    console.log("Diff mat:", diff);

    // Sum the absolute values of the difference
    const sum = sumMatrix(diff);
    console.log("Sum:", sum);

    throw new Error("Test failed");
  } else {
    // console.log("Test passed!");
    if (verbose) {
      console.log("Result mat:", resultArray);
      // console.log("Validate mat:", validateArray);
    }
  }
}
