async function runGPT(
  device,
  queue,
  seq_length,
  vocab_size,
  n_embd,
  n_heads,
  n_layers,
  attentionDotProductScale,
  idx,
  embdBuffer,
  posEmbdBuffer,
  layer_buffers,
  normGammaBuffer,
  normBetaBuffer,
  deEmbedBuffer
) {
  const inputMatrix = new Float32Array(seq_length * vocab_size);
  for (let i = 0; i < seq_length; i++) {
    inputMatrix[i * vocab_size + idx[i]] = 1;
  }
  const inputBuffer = createBuffer(device, bufferSizeCalc(seq_length, vocab_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(inputBuffer, 0, inputMatrix);

  const commandEncoder = device.createCommandEncoder();

  const embdOutputBuffer = inlineMatMul(device, queue, commandEncoder, inputBuffer, embdBuffer, seq_length, n_embd, vocab_size);
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

  const deEmbedOutputBuffer = inlineMatMul(device, queue, commandEncoder, layerNormOutputBuffer, deEmbedBuffer, seq_length, vocab_size, n_embd);

  // OUTPUT and VALIDATION

  const outputEmbedBuffer = createOutputBuffer(device, commandEncoder, embeddedInputBuffer, seq_length, n_embd);

  const outputBlockBuffers = [];
  for (let i = 0; i < n_layers; i++) {
    const block = buffers[i];
    const outputLayerNormAttentionBuffer = createOutputBuffer(device, commandEncoder, block.layerNormAttentionOutputBuffer, seq_length, n_embd);
    const outputAttentionBuffer = createOutputBuffer(device, commandEncoder, block.attentionOutputBuffer, seq_length, n_embd);
    const outputResidualAttentionBuffer = createOutputBuffer(device, commandEncoder, block.residualAttentionOutputBuffer, seq_length, n_embd);
    const outputLayerNormLinearBuffer = createOutputBuffer(device, commandEncoder, block.layerNormLinearOutputBuffer, seq_length, n_embd);
    const outputLinearBuffer = createOutputBuffer(device, commandEncoder, block.linearOutputBuffer, seq_length, n_embd);
    const outputResidualLinearBuffer = createOutputBuffer(device, commandEncoder, block.residualLinearOutputBuffer, seq_length, n_embd);
    outputBlockBuffers.push([
      outputLayerNormAttentionBuffer,
      outputAttentionBuffer,
      outputResidualAttentionBuffer,
      outputLayerNormLinearBuffer,
      outputLinearBuffer,
      outputResidualLinearBuffer,
    ]);
  }
  const outputLayerBuffer = createOutputBuffer(device, commandEncoder, layerBuffer, seq_length, n_embd);
  const outputLayerNormBuffer = createOutputBuffer(device, commandEncoder, layerNormOutputBuffer, seq_length, n_embd);
  const outputBuffer = createOutputBuffer(device, commandEncoder, deEmbedOutputBuffer, seq_length, vocab_size);

  queue.submit([commandEncoder.finish()]);

  await outputEmbedBuffer.mapAsync(GPUMapMode.READ);
  // await outputNormGammaBuffer.mapAsync(GPUMapMode.READ);

  for (let i = 0; i < n_layers; i++) {
    const block = outputBlockBuffers[i];
    for (let j = 0; j < block.length; j++) {
      await block[j].mapAsync(GPUMapMode.READ);
    }
  }
  await outputLayerBuffer.mapAsync(GPUMapMode.READ);
  await outputLayerNormBuffer.mapAsync(GPUMapMode.READ);
  await outputBuffer.mapAsync(GPUMapMode.READ);

  // You can't read twice from mapped range.
  const output = outputBuffer.getMappedRange();

  if (doValidation) {
    console.log("Validating output...");
    console.log("Expected output block:", validateModel[validateIndex]);
    console.log("Validating embedding...");
    validateResult(new Float32Array(outputEmbedBuffer.getMappedRange()), validateModel[validateIndex].tok_pos_emb);
    // console.log("norm gamma", new Float32Array(outputNormGammaBuffer.getMappedRange()));
    console.log("Validating blocks...");
    for (let i = 0; i < n_layers; i++) {
      console.log(`\tValidating block ${i}...`);
      const block = outputBlockBuffers[i];
      console.log("\t\tValidating first layer norm...");
      validateResult(new Float32Array(outputBlockBuffers[i][0].getMappedRange()), validateModel[validateIndex][`block${i}_ln1`]);
      console.log("\t\tValidating attention...");
      validateResult(new Float32Array(outputBlockBuffers[i][1].getMappedRange()), validateModel[validateIndex][`block${i}_attn`]);
      console.log("\t\tValidating residual attention...");
      validateResult(new Float32Array(outputBlockBuffers[i][2].getMappedRange()), validateModel[validateIndex][`block${i}_r1`]);
      console.log("\t\tValidating second layer norm...");
      validateResult(new Float32Array(outputBlockBuffers[i][3].getMappedRange()), validateModel[validateIndex][`block${i}_ln2`]);
      console.log("\t\tValidating mlp...");
      validateResult(new Float32Array(outputBlockBuffers[i][4].getMappedRange()), validateModel[validateIndex][`block${i}_mlp`]);
      console.log("\t\tValidating residual mlp...");
      validateResult(new Float32Array(outputBlockBuffers[i][5].getMappedRange()), validateModel[validateIndex][`block${i}_r2`]);
    }
    console.log("Validating layer norm...");
    validateResult(new Float32Array(outputLayerNormBuffer.getMappedRange()), validateModel[validateIndex].ln_f);
    console.log("Validating logits...");
    validateResult(new Float32Array(output).slice((seq_length - 1) * vocab_size), validateModel[validateIndex].logits);
  }

  return output;
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

function sumMatrix(matrix) {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      sum += Math.abs(matrix[i][j]);
    }
  }
  return sum;
}
