async function runGPTValidation(
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
  normBetaBuffer,
  validateIndex
) {
  console.log("Running GPT validation...");

  const commandEncoder = device.createCommandEncoder();

  console.log("Mixing embeddings...");
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
    console.log(`Processing block ${i}...`);
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

  console.log("Normalizing output...");

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerBuffer, normGammaBuffer, normBetaBuffer);

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

  queue.submit([commandEncoder.finish()]);

  await outputEmbedBuffer.mapAsync(GPUMapMode.READ);

  for (let i = 0; i < n_layers; i++) {
    const block = outputBlockBuffers[i];
    for (let j = 0; j < block.length; j++) {
      await block[j].mapAsync(GPUMapMode.READ);
    }
  }
  await outputLayerBuffer.mapAsync(GPUMapMode.READ);
  await outputLayerNormBuffer.mapAsync(GPUMapMode.READ);

  // You can't read twice from mapped range.
  const layerNormOutput = outputLayerNormBuffer.getMappedRange();
  const output = deEmbedCPU(layerNormOutput, seq_length, n_embd, vocab_size);

  console.log("Validating output...");
  console.log("Expected output block:", validateModel[validateIndex]);
  console.log("Validating embedding...");
  validateResult(new Float32Array(outputEmbedBuffer.getMappedRange()), validateModel[validateIndex].tok_pos_emb);
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
  validateResult(new Float32Array(layerNormOutput), validateModel[validateIndex].ln_f);
  console.log("Validating logits...");
  validateResult(new Float32Array(output), validateModel[validateIndex].logits);

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
      console.log("Result mat:", resultArray, validateArray);
      // console.log("Validate mat:", validateArray);
    }
  }
}

function reshapeRecursively(flatArray, shape) {
  if (shape.length === 1) {
    return flatArray.slice(0, shape[0]);
  }

  let result = [];
  let elementsPerSection = shape.slice(1).reduce((a, b) => a * b);
  for (let i = 0; i < flatArray.length; i += elementsPerSection) {
    result.push(reshapeRecursively(flatArray.slice(i, i + elementsPerSection), shape.slice(1)));
  }

  return result;
}

async function loadValidateModel(validateFile) {
  console.log("Loading validation model...");

  const validateData = await (await fetch(`test/${validateFile}`)).json();

  const steps = [];
  for (let i = 0; i < validateData.length; i++) {
    const loadedData = {};
    for (const key in validateData[i]) {
      const shape = validateData[i][key].shape;
      const data = validateData[i][key].data.flat(Infinity).map((value) => parseFloat(value));
      const typedArray = new Float32Array(data);

      loadedData[key] = {
        shape,
        data: reshapeRecursively(typedArray, shape),
      };
    }
    steps.push(loadedData);
  }

  return steps;
}

function checkAlmostEqualMatrices(a, b) {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) {
      return false;
    }
    for (let j = 0; j < a[i].length; j++) {
      if (a[i][j] - b[i][j] > 0.001) {
        return false;
      }
    }
  }
  return true;
}

function formatAsMatrix(floatArray, dimA, dimB) {
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(floatArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}

async function runValidation(idx, validationIndex) {
  if (!modelParams || !embeddingWeights) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("\nRunning model inference.");
  console.log("Starting with", idx.length, "tokens.");

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer } = modelParams;
  const { attentionDotProductScale, n_embd, n_heads, n_layers, vocab_size } = params;
  const seq_length = idx.length;

  console.log("Embedding inputs...");

  const embeddings = idx.map((token) => embeddingWeights.slice(token * n_embd, (token + 1) * n_embd));
  const flattened = flattenEmbeddings(embeddings);
  const embdOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdOutputBuffer, 0, flattened);

  const startTime = performance.now();
  const result = await runGPTValidation(
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
    normBetaBuffer,
    validationIndex
  );

  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  return new Float32Array(result);
}

async function validateAgainstModel() {
  if (!modelParams || !validateModel) {
    console.log("Model not loaded yet");
    return;
  }

  const context_size = modelParams.params.context_size;

  console.log(`Starting validation.`);
  console.log("Validate model loaded", validateModel);
  console.log("Model params", modelParams);
  console.log("Context size", context_size);

  for (let i = 0; i < validateModel.length; i++) {
    const step = validateModel[i];

    const idx_cond = Array.from(step.idx.data[0].slice(-context_size));
    const logits = await runInference(idx_cond, i);
    const probs = cpuSoftmax(logits, 1.0);

    const idx_next = sampleFromDistribution(probs, 1);

    console.log("Next token", idx_next);
    console.log("Expected token", sampleFromDistribution(step.probs.data[0], 1));

    if (idx_next !== sampleFromDistribution(step.probs.data[0], 1)) {
      throw new Error("Validation failed");
    }
  }
}
