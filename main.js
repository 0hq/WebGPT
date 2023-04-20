// These are globals so we don't need to load these multiple times.
let modelParams = null;
let tokenizer = null;

async function* streamModelOutput(prompt, max_new_tokens, top_k = 10, temperature = 1.0) {
  if (!modelParams || !tokenizer) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  let history = tokenizer.encode(prompt);
  console.log("Tokenized prompt", history);

  const block_size = modelParams.params.block_size;
  console.log("block_size", block_size);
  for (let i = 0; i < max_new_tokens; i++) {
    const idx_cond = history.slice(-block_size);

    const result = await runGPT(idx_cond);
    const logits = result;
    const probs = cpuSoftmax(logits, temperature);
    const idx_next = sampleFromDistribution(probs, top_k);

    history = history.concat(idx_next);

    console.log(`Output:\n${tokenizer.decode(history)}`);

    // Yield the generated text to the caller
    yield tokenizer.decode([idx_next]);
  }
}

async function runGPT(idx) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeights } = modelParams;
  const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size } = params;
  const seq_length = idx.length;

  const embeddings = idx.map((token) => embeddingWeights.slice(token * n_embd, (token + 1) * n_embd));
  const flattened = flattenEmbeddings(embeddings);
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
    const {
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
    } = layer_buffers[i];

    const layerNormAttentionOutputBuffer = inlineLayerNorm(
      device,
      queue,
      commandEncoder,
      seq_length,
      n_embd,
      layerOutputBuffer,
      normAttentionGammaBuffer,
      normAttentionBetaBuffer
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
      qkvWeightsBuffer,
      qkvBiasBuffer,
      linearWeightsBuffer,
      linearBiasBuffer
    );

    const residualAttentionOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, attentionOutputBuffer, layerOutputBuffer);

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

    const residualLinearOutputBuffer = inlineResidual(device, queue, commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

    layerOutputBuffer = residualLinearOutputBuffer;
  }

  const layerNormOutputBuffer = inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, layerOutputBuffer, normGammaBuffer, normBetaBuffer);

  const outputBuffer = createOutputBuffer(device, commandEncoder, layerNormOutputBuffer, seq_length, n_embd);

  queue.submit([commandEncoder.finish()]);

  await outputBuffer.mapAsync(GPUMapMode.READ);
  const output = outputBuffer.getMappedRange();
  const deEmbed = deEmbedCPU(output, embeddingWeights, seq_length, n_embd, vocab_size);

  return new Float32Array(deEmbed);
}

async function loadGPTModel(folder) {
  if (modelParams) {
    console.error("Model already loaded");
    return;
  }
  const { device, queue } = await initializeWebGPU();
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, 1);
  // Should be device.limits.minStorageBufferOffsetAlignment once bug is fixed.

  console.log("Loading model from folder:", folder);
  const paramsJSON = await (await fetch(`models/${folder}/params_gpt.json`)).json();
  const { block_size, n_embd, n_head, n_layer, bias } = paramsJSON;
  paramsJSON.hidden_size = n_embd * 4;
  paramsJSON.attentionDotProductScale = 1 / Math.sqrt(n_embd / n_head);
  const { hidden_size } = paramsJSON;
  console.log("Params:", paramsJSON);

  tokenizer = await loadGPT2Tokenizer(); // Sets global tokenizer variable.

  console.log("Loading token embeddings...");
  const embeddingWeights = await loadBinaryFile("models/" + folder + "/transformer.wte.weight_gpt.bin");

  console.log("Loading positional embeddings...");
  const posEmbeddings = await loadBinaryFile("models/" + folder + "/transformer.wpe.weight_gpt.bin");
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(block_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  const layer_buffers = [];
  for (let i = 0; i < n_layer; i++) {
    console.log("Loading layer", i);
    const prefix = `transformer.h.${i}.`;

    console.log("\tLoading attention layer norm...");
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.weight_gpt.bin`));
    queue.writeBuffer(normAttentionBetaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.bias_gpt.bin`));

    console.log("\tLoading qkv transform...");
    const qkv_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.weight_gpt.bin`);
    const qkv_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.bias_gpt.bin`) : new Array(3 * n_embd).fill(0);
    const qkvWeightsBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, 3 * n_embd),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    queue.writeBuffer(qkvWeightsBuffer, 0, transposeArray(qkv_weights, 3 * n_embd, n_embd));
    queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

    console.log("\tLoading attention c_proj...");
    const linear_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.weight_gpt.bin`);
    const linear_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, transposeArray(linear_weights, n_embd, n_embd));
    queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

    console.log("\tLoading MLP layer norm...");
    const layerNormLinearGamma = await loadBinaryFile(`models/${folder}/${prefix}ln_2.weight_gpt.bin`);
    const layerNormLinearBeta = bias ? await loadBinaryFile(`models/${folder}/${prefix}ln_2.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
    queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

    console.log("\tLoading MLP first layer...");
    const firstLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.weight_gpt.bin`);
    const firstLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.bias_gpt.bin`) : new Array(hidden_size).fill(0);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, transposeArray(firstLayerWeights, hidden_size, n_embd));
    queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

    console.log("\tLoading MLP second layer...");
    const secondLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.weight_gpt.bin`);
    const secondLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, transposeArray(secondLayerWeights, n_embd, hidden_size));
    queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

    layer_buffers.push({
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
    });
  }

  console.log("Loading final layer norm...");
  const layerNormGamma = await loadBinaryFile(`models/${folder}/transformer.ln_f.weight_gpt.bin`);
  const layerNormBeta = bias ? await loadBinaryFile(`models/${folder}/transformer.ln_f.bias_gpt.bin`) : new Array(n_embd).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  const output = {
    device,
    queue,
    params: paramsJSON,
    layer_buffers,
    embeddingWeights,
    posEmbdBuffer,
    normGammaBuffer,
    normBetaBuffer,
  };

  console.log("Finished loading model.", output);
  return output;
}
