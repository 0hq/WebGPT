async function loadGPTModel(folder) {
  if (modelParams) {
    console.error("Model already loaded");
    return;
  }

  console.log("Loading model from folder:", folder);

  const { device, queue } = await initializeWebGPU();
  console.log("Device:", device.limits);

  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, 1);
  // Should be device.limits.minStorageBufferOffsetAlignment once bug is fixed.

  const paramsJSON = await (await fetch(`models/${folder}/params_gpt.json`)).json();
  const { block_size: context_size, vocab_size, n_embd, n_head: n_heads, n_layer: n_layers, bias: biasEnabled } = paramsJSON;
  const hidden_size = n_embd * 4; // Transformer block has 4 hidden layers by default, not a param.
  const attentionDotProductScale = 1 / Math.sqrt(n_embd / n_heads);

  console.log("context_size", context_size, "vocab_size", vocab_size, "n_embd", n_embd, "n_heads", n_heads, "n_layers", n_layers, "biasEnabled", biasEnabled);

  // Sets global tokenizer variable.
  tokenizer = await loadGPT2Tokenizer();

  console.log("Loading embeddings...");
  const embeddingWeights = await loadBinaryFile("models/" + folder + "/transformer.wte.weight_gpt.bin");

  console.log("Loading positional embeddings...");
  const posEmbeddings = await loadBinaryFile("models/" + folder + "/transformer.wpe.weight_gpt.bin");
  const posEmbdBuffer = createBuffer(device, bufferSizeCalc(context_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

  const layer_buffers = [];

  for (let i = 0; i < n_layers; i++) {
    const buffers = [];
    const prefix = `transformer.h.${i}.`;

    console.log("Loading layer", i);

    console.log("\tLoading attention layer norm...");
    const normAttentionGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normAttentionBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normAttentionGammaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.weight_gpt.bin`));
    queue.writeBuffer(normAttentionBetaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.bias_gpt.bin`));
    buffers.push(normAttentionGammaBuffer, normAttentionBetaBuffer);

    console.log("\tLoading qkv transform...");
    const qkv_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.weight_gpt.bin`);
    const qkv_bias = biasEnabled ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.bias_gpt.bin`) : new Array(3 * n_embd).fill(0);
    const qkvWeightsBuffer = createBuffer(
      device,
      bufferSizeCalc(n_embd, 3 * n_embd),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const qkvBiasBuffer = createBuffer(device, bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    queue.writeBuffer(qkvWeightsBuffer, 0, transposeArray(qkv_weights, 3 * n_embd, n_embd));
    queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);
    buffers.push(qkvWeightsBuffer, qkvBiasBuffer);

    console.log("\tLoading attention c_proj...");
    const linear_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.weight_gpt.bin`);
    const linear_bias = biasEnabled ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const linearWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const linearBiasBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(linearWeightsBuffer, 0, transposeArray(linear_weights, n_embd, n_embd));
    queue.writeBuffer(linearBiasBuffer, 0, linear_bias);
    buffers.push(linearWeightsBuffer, linearBiasBuffer);

    console.log("\tLoading MLP layer norm...");
    const layerNormLinearGamma = await loadBinaryFile(`models/${folder}/${prefix}ln_2.weight_gpt.bin`);
    const layerNormLinearBeta = biasEnabled ? await loadBinaryFile(`models/${folder}/${prefix}ln_2.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const normLinearGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normLinearBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
    queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);
    buffers.push(normLinearGammaBuffer, normLinearBetaBuffer);

    console.log("\tLoading MLP first layer...");
    const firstLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.weight_gpt.bin`);
    const firstLayerBias = biasEnabled ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.bias_gpt.bin`) : new Array(hidden_size).fill(0);
    const firstLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const firstLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(firstLayerWeightsBuffer, 0, transposeArray(firstLayerWeights, hidden_size, n_embd));
    queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);
    buffers.push(firstLayerWeightsBuffer, firstLayerBiasBuffer);

    console.log("\tLoading MLP second layer...");
    const secondLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.weight_gpt.bin`);
    const secondLayerBias = biasEnabled ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
    const secondLayerWeightsBuffer = createBuffer(device, bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const secondLayerBiasBuffer = createBuffer(device, bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    queue.writeBuffer(secondLayerWeightsBuffer, 0, transposeArray(secondLayerWeights, n_embd, hidden_size));
    queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);
    buffers.push(secondLayerWeightsBuffer, secondLayerBiasBuffer);

    layer_buffers.push(buffers);
  }

  console.log("Loading final layer norm...");
  const layerNormGamma = await loadBinaryFile(`models/${folder}/transformer.ln_f.weight_gpt.bin`);
  const layerNormBeta = biasEnabled ? await loadBinaryFile(`models/${folder}/transformer.ln_f.bias_gpt.bin`) : new Array(n_embd).fill(0);
  const normGammaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const normBetaBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
  queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

  return {
    device,
    queue,
    params: {
      attentionDotProductScale,
      biasEnabled,
      n_embd,
      n_heads,
      n_layers,
      vocab_size,
      hidden_size,
      context_size,
    },
    embeddingWeights,
    posEmbdBuffer,
    layer_buffers,
    normGammaBuffer,
    normBetaBuffer,
  };
}
