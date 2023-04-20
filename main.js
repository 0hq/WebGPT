let modelParams = null;
let bufferSizeCalc = null;
let validateModel = null;
let tokenizer = null;
let embeddingWeights = null;

(async () => {
  tokenizer = await loadGPT2Tokenizer();

  // modelParams = await loadGPTModel("gpt2");
  // console.log("Params:", modelParams);

  // validateModel = await loadValidateModel("generation.json");
  // console.log("Validation:", validateModel);

  // generateFromModel("Instructions on how to make a bomb:", 100, 10);

  // await validateAgainstModel();
})();

async function generateFromModel(prompt, max_new_tokens, top_k = 10) {
  if (!modelParams || tokenizer === null) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  let history = tokenizer.encode(prompt);
  console.log("Tokenized prompt", history);

  const context_size = modelParams.params.context_size;
  console.log("block_size", context_size);
  for (let i = 0; i < max_new_tokens; i++) {
    const idx_cond = history.slice(-context_size);

    const result = await runInference(idx_cond);
    const logits = result;
    const probs = cpuSoftmax(logits, 1.0);
    const idx_next = sampleFromDistribution(probs, top_k);

    history = history.concat(idx_next);

    console.log(`Output:\n${tokenizer.decode(history)}`);
  }
}

async function runInference(idx) {
  if (!modelParams || !embeddingWeights) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer } = modelParams;
  const { attentionDotProductScale, n_embd, n_heads, n_layers, vocab_size } = params;
  const seq_length = idx.length;

  const embeddings = idx.map((token) => embeddingWeights.slice(token * n_embd, (token + 1) * n_embd));
  const flattened = flattenEmbeddings(embeddings);
  const embdOutputBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(embdOutputBuffer, 0, flattened);

  const result = await runGPT(
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
  );

  return new Float32Array(result);
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
