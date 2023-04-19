let modelParams = null;
let bufferSizeCalc = null;

let validateIndex = 0;
let validateModel = null;
const validateFile = "generation copy.json";
const doValidation = false;

let tokenizer = null;

let embeddingWeights = null;

(async () => {
  tokenizer = await loadSimpleTokenizer();

  modelParams = await loadGPTModel("shakespeare_gpt");
  console.log("Params:", modelParams);

  generateFromModel("What is the answer to life, the universe, and everything?", 1, 1);
})();

async function generateFromModel(prompt, max_new_tokens, top_k = 1) {
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
    validateIndex = i;

    const idx_cond = history.slice(-context_size);
    const result = await runInference(idx_cond);
    // const logits = result.slice((idx_cond.length - 1) * modelParams.params.vocab_size);
    const logits = result;
    const probs = cpuSoftmax(logits, 1.0);
    const idx_next = sampleFromDistribution(probs, top_k);

    console.log("Next token", idx_next);
    history = history.concat(idx_next);

    console.log(`Output:\n${tokenizer.decode(history)}`);
  }
}

async function runInference(idx) {
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
  const endTime = performance.now();
  console.log(`Time: ${endTime - startTime} ms`);

  return new Float32Array(result);
}

async function validateAgainstModel() {
  if (!modelParams || !stoi || !itos) {
    console.log("Model not loaded yet");
    return;
  }

  const context_size = modelParams.params.context_size;

  console.log(`Starting validation against ${validateFile}`);
  console.log("Validate model loaded", validateModel);
  console.log("Model params", modelParams);
  console.log("Context size", context_size);

  for (let i = 0; i < validateModel.length; i++) {
    const step = validateModel[i];

    validateIndex = i;

    const idx_cond = step.idx.data[0].slice(-context_size);
    console.log("idx_cond", idx_cond);
    const logits = await runInference(idx_cond);

    // console.log("Logits", logits);
    // console.log("Expected logits", step.logits);

    const probs = cpuSoftmax(logits, 1.0);

    // console.log("Probs", probs);
    // console.log("Expected probs", step.probs);

    const idx_next = sampleFromDistribution(probs, 1);
    console.log("Next token", idx_next);
    console.log("Expected token", sampleFromDistribution(step.probs.data[0], 1));
    if (idx_next !== sampleFromDistribution(step.probs.data[0], 1)) {
      throw new Error("Validation failed");
    }
  }
}
