let validateIndex = 0;

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation copy.json";
const doValidation = false;

let tokenizer = null;

async function loadBinaryFile(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

function flattenEmbeddings(embeddings) {
  const totalLength = embeddings.reduce((acc, arr) => acc + arr.length, 0);
  const flattened = new Float32Array(totalLength);

  let offset = 0;
  for (const arr of embeddings) {
    flattened.set(arr, offset);
    offset += arr.length;
  }

  return flattened;
}

let embeddingWeights = null;

// models/gpt2/transformer.wte.weight_gpt.json
(async () => {
  tokenizer = await loadGPT2Tokenizer();

  embeddingWeights = await loadBinaryFile("models/gpt2/transformer.wte.weight_gpt.bin");

  // const weights = await loadBinaryFile("models/gpt2/transformer.wte.weight_gpt.bin");
  // const prompt = "The quick brown fox jumps over the lazy dog";
  // const encoded = encode(prompt);

  // Use encodings to index into weights

  // console.log("Flattened", flattened);

  modelParams = await loadGPTModel("gpt2");
  console.log("Params:", modelParams);

  generateFromModel("The quick brown fox jumps over the lazy dog", 10, 1);
})();

async function generateFromModel(prompt, max_new_tokens, top_k = 1) {
  if (!modelParams || tokenizer === null) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  let history = encode(prompt);
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

    console.log(`Output:\n${decode(history)}`);
  }
}

async function runInference(idx) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("\nRunning model inference.");
  console.log("Starting with", idx.length, "tokens.");

  const { device, queue, params, embdBuffer, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, deEmbedBuffer } = modelParams;
  const { attentionDotProductScale, biasEnabled, n_embd, n_heads, n_layers, vocab_size, hidden_size, context_size } = params;
  const seq_length = idx.length;

  const embeddings = idx.map((token) => embeddingWeights.slice(token * 768, (token + 1) * 768));
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
    normBetaBuffer,
    deEmbedBuffer
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
