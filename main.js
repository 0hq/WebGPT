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

  const context_size = modelParams.params.context_size;
  console.log("block_size", context_size);
  for (let i = 0; i < max_new_tokens; i++) {
    const idx_cond = history.slice(-context_size);

    const result = await runInference(idx_cond);
    const logits = result;
    const probs = cpuSoftmax(logits, temperature);
    const idx_next = sampleFromDistribution(probs, top_k);

    history = history.concat(idx_next);

    console.log(`Output:\n${tokenizer.decode(history)}`);

    // Yield the generated text to the caller
    yield tokenizer.decode([idx_next]);
  }
}

async function runInference(idx) {
  if (!modelParams) {
    console.log("Model not loaded yet");
    return;
  }

  const { device, queue, params, posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeights } = modelParams;
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
    normBetaBuffer,
    embeddingWeights
  );

  return new Float32Array(result);
}
