class Model {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;

    this.defaultPrompt;
    this.defaultTopK;
    this.defaultTemperature;
    this.defaultTokens;

    this.bufferDeletionStack = [];
    this.unloadDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    [this.model, this.params] = await this.loadModel(this.folder);
    this.tokenizer = this.tokenizerType == "bpe" ? new GPT2Tokenizer() : new SimpleTokenizer();
    await this.tokenizer.load();

    if (this.params.n_embd % 4 !== 0 || this.params.n_head % 4 !== 0) {
      throw new Error("Model incompatible. n_embd and n_head must be divisible by 4 for fast matmul.");
    }

    if (this.folder == "gpt2") {
      this.defaultPrompt = `What is the answer to life, the universe, and everything?\n`;
      this.defaultTopK = 3;
      this.defaultTemperature = 1;
      this.defaultTokens = 30;
    } else {
      this.defaultPrompt = `WILL:\nAh, how dare you challenge me?\nHave you forgotten I built WebGPT?\n`;
      this.defaultTopK = 1;
      this.defaultTemperature = 1;
      this.defaultTokens = 80;
    }

    this.initialized = true;

    console.log("Model initialized");
  }

  // Fetch bin should be parallelized for big loading time reduction.
  async loadModel(folder) {
    if (this.initialized) return console.error("Model already loaded");

    console.log("Loading model from folder:", folder);
    const fldr = `models/${folder}/`;
    const zeros = (dim) => new Float32Array(dim).fill(0);

    const params = await (await fetch(`${fldr}/params_gpt.json`)).json();
    params.hidden_size = params.n_embd * 4;
    params.attentionDotProductScale = 1 / Math.sqrt(params.n_embd / params.n_head);
    const { block_size, n_embd, n_head, n_layer, bias, vocab_size, hidden_size } = params;
    console.log("Params:", params);

    // Did you enable GitHub LFS? Won't work without it.
    if (n_embd % n_head != 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

    const FastMatMulBlock = new FastMatMulBlock(this.device);
    const ResidualBlock = new ResidualBlock(this.device);
    const NaiveMatMulBlock = new NaiveMatMulBlock(this.device);
    const TransposeBlock = new TransposeBlock(this.device);
    const FastRowAddBlock = new FastRowAddBlock(this.device);
    const LayerNormBlock = new LayerNormBlock(this.device);
    const SoftmaxBlock = new SoftmaxBlock(this.device);
    const GeluBlock = new GeluBlock(this.device);
    const AttentionBlock = new AttentionBlock(this.device);

    console.log("Loading token embeddings...");
    const embeddingWeights = await fetchBin(`${fldr}/transformer.wte.weight_gpt.bin`);
    const embeddingWeightsBuffer = this.initTensor(embeddingWeights, [vocab_size, n_embd], ["copy_from"]);

    console.log("Loading positional embeddings...");
    const posEmbeddings = await fetchBin(`${fldr}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, [block_size, n_embd], ["copy_from"]);

    const layer_buffers = [];
    for (let i = 0; i < n_layer; i++) {
      console.log("Loading layer...", i);
      const prefix = `${fldr}transformer.h.${i}.`;

      const normAttentionGamma = await fetchBin(`${prefix}ln_1.weight_gpt.bin`);
      const normAttentionBeta = bias ? await fetchBin(`${prefix}ln_1.bias_gpt.bin`) : zeros(n_embd);

      const qkvWeights = transpose(await fetchBin(`${prefix}attn.c_attn.weight_gpt.bin`), 3 * n_embd, n_embd);
      const qkvBias = bias ? await fetchBin(`${prefix}attn.c_attn.bias_gpt.bin`) : zeros(3 * n_embd);

      const linearWeights = transpose(await fetchBin(`${prefix}attn.c_proj.weight_gpt.bin`), n_embd, n_embd);
      const linearBias = bias ? await fetchBin(`${prefix}attn.c_proj.bias_gpt.bin`) : zeros(n_embd);

      const attentionCache = zeros(block_size * n_head * block_size);

      const normLinearGamma = await fetchBin(`${prefix}ln_2.weight_gpt.bin`);
      const normLinearBeta = bias ? await fetchBin(`${prefix}ln_2.bias_gpt.bin`) : zeros(n_embd);

      const firstLayerWeights = transpose(await fetchBin(`${prefix}mlp.c_fc.weight_gpt.bin`), hidden_size, n_embd);
      const firstLayerBias = bias ? await fetchBin(`${prefix}mlp.c_fc.bias_gpt.bin`) : zeros(hidden_size);

      const secondLayerWeights = transpose(await fetchBin(`${prefix}mlp.c_proj.weight_gpt.bin`), n_embd, hidden_size);
      const secondLayerBias = bias ? await fetchBin(`${prefix}mlp.c_proj.bias_gpt.bin`) : zeros(n_embd);

      layer_buffers.push({
        normAttentionGammaBuffer: this.initTensor(normAttentionGamma, [n_embd], ["storage"]),
        normAttentionBetaBuffer: this.initTensor(normAttentionBeta, [n_embd], ["storage"]),
        qkvWeightsBuffer: this.initTensor(qkvWeights, [n_embd, 3 * n_embd], ["storage"]),
        qkvBiasBuffer: this.initTensor(qkvBias, [3 * n_embd], ["storage"]),
        linearWeightsBuffer: this.initTensor(linearWeights, [n_embd, n_embd], ["storage"]),
        linearBiasBuffer: this.initTensor(linearBias, [n_embd], ["storage"]),
        normLinearGammaBuffer: this.initTensor(normLinearGamma, [n_embd], ["storage"]),
        normLinearBetaBuffer: this.initTensor(normLinearBeta, [n_embd], ["storage"]),
        firstLayerWeightsBuffer: this.initTensor(firstLayerWeights, [n_embd, hidden_size], ["storage"]),
        firstLayerBiasBuffer: this.initTensor(firstLayerBias, [hidden_size], ["storage"]),
        secondLayerWeightsBuffer: this.initTensor(secondLayerWeights, [hidden_size, n_embd], ["storage"]),
        secondLayerBiasBuffer: this.initTensor(secondLayerBias, [n_embd], ["storage"]),
        attentionCacheBuffer: this.initTensor(attentionCache, [block_size * n_head, block_size], ["storage", "copy_from", "copy_to"]),
      });
    }

    console.log("Loading final layer norm...");
    const layerNormGamma = await fetchBin(`${fldr}/transformer.ln_f.weight_gpt.bin`);
    const layerNormBeta = bias ? await fetchBin(`${fldr}/transformer.ln_f.bias_gpt.bin`) : zeros(n_embd);
    const normGammaBuffer = this.initTensor(layerNormGamma, [n_embd], ["storage"]);
    const normBetaBuffer = this.initTensor(layerNormBeta, [n_embd], ["storage"]);

    const output = { layer_buffers, embeddingWeightsBuffer, posEmbdBuffer, normGammaBuffer, normBetaBuffer };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async *generate(prompt, max_new_tokens, top_k, temperature) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    let history = this.tokenizer.encode(prompt);
    console.log(`Prompt (${history.length} tokens):\n${prompt}`);

    let totalTime = 0;

    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = history.slice(-this.params.block_size);
      const useAttCache = i !== 0 && history.length <= this.params.block_size && this.doAttentionCache;

      const startTime = performance.now();
      const logits = await this.run(idx_cond, useAttCache);
      const endTime = performance.now();

      console.log(`\nIteration ${i + 1} of ${max_new_tokens}`);
      console.log(`Using attention cache? ${useAttCache}`);
      console.log(`Kernel execution time: ${endTime - startTime} ms`);
      totalTime += endTime - startTime;

      const { topKIndices, topKProbs } = selectTopK(logits, top_k);
      const probs = cpuSoftmax(topKProbs, temperature);
      const idx_next = topKIndices[sampleFromDistribution(probs)];

      history = history.concat(idx_next);

      console.log(`Output:\n${this.tokenizer.decode(history)}`);

      yield this.tokenizer.decode([idx_next]);
    }

    console.log(`Average kernel execution time: ${totalTime / max_new_tokens} ms`);
  }

  async run(idx, useAttCache = false) {
    const { posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeightsBuffer } = this.model;
    const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size, block_size } = this.params;
    const seq_length = idx.length;

    const commandEncoder = this.device.createCommandEncoder();

    const embdOutputBuffer = this.initBuffer(["storage", "copy_to"], [seq_length, n_embd]);
    for (let i = 0; i < seq_length; i++) {
      commandEncoder.copyBufferToBuffer(
        embeddingWeightsBuffer,
        this.bufferSize(n_embd) * idx[i],
        embdOutputBuffer,
        this.bufferSize(n_embd) * i,
        this.bufferSize(n_embd)
      );
    }
    // Crop position embeddings.
    const posEmbdOutputBuffer = this.initBuffer(["storage", "copy_to"], [seq_length, n_embd]);
    commandEncoder.copyBufferToBuffer(posEmbdBuffer, 0, posEmbdOutputBuffer, 0, this.bufferSize(seq_length, n_embd));

    const embeddedInputBuffer = this.inlineResidual(commandEncoder, seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer); // Residual connection is just elementwise addition.

    let layerOutputBuffer = embeddedInputBuffer;
    for (let i = 0; i < n_layer; i++) {
      const buffers = layer_buffers[i];

      const layerNormAttentionOutputBuffer = this.inlineLayerNorm(
        commandEncoder,
        seq_length,
        n_embd,
        layerOutputBuffer,
        buffers.normAttentionGammaBuffer,
        buffers.normAttentionBetaBuffer
      );

      let attentionOutputBuffer;
      if (useAttCache) {
        attentionOutputBuffer = this.inlineFastCachedAttention(
          commandEncoder,
          seq_length,
          n_embd,
          attentionDotProductScale,
          layerNormAttentionOutputBuffer,
          n_head,
          buffers.qkvWeightsBuffer,
          buffers.qkvBiasBuffer,
          buffers.linearWeightsBuffer,
          buffers.linearBiasBuffer,
          buffers.attentionCacheBuffer
        );
      } else {
        attentionOutputBuffer = this.inlineFastAttention(
          commandEncoder,
          seq_length,
          n_embd,
          attentionDotProductScale,
          layerNormAttentionOutputBuffer,
          n_head,
          buffers.qkvWeightsBuffer,
          buffers.qkvBiasBuffer,
          buffers.linearWeightsBuffer,
          buffers.linearBiasBuffer,
          buffers.attentionCacheBuffer
        );
      }

      const residualAttentionOutputBuffer = this.inlineResidual(commandEncoder, seq_length, n_embd, attentionOutputBuffer, layerOutputBuffer);

      const layerNormLinearOutputBuffer = this.inlineLayerNorm(
        commandEncoder,
        seq_length,
        n_embd,
        residualAttentionOutputBuffer,
        buffers.normLinearGammaBuffer,
        buffers.normLinearBetaBuffer
      );

      let linearOutputBuffer;
      linearOutputBuffer = this.inlineFastFFN(
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

      const residualLinearOutputBuffer = this.inlineResidual(commandEncoder, seq_length, n_embd, linearOutputBuffer, residualAttentionOutputBuffer);

      layerOutputBuffer = residualLinearOutputBuffer;
    }

    const layerNormOutputBuffer = this.inlineLayerNorm(commandEncoder, seq_length, n_embd, layerOutputBuffer, normGammaBuffer, normBetaBuffer);

    const slicedEmbedOutputBuffer = this.initBuffer(["storage", "copy_to"], [n_embd]);
    commandEncoder.copyBufferToBuffer(layerNormOutputBuffer, this.bufferSize(seq_length - 1, n_embd), slicedEmbedOutputBuffer, 0, this.bufferSize(1, n_embd));

    const deEmbedOutputBuffer = this.initBuffer(["map_read", "copy_to"], [vocab_size]);

    // Assumes that vocab_size has a decent least prime factor.
    const maxStorageBufferSize = this.device.limits.maxStorageBufferBindingSize;
    const totalElements = this.bufferSize(vocab_size, n_embd);
    var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
    if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
    var vocabChunkSize = vocab_size / numInstances;

    for (let i = 0; i < numInstances; i++) {
      const deEmbedChunkInputBuffer = this.initBuffer(["storage", "copy_to"], [n_embd, vocabChunkSize]);
      commandEncoder.copyBufferToBuffer(
        embeddingWeightsBuffer,
        i * this.bufferSize(n_embd * vocabChunkSize),
        deEmbedChunkInputBuffer,
        0,
        this.bufferSize(n_embd, vocabChunkSize)
      );
      // We're doing some buffer tricks here. Since slicedEmbedOutputBuffer is a row matrix, we can just pretend it's a column matrix without any changes to the way it's stored. We then multiply it by the transposed embeddingWeights chunk, resulting in a column vector which, once again, we can pretend is a row vector.
      const deEmbedChunkResultBuffer = this.inlineMatMul(commandEncoder, deEmbedChunkInputBuffer, slicedEmbedOutputBuffer, vocabChunkSize, 1, n_embd);
      commandEncoder.copyBufferToBuffer(deEmbedChunkResultBuffer, 0, deEmbedOutputBuffer, i * this.bufferSize(vocabChunkSize), this.bufferSize(vocabChunkSize));
    }

    this.device.queue.submit([commandEncoder.finish()]);

    await deEmbedOutputBuffer.mapAsync(GPUMapMode.READ);
    const output = deEmbedOutputBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.

    this.destroyBuffers();

    return outputArray;
  }

  initOutputBuffer(commandEncoder, buffer, row, col) {
    const outputBuffer = this.initBuffer(["map_read", "copy_to"], [row, col]);
    commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, this.bufferSize(row, col));
    return outputBuffer;
  }

  initBuffer(ops, dims) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    this.bufferDeletionStack.push(buffer);
    return buffer;
  }

  initTensor(data, dims, ops) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    return Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }
}
