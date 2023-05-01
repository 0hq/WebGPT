class GPT {
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
      this.defaultTokens = 20;
    }

    this.NaiveMatMulBlock = new NaiveMatMulBlock(this.device);
    this.FastMatMulBlock = new FastMatMulBlock(this.device);
    this.FastRowAddBlock = new FastRowAddBlock(this.device);
    this.FastFFNBlock = new FastFFNBlock(this.device);
    this.AttentionBlock = new AttentionBlock(this.device);
    this.ResidualBlock = new ResidualBlock(this.device);
    this.EmbedBlock = new EmbedBlock(this.device);
    this.DeEmbedBlock = new DeEmbedBlock(this.device);
    this.GeluBlock = new GeluBlock(this.device);
    this.LayerNormBlock = new LayerNormBlock(this.device);
    this.TransposeBlock = new TransposeBlock(this.device);
    this.SoftmaxBlock = new SoftmaxBlock(this.device);

    // Needed for deletion.
    this.operations = [
      this.NaiveMatMulBlock,
      this.FastMatMulBlock,
      this.FastRowAddBlock,
      this.FastFFNBlock,
      this.AttentionBlock,
      this.ResidualBlock,
      this.EmbedBlock,
      this.DeEmbedBlock,
      this.GeluBlock,
      this.LayerNormBlock,
      this.TransposeBlock,
      this.SoftmaxBlock,
    ];

    this.initialized = true;

    console.log("Model initialized");
  }

  // Fetch bin should be parallelized for big loading time reduction.
  async loadModel(folder) {
    if (this.initialized) return console.error("Model already loaded");

    console.log("Loading model from folder:", folder);
    const fldr = `models/${folder}/`;
    const zeros = (dim) => new Float32Array(dim).fill(0);

    console.log("Loading params...");
    const params = await (await fetch(`${fldr}/params_gpt.json`)).json();
    params.hidden_size = params.n_embd * 4;
    params.attentionDotProductScale = 1 / Math.sqrt(params.n_embd / params.n_head);
    const { block_size, n_embd, n_head, n_layer, bias, vocab_size, hidden_size } = params;
    console.log("Params:", params);

    // Did you enable GitHub LFS? Won't work without it.
    if (n_embd % n_head != 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

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

  async run(idx) {
    const { posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeightsBuffer } = this.model;
    const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size, block_size } = this.params;
    const seq_length = idx.length;

    // ---------------- Create Passes ----------------

    this.computePasses = [];
    let intermediateBuffer;
    let residualBuffer;
    {
      const { passes, resultBuffer } = this.EmbedBlock.newInstance(idx, seq_length, n_embd, embeddingWeightsBuffer, posEmbdBuffer, this.ResidualBlock);
      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    for (let i = 0; i < n_layer; i++) {
      const buffers = layer_buffers[i];
      {
        const { passes, resultBuffer } = this.LayerNormBlock.newInstance(
          seq_length,
          n_embd,
          intermediateBuffer,
          buffers.normAttentionGammaBuffer,
          buffers.normAttentionBetaBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = this.AttentionBlock.newInstance(
          seq_length,
          n_embd,
          attentionDotProductScale,
          n_head,
          intermediateBuffer,
          buffers.qkvWeightsBuffer,
          buffers.qkvBiasBuffer,
          buffers.linearWeightsBuffer,
          buffers.linearBiasBuffer,
          this.FastMatMulBlock,
          this.FastRowAddBlock,
          this.SoftmaxBlock
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = this.ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = this.LayerNormBlock.newInstance(
          seq_length,
          n_embd,
          intermediateBuffer,
          buffers.normLinearGammaBuffer,
          buffers.normLinearBetaBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = this.FastFFNBlock.newInstance(
          seq_length,
          n_embd,
          hidden_size,
          intermediateBuffer,
          buffers.firstLayerWeightsBuffer,
          buffers.firstLayerBiasBuffer,
          buffers.secondLayerWeightsBuffer,
          buffers.secondLayerBiasBuffer,
          this.FastMatMulBlock,
          this.FastRowAddBlock,
          this.GeluBlock
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = this.ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
    }
    {
      const { passes, resultBuffer } = this.LayerNormBlock.newInstance(seq_length, n_embd, intermediateBuffer, normGammaBuffer, normBetaBuffer);
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    {
      const { passes, resultBuffer } = this.DeEmbedBlock.newInstance(
        vocab_size,
        n_embd,
        seq_length,
        intermediateBuffer,
        embeddingWeightsBuffer,
        this.NaiveMatMulBlock
      );
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    const resultBuffer = intermediateBuffer;

    // ---------------- Compute Passes ----------------

    const commandEncoder = this.device.createCommandEncoder();
    for (const pass of this.computePasses) {
      if (pass.flag === "compute") {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pass.pipeline);
        for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
        passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
        passEncoder.end();
      } else if (pass.flag === "copy") {
        commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);

    // ---------------- Read Results ----------------

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const output = resultBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.

    for (const operation of this.operations) operation.destroyBuffers();

    return outputArray;
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

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    return Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }
}
