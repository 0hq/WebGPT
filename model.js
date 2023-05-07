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

    initializeOperations(this.device);

    [this.model, this.params] = await this.loadModel(this.folder);
    this.tokenizer = this.tokenizerType == "bpe" ? new GPT2Tokenizer() : new SimpleTokenizer();
    await this.tokenizer.load();

    if (this.params.n_embd % 4 !== 0 || this.params.n_head % 4 !== 0) {
      throw new Error("Model incompatible. n_embd and n_head must be divisible by 4 for fast matmul.");
    }

    if (this.tokenizerType == "bpe") {
      this.defaultPrompt = `What is the answer to life, the universe, and everything?\n`;
      this.defaultTopK = 3;
      this.defaultTemperature = 1;
      this.defaultTokens = 30;
    } else {
      this.defaultPrompt = `WILL:\nAh, how dare you challenge me?\nHave you forgotten I built WebGPT?\n`;
      this.defaultTopK = 2;
      this.defaultTemperature = 1;
      this.defaultTokens = 80;
    }

    this.initialized = true;

    console.log("Model initialized");
  }

  async *generate(prompt, max_new_tokens, top_k, temperature) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    let history = this.tokenizer.encode(prompt);
    console.log(`Prompt (${history.length} tokens):\n${prompt}`);

    const warmupRuns = 3;
    let totalTime = 0;

    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = history.slice(-this.params.n_ctx);
      const useAttCache = i !== 0 && history.length <= this.params.n_ctx;

      const startTime = performance.now();
      const logits = await this.run(idx_cond, useAttCache);
      const endTime = performance.now();

      // console.log(`\nIteration ${i + 1} of ${max_new_tokens}`);
      const lapsedTime = endTime - startTime;
      console.log(`Kernel execution time: ${lapsedTime} ms`);
      i >= warmupRuns && (totalTime += lapsedTime);

      const { topKIndices, topKProbs } = selectTopK(logits, top_k);
      const probs = cpuSoftmax(topKProbs, temperature);
      const idx_next = topKIndices[sampleFromDistribution(probs)];

      history = history.concat(idx_next);

      // console.log(`Output:\n${this.tokenizer.decode(history)}`);

      // const totalProbs = cpuSoftmax(logits, temperature);
      // const tokenProbsString = Array.from(totalProbs)
      //   .map((value, index) => ({ value, index }))
      //   .sort((a, b) => b.value - a.value)
      //   .slice(0, 8)
      //   .map((prob) => `{ ${this.tokenizer.decode([prob.index]).replace(/(\r\n|\n|\r)/gm, "newline")} } : ${prob.value.toPrecision(3)}`)
      //   .join(" | ");
      // console.log("Top 8 token probs:", tokenProbsString);

      yield this.tokenizer.decode([idx_next]);
    }

    console.log(`Average kernel execution time: ${totalTime / (max_new_tokens - warmupRuns)} ms`);
  }

  async run(idx) {
    const { posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingsBuffer, deEmbeddingsBuffers } = this.model;
    const { attention_scale, n_embd, n_head, head_size, n_layer, vocab_size, hidden_size, vocab_chunk_size, vocab_chunk_instances } = this.params;
    const seq_length = idx.length;

    // ---------------- Create Passes ---------------- //
    // Note: These are re-initialized because everytime seq_length changes buffers are different sizes.

    // Pipeline creation is major bottleneck to spin up speed! Also buffer re-use.

    this.computePasses = [];
    let intermediateBuffer;
    let residualBuffer;
    {
      const { passes, resultBuffer } = EmbedBlock.newInstance(idx, seq_length, n_embd, embeddingsBuffer, posEmbdBuffer, ResidualBlock);
      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    for (let i = 0; i < n_layer; i++) {
      const buffers = layer_buffers[i];
      {
        const { passes, resultBuffer } = LayerNormBlock.newInstance(
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
        const { passes, resultBuffer } = AttentionBlock.newFusedInstance(
          seq_length,
          n_embd,
          attention_scale,
          n_head,
          head_size,
          intermediateBuffer,
          buffers.qkvWeightArray[0],
          buffers.qkvBiasArray[0],
          buffers.qkvWeightArray[1],
          buffers.qkvBiasArray[1],
          buffers.qkvWeightArray[2],
          buffers.qkvBiasArray[2],
          buffers.linearWeightsBuffer,
          buffers.linearBiasBuffer,
          FastMatMulBlock,
          SoftmaxBlock
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = LayerNormBlock.newInstance(
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
        const { resultBuffer, passes } = FastMatMulBlock.newInstance(
          seq_length,
          hidden_size,
          n_embd,
          intermediateBuffer,
          buffers.firstLayerWeightsBuffer,
          buffers.firstLayerBiasBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { resultBuffer, passes } = GeluBlock.newInstance(seq_length, hidden_size, intermediateBuffer);
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { resultBuffer, passes } = FastMatMulBlock.newInstance(
          seq_length,
          n_embd,
          hidden_size,
          intermediateBuffer,
          buffers.secondLayerWeightsBuffer,
          buffers.secondLayerBiasBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
    }
    {
      const { passes, resultBuffer } = LayerNormBlock.newInstance(seq_length, n_embd, intermediateBuffer, normGammaBuffer, normBetaBuffer);
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    {
      const { passes, resultBuffer } = DeEmbedBlock.newInstance(
        n_embd,
        vocab_size,
        vocab_chunk_size * vocab_chunk_instances,
        seq_length,
        vocab_chunk_size,
        intermediateBuffer,
        deEmbeddingsBuffers
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

    destroyOperationBuffers();

    return outputArray;
  }

  async loadModel(folder) {
    if (this.initialized) return console.error("Model already loaded");

    console.log("Loading model from folder:", folder);
    const weightsFolder = `weights/${folder}/`;

    const params = await this.loadParameters(weightsFolder);
    const { embeddingsBuffer, deEmbeddingsBuffers } = await this.loadEmbeddings(params, weightsFolder);
    const { posEmbdBuffer } = await this.loadPositionalEmbeddings(params, weightsFolder);
    const layer_buffers = await this.loadLayers(params, weightsFolder);

    console.log("Loading final layer norm...");
    const { normGammaBuffer, normBetaBuffer } = await this.loadFinalLayerNorm(params, weightsFolder);

    const output = { layer_buffers, embeddingsBuffer, deEmbeddingsBuffers, posEmbdBuffer, normGammaBuffer, normBetaBuffer };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async loadParameters(weightsFolder) {
    console.log("Loading params...");
    const params = await (await fetch(`${weightsFolder}/params_gpt.json`)).json();

    // Did you enable GitHub LFS? Won't work without it.
    if (params.n_embd % 4 !== 0) throw new Error("Model load failed: n_embd must be divisible by 4.");
    if (params.n_embd % params.n_head !== 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");
    // I'm unsure if this is a reasonable requirement here. At worst, I can figure out some padding method.
    if ((params.n_embd / params.n_head) % 4 !== 0) throw new Error("Model load failed: n_embd / n_head must be divisible by 4.");
    const tokenParam = this.bufferSize(params.vocab_size, params.n_embd);
    let minSplits = Math.ceil(tokenParam / this.device.limits.maxStorageBufferBindingSize);
    function vocabChunkSizeCalc(vocab_size, n_embd, splits, maxStorageBufferBindingSize) {
      // Possibly could be better? Needs actual benchmarking to know what approach is best.
      const optimisticSize = Math.ceil(vocab_size / splits / 4) * 4 * n_embd;
      const pessimiticSize = Math.floor(vocab_size / splits / 4) * 4 * n_embd;
      let vocab_chunk_size = optimisticSize;
      if (optimisticSize > maxStorageBufferBindingSize) {
        vocab_chunk_size = pessimiticSize;
        if (pessimiticSize * splits < tokenParam) {
          return vocabChunkSizeCalc(vocab_size, n_embd, splits + 1, maxStorageBufferBindingSize);
        }
      }
      return { vocab_chunk_size: vocab_chunk_size / n_embd, splits };
    }
    const { vocab_chunk_size, splits } = vocabChunkSizeCalc(params.vocab_size, params.n_embd, minSplits, this.device.limits.maxStorageBufferBindingSize);
    if (splits > minSplits) console.warn(`Non-optimal number of vocab splits. Optimal: ${minSplits}, Selected: ${splits}`);

    // Set derived parameters
    params.vocab_chunk_size = vocab_chunk_size;
    params.vocab_chunk_instances = splits;
    params.head_size = params.n_embd / params.n_head;
    params.hidden_size = params.n_embd * 4;
    params.attention_scale = 1 / Math.sqrt(params.n_embd / params.n_head);
    params.bias = params.bias == undefined ? true : params.bias;

    return params;
  }

  async loadEmbeddings(params, weightsFolder) {
    console.log("Loading token embeddings...");
    const embeddingWeights = await fetchBin(`${weightsFolder}/transformer.wte.weight_gpt.bin`);
    const embeddingsBuffer = this.initTensor(embeddingWeights, [params.vocab_size, params.n_embd], ["copy_from"]);

    // Chunks are stored in row-major order and are of dimensions n_embd x vocab_chunk_size.
    // Embedding weights are imported in column-major order and are of dimensions vocab_size x n_embd.
    // We pre-transpose the chunk for the deEmbedding process for the matmul. Could do this on GPU later.
    const deEmbeddingsBuffers = [];
    for (let i = 0; i < params.vocab_chunk_instances; i++) {
      console.log(`Loading deEmbedding chunk ${i + 1}/${params.vocab_chunk_instances}...`);
      const offset = i * params.vocab_chunk_size;
      let size = params.vocab_chunk_size;
      const paddedArray = new Float32Array(params.vocab_chunk_size * params.n_embd);
      if (i === params.vocab_chunk_instances - 1) {
        size = params.vocab_size - offset;
        paddedArray.set(size * params.n_embd, zeros((params.vocab_chunk_size * params.vocab_chunk_instances - params.vocab_size) * params.n_embd));
      }
      paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));
      const chunk = transpose(paddedArray, params.vocab_chunk_size, params.n_embd); // Use GPU perhaps?
      deEmbeddingsBuffers.push(this.initTensor(chunk, [params.n_embd, params.vocab_chunk_size], ["storage"]));
    }

    return { embeddingsBuffer, deEmbeddingsBuffers };
  }

  async loadPositionalEmbeddings(params, weightsFolder) {
    console.log("Loading positional embeddings...");
    const posEmbeddings = await fetchBin(`${weightsFolder}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, [params.n_ctx, params.n_embd], ["copy_from"]);

    return { posEmbdBuffer };
  }

  async loadFinalLayerNorm(params, weightsFolder) {
    console.log("Loading final norm...");
    const prefix = `${weightsFolder}/transformer.ln_f.`;

    const tensorPromises = [
      this.fetchAndInitTensor(`${prefix}weight_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}bias_gpt.bin`, [params.n_embd], ["storage"]),
    ];

    const [normGammaBuffer, normBetaBuffer] = await Promise.all(tensorPromises);

    return { normGammaBuffer, normBetaBuffer };
  }

  async loadLayers(params, weightsFolder) {
    console.log("Loading layers...");
    const layerPromises = [];

    for (let i = 0; i < params.n_layer; i++) {
      layerPromises.push(this.loadLayer(params, weightsFolder, i));
    }

    const layer_buffers = await Promise.all(layerPromises);
    return layer_buffers;
  }

  async loadLayer(params, weightsFolder, layerIndex) {
    console.log("Starting to load layer...", layerIndex);
    const prefix = `${weightsFolder}transformer.h.${layerIndex}.`;

    // Create an array of promises for fetching and initializing the tensors
    const tensorPromises = [
      this.fetchAndInitTensor(`${prefix}ln_1.weight_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}ln_1.bias_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndSplitQKVWeightTensors(`${prefix}attn.c_attn.weight_gpt.bin`, [params.n_embd, 3 * params.n_embd], ["storage"]),
      this.fetchAndSplitQKVBiasTensors(`${prefix}attn.c_attn.bias_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}attn.c_proj.weight_gpt.bin`, [params.n_embd, params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}attn.c_proj.bias_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}ln_2.weight_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}ln_2.bias_gpt.bin`, [params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_fc.weight_gpt.bin`, [params.n_embd, params.hidden_size], ["storage"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_fc.bias_gpt.bin`, [params.hidden_size], ["storage"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_proj.weight_gpt.bin`, [params.hidden_size, params.n_embd], ["storage"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_proj.bias_gpt.bin`, [params.n_embd], ["storage"]),
    ];

    // Wait for all tensors to be fetched and initialized
    const [
      normAttentionGammaBuffer,
      normAttentionBetaBuffer,
      qkvWeightArray,
      qkvBiasArray,
      linearWeightsBuffer,
      linearBiasBuffer,
      normLinearGammaBuffer,
      normLinearBetaBuffer,
      firstLayerWeightsBuffer,
      firstLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerBiasBuffer,
    ] = await Promise.all(tensorPromises);

    // Process the fetched data and return the layer buffers
    return {
      normAttentionGammaBuffer,
      normAttentionBetaBuffer,
      qkvWeightArray,
      qkvBiasArray,
      linearWeightsBuffer,
      linearBiasBuffer,
      normLinearGammaBuffer,
      normLinearBetaBuffer,
      firstLayerWeightsBuffer,
      firstLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerBiasBuffer,
    };
  }

  async fetchAndSplitQKVWeightTensors(url, dims, ops) {
    const data = transpose(await fetchBin(url), dims[0], dims[1]);

    const qWeights = transpose(data.subarray(0, dims[0] * dims[0]), dims[0], dims[0]);
    const kWeights = transpose(data.subarray(dims[0] * dims[0], dims[0] * dims[0] * 2), dims[0], dims[0]);
    const vWeights = transpose(data.subarray(dims[0] * dims[0] * 2, dims[0] * dims[0] * 3), dims[0], dims[0]);

    const qWeightsBuffer = this.initTensor(qWeights, [dims[0], dims[0]], ops);
    const kWeightsBuffer = this.initTensor(kWeights, [dims[0], dims[0]], ops);
    const vWeightsBuffer = this.initTensor(vWeights, [dims[0], dims[0]], ops);

    return [qWeightsBuffer, kWeightsBuffer, vWeightsBuffer];
  }

  async fetchAndSplitQKVBiasTensors(url, dims, ops) {
    const data = await fetchBin(url);

    const qBias = data.subarray(0, dims[0]);
    const kBias = data.subarray(dims[0], dims[0] * 2);
    const vBias = data.subarray(dims[0] * 2, dims[0] * 3);

    const qBiasBuffer = this.initTensor(qBias, [dims[0]], ops);
    const kBiasBuffer = this.initTensor(kBias, [dims[0]], ops);
    const vBiasBuffer = this.initTensor(vBias, [dims[0]], ops);

    return [qBiasBuffer, kBiasBuffer, vBiasBuffer];
  }

  async fetchAndInitTensor(url, dims, ops) {
    const data = await fetchBin(url);
    return this.initTensor(data, dims, ops);
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
