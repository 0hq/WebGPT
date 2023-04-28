class GPT {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device = null;
    this.model = null;
    this.tokenizer = null;
    this.params = null;
    this.doFastMatMul = false;
    this.minStorageBufferOffsetAlignment = 1;

    this.bufferDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    [this.model, this.params] = await this.loadModel(this.folder);
    this.tokenizer = this.tokenizerType == "bpe" ? new GPT2Tokenizer() : new SimpleTokenizer();
    await this.tokenizer.load();

    this.initBindGroups();
    this.initPipelines();

    if (this.params.n_embd % 4 !== 0 || this.params.n_head % 4 !== 0) {
      throw new Error("Model incompatible. n_embd and n_head must be divisible by 4 for fast matmul.");
    }

    this.initialized = true;
  }

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

    if (n_embd % n_head != 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

    console.log("Loading token embeddings...");
    const embeddingWeights = await fetchBin(`${fldr}/transformer.wte.weight_gpt.bin`);
    const embeddingWeightsBuffer = this.initTensor(embeddingWeights, vocab_size, n_embd, ["copy_from"]);

    console.log("Loading positional embeddings...");
    const posEmbeddings = await fetchBin(`${fldr}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, block_size, n_embd, ["copy_from"]);

    const layer_buffers = [];
    for (let i = 0; i < n_layer; i++) {
      console.log("Loading layer", i);
      const prefix = `${fldr}transformer.h.${i}.`;

      console.log("\tLoading attention layer norm...");
      const normAttentionGamma = await fetchBin(`${prefix}ln_1.weight_gpt.bin`);
      const normAttentionBeta = bias ? await fetchBin(`${prefix}ln_1.bias_gpt.bin`) : zeros(n_embd);

      console.log("\tLoading qkv transform...");
      const qkvWeights = transpose(await fetchBin(`${prefix}attn.c_attn.weight_gpt.bin`), 3 * n_embd, n_embd);
      const qkvBias = bias ? await fetchBin(`${prefix}attn.c_attn.bias_gpt.bin`) : zeros(3 * n_embd);

      console.log("\tLoading attention c_proj...");
      const linearWeights = transpose(await fetchBin(`${prefix}attn.c_proj.weight_gpt.bin`), n_embd, n_embd);
      const linearBias = bias ? await fetchBin(`${prefix}attn.c_proj.bias_gpt.bin`) : zeros(n_embd);

      console.log("\tInitializing attention cache...");
      const attentionCache = zeros(block_size * n_head * block_size);

      console.log("\tLoading MLP layer norm...");
      const normLinearGamma = await fetchBin(`${prefix}ln_2.weight_gpt.bin`);
      const normLinearBeta = bias ? await fetchBin(`${prefix}ln_2.bias_gpt.bin`) : zeros(n_embd);

      console.log("\tLoading MLP first layer...");
      const firstLayerWeights = transpose(await fetchBin(`${prefix}mlp.c_fc.weight_gpt.bin`), hidden_size, n_embd);
      const firstLayerBias = bias ? await fetchBin(`${prefix}mlp.c_fc.bias_gpt.bin`) : zeros(hidden_size);

      console.log("\tLoading MLP second layer...");
      const secondLayerWeights = transpose(await fetchBin(`${prefix}mlp.c_proj.weight_gpt.bin`), n_embd, hidden_size);
      const secondLayerBias = bias ? await fetchBin(`${prefix}mlp.c_proj.bias_gpt.bin`) : zeros(n_embd);

      layer_buffers.push({
        normAttentionGammaBuffer: this.initTensor(normAttentionGamma, n_embd, 1, ["storage"]),
        normAttentionBetaBuffer: this.initTensor(normAttentionBeta, n_embd, 1, ["storage"]),
        qkvWeightsBuffer: this.initTensor(qkvWeights, n_embd, 3 * n_embd, ["storage"]),
        qkvBiasBuffer: this.initTensor(qkvBias, 3 * n_embd, 1, ["storage"]),
        linearWeightsBuffer: this.initTensor(linearWeights, n_embd, n_embd, ["storage"]),
        linearBiasBuffer: this.initTensor(linearBias, n_embd, 1, ["storage"]),
        normLinearGammaBuffer: this.initTensor(normLinearGamma, n_embd, 1, ["storage"]),
        normLinearBetaBuffer: this.initTensor(normLinearBeta, n_embd, 1, ["storage"]),
        firstLayerWeightsBuffer: this.initTensor(firstLayerWeights, n_embd, hidden_size, ["storage"]),
        firstLayerBiasBuffer: this.initTensor(firstLayerBias, hidden_size, 1, ["storage"]),
        secondLayerWeightsBuffer: this.initTensor(secondLayerWeights, hidden_size, n_embd, ["storage"]),
        secondLayerBiasBuffer: this.initTensor(secondLayerBias, n_embd, 1, ["storage"]),
        attentionCacheBuffer: this.initTensor(attentionCache, block_size * n_head, block_size, ["storage", "copy_from"]),
      });
    }

    console.log("Loading final layer norm...");
    const layerNormGamma = await fetchBin(`${fldr}/transformer.ln_f.weight_gpt.bin`);
    const layerNormBeta = bias ? await fetchBin(`${fldr}/transformer.ln_f.bias_gpt.bin`) : zeros(n_embd);
    const normGammaBuffer = this.initTensor(layerNormGamma, n_embd, 1, ["storage"]);
    const normBetaBuffer = this.initTensor(layerNormBeta, n_embd, 1, ["storage"]);

    const output = { layer_buffers, embeddingWeightsBuffer, posEmbdBuffer, normGammaBuffer, normBetaBuffer };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async profile(prompt, tokens, runs, retries) {
    if (!this.initialized) return console.error("Model not loaded yet");

    let avgRetryTime = 0;
    for (let t = 0; t < retries; t++) {
      let avgRunTime = 0;
      for (let r = 0; r < runs; r++) {
        let history = this.tokenizer.encode(prompt);
        let totalTime = 0;
        for (let i = 0; i < tokens; i++) {
          const idx_cond = history.slice(-this.params.block_size);

          const startTime = performance.now();
          const logits = await this.run(idx_cond, false);
          const endTime = performance.now();

          totalTime += endTime - startTime;

          const { topKIndices, topKProbs } = selectTopK(logits, 3);
          const probs = cpuSoftmax(topKProbs, 1.0);
          const idx_next = topKIndices[sampleFromDistribution(probs)];

          history = history.concat(idx_next);
        }
        const avgTime = totalTime / tokens;
        console.log(`Run ${r + 1} of ${runs}: Average kernel execution time: ${avgTime} ms`);
        avgRunTime += avgTime;
      }
      console.log(`Average kernel execution time over ${runs} runs: ${avgRunTime / runs} ms`);
      avgRetryTime += avgRunTime / runs;
    }
    console.log(`Average kernel execution time over ${retries} retries: ${avgRetryTime / retries} ms`);
  }

  async *generate(prompt, max_new_tokens, top_k = 10, temperature = 1.0) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    console.log("Starting generation with prompt", prompt);
    let history = this.tokenizer.encode(prompt);

    let totalTime = 0;

    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = history.slice(-this.params.block_size);

      const startTime = performance.now();
      const logits = await this.run(idx_cond, i !== 0 && history.length <= block_size);
      const endTime = performance.now();

      console.log(`(Loop ${i}) Kernel execution time: ${endTime - startTime} ms`);
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
    const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size } = this.params;
    const seq_length = idx.length;

    const commandEncoder = this.device.createCommandEncoder();

    const embdOutputBuffer = this.initBuffer(["storage", "copy_to"], seq_length, n_embd);
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
    const posEmbdOutputBuffer = this.initBuffer(["storage", "copy_to"], seq_length, n_embd);
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

    const slicedEmbedOutputBuffer = this.initBuffer(["storage", "copy_to"], 1, n_embd);
    commandEncoder.copyBufferToBuffer(layerNormOutputBuffer, this.bufferSize(seq_length - 1, n_embd), slicedEmbedOutputBuffer, 0, this.bufferSize(1, n_embd));

    const deEmbedOutputBuffer = this.initBuffer(["map_read", "copy_to"], 1, vocab_size, true);

    // Assumes that vocab_size has a decent least prime factor.
    const maxStorageBufferSize = this.device.limits.maxStorageBufferBindingSize;
    const totalElements = this.bufferSize(vocab_size, n_embd);
    var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
    if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
    var vocabChunkSize = vocab_size / numInstances;

    for (let i = 0; i < numInstances; i++) {
      const deEmbedChunkInputBuffer = this.initBuffer(["storage", "copy_to"], n_embd, vocabChunkSize);
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
    const outputArray = new Float32Array(output);

    for (let i = 0; i < this.bufferDeletionStack.length; i++) {
      this.bufferDeletionStack[i].destroy();
    }

    return outputArray;
  }

  maskedInlineSoftmax(commandEncoder, rows, cols, inputBuffer) {
    const dimUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    this.device.queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

    const maxResultBuffer = this.initBuffer(["storage", "copy_from"], rows);
    const maxBindGroup = this.initBindGroup(this.u_s_Layout, [dimUniformBuffer, maxResultBuffer]);

    const addExpResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const addExpBindGroup = this.initBindGroup(this.u_s_Layout, [dimUniformBuffer, addExpResultBuffer]);

    const sumResultBuffer = this.initBuffer(["storage", "copy_from"], rows);
    const sumBindGroup = this.initBindGroup(this.u_s_Layout, [dimUniformBuffer, sumResultBuffer]);

    const divResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const divBindGroup = this.initBindGroup(this.u_s_Layout, [dimUniformBuffer, divResultBuffer]);

    const passEncoder_max = commandEncoder.beginComputePass();
    passEncoder_max.setPipeline(this.maskedMaxPipeline);
    passEncoder_max.setBindGroup(0, maxBindGroup);
    passEncoder_max.setBindGroup(1, this.initBindGroup(this.r_Layout, [inputBuffer]));
    passEncoder_max.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder_max.end();

    const passEncoder_addExp = commandEncoder.beginComputePass();
    passEncoder_addExp.setPipeline(this.addExpPipeline);
    passEncoder_addExp.setBindGroup(0, addExpBindGroup);
    passEncoder_addExp.setBindGroup(1, this.initBindGroup(this.r_Layout, [inputBuffer]));
    passEncoder_addExp.setBindGroup(2, this.initBindGroup(this.r_Layout, [maxResultBuffer]));
    passEncoder_addExp.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder_addExp.end();

    const passEncoder_sum = commandEncoder.beginComputePass();
    passEncoder_sum.setPipeline(this.sumPipeline);
    passEncoder_sum.setBindGroup(0, sumBindGroup);
    passEncoder_sum.setBindGroup(1, this.initBindGroup(this.r_Layout, [addExpResultBuffer]));
    passEncoder_sum.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder_sum.end();

    const passEncoder_div = commandEncoder.beginComputePass();
    passEncoder_div.setPipeline(this.dividePipeline);
    passEncoder_div.setBindGroup(0, divBindGroup);
    passEncoder_div.setBindGroup(1, this.initBindGroup(this.r_Layout, [addExpResultBuffer]));
    passEncoder_div.setBindGroup(2, this.initBindGroup(this.r_Layout, [sumResultBuffer]));
    passEncoder_div.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder_div.end();

    return divResultBuffer;
  }

  inlineResidual(commandEncoder, rows, cols, layerOutputBuffer, residualBuffer) {
    const residualUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const residualResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const residualBindGroup = this.initBindGroup(this.u_s_Layout, [residualUniformBuffer, residualResultBuffer]);
    this.device.queue.writeBuffer(residualUniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.elementAddPipeline);
    passEncoder.setBindGroup(0, residualBindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_Layout, [residualBuffer]));
    passEncoder.setBindGroup(2, this.initBindGroup(this.r_Layout, [layerOutputBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder.end();

    return residualResultBuffer;
  }

  inlineMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    const matmulUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const matmulResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const matMulBindGroup = this.initBindGroup(this.u_s_Layout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.matmulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder.end();

    return matmulResultBuffer;
  }

  inlineFastMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    if (cols % 4 !== 0 || shared % 4 !== 0) throw new Error(`cols and shared must be div by 4, got ${rows}x${cols}x${shared}`);

    const matmulUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const matmulResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const matMulBindGroup = this.initBindGroup(this.u_s_Layout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fastMatMulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(wgSize(cols, 64), wgSize(rows, 32));
    passEncoder.end();

    return matmulResultBuffer;
  }

  inlineTranspose(commandEncoder, inputBuffer, rows, cols) {
    const transposeUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const transposeResultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const transposeBindGroup = this.initBindGroup(this.u_s_Layout, [transposeUniformBuffer, transposeResultBuffer]);
    this.device.queue.writeBuffer(transposeUniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.transposePipeline);
    passEncoder.setBindGroup(0, transposeBindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_Layout, [inputBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(rows, 16), wgSize(cols, 16));
    passEncoder.end();

    return transposeResultBuffer;
  }

  inlineLayerNorm(commandEncoder, seq_length, n_embd, inputBuffer, gammaBuffer, betaBuffer) {
    const statsUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const statsResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, 2);
    const statsBindGroup = this.initBindGroup(this.u_s_Layout, [statsUniformBuffer, statsResultBuffer]);
    this.device.queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const normUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const normResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, n_embd);
    const normBindGroup = this.initBindGroup(this.u_s_Layout, [normUniformBuffer, normResultBuffer]);
    this.device.queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const passEncoder_stats = commandEncoder.beginComputePass();
    passEncoder_stats.setPipeline(this.statsPipeline);
    passEncoder_stats.setBindGroup(0, statsBindGroup);
    passEncoder_stats.setBindGroup(1, this.initBindGroup(this.r_Layout, [inputBuffer]));
    passEncoder_stats.dispatchWorkgroups(wgSize(seq_length, 16));
    passEncoder_stats.end();

    const passEncoder_norm = commandEncoder.beginComputePass();
    passEncoder_norm.setPipeline(this.normPipeline);
    passEncoder_norm.setBindGroup(0, normBindGroup);
    passEncoder_norm.setBindGroup(1, this.initBindGroup(this.r_r_r_Layout, [inputBuffer, gammaBuffer, betaBuffer]));
    passEncoder_norm.setBindGroup(2, this.initBindGroup(this.r_Layout, [statsResultBuffer]));
    passEncoder_norm.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(n_embd, 16));
    passEncoder_norm.end();

    return normResultBuffer;
  }

  inlineFastFFN(
    commandEncoder,
    seq_length,
    n_embed,
    hidden_size,
    inputBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer
  ) {
    const geluUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const geluResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, hidden_size);
    const geluBindGroup = this.initBindGroup(this.u_s_Layout, [geluUniformBuffer, geluResultBuffer]);
    this.device.queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([seq_length, hidden_size]));

    const firstLayerMatMulBuffer = this.inlineFastMatMul(commandEncoder, inputBuffer, firstLayerWeightsBuffer, seq_length, hidden_size, n_embed);
    const firstLayerResultBuffer = this.inlineFastRowAdd(commandEncoder, firstLayerMatMulBuffer, firstLayerBiasBuffer, seq_length, hidden_size);

    const passEncoder_gelu = commandEncoder.beginComputePass();
    passEncoder_gelu.setPipeline(this.GELUpipeline);
    passEncoder_gelu.setBindGroup(0, geluBindGroup);
    passEncoder_gelu.setBindGroup(1, this.initBindGroup(this.r_Layout, [firstLayerResultBuffer]));
    passEncoder_gelu.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(hidden_size, 16));
    passEncoder_gelu.end();

    const secondLayerMatMulBuffer = this.inlineFastMatMul(commandEncoder, geluResultBuffer, secondLayerWeightsBuffer, seq_length, n_embed, hidden_size);
    const secondLayerResultBuffer = this.inlineFastRowAdd(commandEncoder, secondLayerMatMulBuffer, secondLayerBiasBuffer, seq_length, n_embed);

    return secondLayerResultBuffer;
  }

  inlineFastRowAdd(commandEncoder, inputBuffer, biasBuffer, rows, cols) {
    if (cols % 4 !== 0) throw new Error(`cols must be a multiple of 4, got ${rows}x${cols}`);

    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], rows, cols);
    const bindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer]);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, cols / 4]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.fastRowAddPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [inputBuffer, biasBuffer]));
    passEncoder.dispatchWorkgroups(wgSize(rows, 8), wgSize(cols, 32));
    passEncoder.end();

    return resultBuffer;
  }

  inlineFastAttention(
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    inputBuffer,
    n_head,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    attentionCacheBuffer
  ) {
    const splitQKVUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const splitQResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, n_embd);
    const splitKResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, n_embd);
    const splitVResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, n_embd);
    const splitQKVBindGroup = this.initBindGroup(this.u_s_s_s_Layout, [splitQKVUniformBuffer, splitQResultBuffer, splitKResultBuffer, splitVResultBuffer]);
    this.device.queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const attentionWeightsUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const attentionWeightsResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, seq_length * n_head);
    const attentionWeightsBindGroup = this.initBindGroup(this.u_s_Layout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head, n_embd / n_head, n_embd]));

    const multiplyUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const multiplyResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, seq_length * n_head);
    const multiplyBindGroup = this.initBindGroup(this.u_s_Layout, [multiplyUniformBuffer, multiplyResultBuffer]);
    this.device.queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head]));
    this.device.queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

    const causalMaskUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const causalMaskResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, seq_length * n_head);
    const causalMaskBindGroup = this.initBindGroup(this.u_s_Layout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
    this.device.queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length])); // Transposes! This is needed for softmax.

    const attentionValuesUniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const attentionValuesResultBuffer = this.initBuffer(["storage", "copy_from"], seq_length, n_embd);
    const attentionValuesBindGroup = this.initBindGroup(this.u_s_Layout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    this.device.queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));

    const qkvMatmulBuffer = this.inlineFastMatMul(commandEncoder, inputBuffer, qkvWeightsBuffer, seq_length, 3 * n_embd, n_embd);
    const qkvResultBuffer = this.inlineFastRowAdd(commandEncoder, qkvMatmulBuffer, qkvBiasBuffer, seq_length, 3 * n_embd);

    const passEncoder_splitQKV = commandEncoder.beginComputePass();
    passEncoder_splitQKV.setPipeline(this.splitQKVpipeline);
    passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
    passEncoder_splitQKV.setBindGroup(1, this.initBindGroup(this.r_Layout, [qkvResultBuffer]));
    passEncoder_splitQKV.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(n_embd, 16));
    passEncoder_splitQKV.end();

    const passEncoder_attentionWeights = commandEncoder.beginComputePass();
    passEncoder_attentionWeights.setPipeline(this.attentionWeightsPipeline);
    passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
    passEncoder_attentionWeights.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [splitQResultBuffer, splitKResultBuffer]));
    passEncoder_attentionWeights.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(seq_length * n_head, 16));
    passEncoder_attentionWeights.end();

    const passEncoder_multiply = commandEncoder.beginComputePass();
    passEncoder_multiply.setPipeline(this.multiplyPipeline);
    passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
    passEncoder_multiply.setBindGroup(1, this.initBindGroup(this.r_Layout, [attentionWeightsResultBuffer]));
    passEncoder_multiply.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(seq_length * n_head, 16));
    passEncoder_multiply.end();

    const passEncoder_causalMask = commandEncoder.beginComputePass();
    passEncoder_causalMask.setPipeline(this.causalMaskPipeline);
    passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
    passEncoder_causalMask.setBindGroup(1, this.initBindGroup(this.r_Layout, [multiplyResultBuffer]));
    passEncoder_causalMask.dispatchWorkgroups(wgSize(seq_length * n_head, 16), wgSize(seq_length, 16));
    passEncoder_causalMask.end();

    // Save attention weights for next iteration's key value cache.
    commandEncoder.copyBufferToBuffer(causalMaskResultBuffer, 0, attentionCacheBuffer, 0, bufferSizeCalc(seq_length * n_head, seq_length));

    const softmaxOutputBuffer = this.maskedInlineSoftmax(commandEncoder, seq_length * n_head, seq_length, causalMaskResultBuffer);

    const passEncoder_attentionValues = commandEncoder.beginComputePass();
    passEncoder_attentionValues.setPipeline(this.attentionValuesPipeline);
    passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
    passEncoder_attentionValues.setBindGroup(1, this.initBindGroup(this.r_r_Layout, [softmaxOutputBuffer, splitVResultBuffer]));
    passEncoder_attentionValues.dispatchWorkgroups(wgSize(seq_length, 16), wgSize(n_embd, 16));
    passEncoder_attentionValues.end();

    const linearMatmulBuffer = this.inlineFastMatMul(commandEncoder, attentionValuesResultBuffer, linearWeightsBuffer, seq_length, n_embd, n_embd);
    const linearResultBuffer = this.inlineFastRowAdd(commandEncoder, linearMatmulBuffer, linearBiasBuffer, seq_length, n_embd);

    return linearResultBuffer;
  }

  // Old code, needs to be updated with fast matmul and verified.
  cachedInlineAttention(
    device,
    queue,
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    inputBuffer,
    n_head,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer,
    attentionCacheBuffer
  ) {
    // This could be cached as well.
    const qkvUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const qkvResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const qkvBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [qkvUniformBuffer, qkvBiasBuffer, qkvWeightsBuffer, qkvResultBuffer]);
    queue.writeBuffer(qkvUniformBuffer, 0, new Uint32Array([seq_length, 3 * n_embd, n_embd]));

    const splitQKVUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const splitQResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitKResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitVResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitQKVBindGroup = createBindGroup(device, u_s_s_s_BindLayout, [splitQKVUniformBuffer, splitQResultBuffer, splitKResultBuffer, splitVResultBuffer]);
    queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const singleQResultBuffer = createBuffer(device, bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const attentionWeightsUniformBuffer = createBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const attentionWeightsResultBuffer = createBuffer(
      device,
      bufferSizeCalc(1, seq_length * n_head),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const attentionWeightsBindGroup = createBindGroup(device, u_s_BindLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
    queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([1, seq_length * n_head, seq_length, n_embd / n_head, n_embd]));

    const multiplyUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const multiplyResultBuffer = createBuffer(device, bufferSizeCalc(1, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const multiplyBindGroup = createBindGroup(device, u_s_BindLayout, [multiplyUniformBuffer, multiplyResultBuffer]);
    queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([1, seq_length * n_head]));
    queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

    const causalMaskCachedResultBuffer = createBuffer(
      device,
      bufferSizeCalc(seq_length * n_head, seq_length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    const attentionValuesUniformBuffer = createBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const attentionValuesResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const attentionValuesBindGroup = createBindGroup(device, u_s_BindLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));

    const linearUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const linearResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const linearBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [linearUniformBuffer, linearBiasBuffer, linearWeightsBuffer, linearResultBuffer]);
    queue.writeBuffer(linearUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_embd]));

    const passEncoder_qkv = commandEncoder.beginComputePass();
    passEncoder_qkv.setPipeline(FFNpipeline);
    passEncoder_qkv.setBindGroup(0, qkvBindGroup);
    passEncoder_qkv.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
    passEncoder_qkv.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(3 * n_embd, workgroup_X));
    passEncoder_qkv.end();

    const passEncoder_splitQKV = commandEncoder.beginComputePass();
    passEncoder_splitQKV.setPipeline(splitQKVpipeline);
    passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
    passEncoder_splitQKV.setBindGroup(1, createBindGroup(device, r_BindLayout, [qkvResultBuffer]));
    passEncoder_splitQKV.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_splitQKV.end();

    commandEncoder.copyBufferToBuffer(splitQResultBuffer, bufferSizeCalc(n_embd) * (seq_length - 1), singleQResultBuffer, 0, bufferSizeCalc(n_embd));

    const passEncoder_attentionWeights = commandEncoder.beginComputePass();
    passEncoder_attentionWeights.setPipeline(attentionWeightsNewPipeline);
    passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
    passEncoder_attentionWeights.setBindGroup(1, createBindGroup(device, r_r_BindLayout, [singleQResultBuffer, splitKResultBuffer]));
    passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(1, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
    passEncoder_attentionWeights.end();

    const passEncoder_multiply = commandEncoder.beginComputePass();
    passEncoder_multiply.setPipeline(multiplyPipeline);
    passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
    passEncoder_multiply.setBindGroup(1, createBindGroup(device, r_BindLayout, [attentionWeightsResultBuffer]));
    passEncoder_multiply.dispatchWorkgroups(workgroupCalc(1, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
    passEncoder_multiply.end();

    for (let i = 0; i < n_head; i++) {
      commandEncoder.copyBufferToBuffer(
        multiplyResultBuffer,
        bufferSizeCalc(seq_length) * i,
        causalMaskCachedResultBuffer,
        bufferSizeCalc(seq_length, seq_length) * (i + 1) - bufferSizeCalc(seq_length),
        bufferSizeCalc(seq_length)
      );
    }
    for (let i = 0; i < n_head; i++) {
      for (let j = 0; j < seq_length - 1; j++) {
        commandEncoder.copyBufferToBuffer(
          attentionCacheBuffer,
          bufferSizeCalc(seq_length - 1) * j + bufferSizeCalc(seq_length - 1, seq_length - 1) * i,
          causalMaskCachedResultBuffer,
          bufferSizeCalc(seq_length) * j + bufferSizeCalc(seq_length, seq_length) * i,
          bufferSizeCalc(seq_length - 1)
        );
      }
    }

    // Save for next iteration.
    commandEncoder.copyBufferToBuffer(causalMaskCachedResultBuffer, 0, attentionCacheBuffer, 0, bufferSizeCalc(seq_length * n_head, seq_length));

    const softmaxOutputBuffer = maskedInlineSoftmax(device, queue, commandEncoder, seq_length * n_head, seq_length, causalMaskCachedResultBuffer);

    const passEncoder_attentionValues = commandEncoder.beginComputePass();
    passEncoder_attentionValues.setPipeline(attentionValuesPipeline);
    passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
    passEncoder_attentionValues.setBindGroup(1, createBindGroup(device, r_r_BindLayout, [softmaxOutputBuffer, splitVResultBuffer]));
    passEncoder_attentionValues.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_attentionValues.end();

    const passEncoder_linear = commandEncoder.beginComputePass();
    passEncoder_linear.setPipeline(FFNpipeline);
    passEncoder_linear.setBindGroup(0, linearBindGroup);
    passEncoder_linear.setBindGroup(1, createBindGroup(device, r_BindLayout, [attentionValuesResultBuffer]));
    passEncoder_linear.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_linear.end();

    return linearResultBuffer;
  }

  initBindGroup(layout, buffers) {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
    });
  }

  initBuffer(ops, row, col = 1, noDelete = false) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    if (!noDelete) this.bufferDeletionStack.push(buffer);
    return buffer;
  }

  initTensor(data, sizeA, sizeB, ops) {
    const buffer = this.initBuffer([...ops, "copy_to"], sizeA, sizeB, true);
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  bufferSize(dimA, dimB = 1) {
    return alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, this.minStorageBufferOffsetAlignment);
  }

  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  initPipelines() {
    const p = (code, bindGroupLayouts) => {
      return this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts }),
        compute: {
          module: this.device.createShaderModule({ code }),
          entryPoint: "main",
        },
      });
    };

    this.statsPipeline = p(normStatsShader, [this.u_s_Layout, this.r_Layout]);
    this.normPipeline = p(normShader, [this.u_s_Layout, this.r_r_r_Layout, this.r_Layout]);
    this.GELUpipeline = p(GELUShader, [this.u_s_Layout, this.r_Layout]);
    this.splitQKVpipeline = p(splitQKVShader, [this.u_s_s_s_Layout, this.r_Layout]);
    this.attentionWeightsPipeline = p(attentionWeightsShader, [this.u_s_Layout, this.r_r_Layout]);
    this.attentionValuesPipeline = p(attentionValuesShader, [this.u_s_Layout, this.r_r_Layout]);
    this.multiplyPipeline = p(multiplyShader, [this.u_s_Layout, this.r_Layout]);
    this.causalMaskPipeline = p(causalMaskShader, [this.u_s_Layout, this.r_Layout]);
    this.matmulPipeline = p(matMulShader, [this.u_s_Layout, this.r_r_Layout]);
    this.elementAddPipeline = p(elementWiseAdditionShader, [this.u_s_Layout, this.r_Layout, this.r_Layout]);
    this.maskedMaxPipeline = p(maskedNegMaxShader, [this.u_s_Layout, this.r_Layout]);
    this.addPipeline = p(addShader, [this.u_s_Layout, this.r_Layout, this.r_Layout]);
    this.addExpPipeline = p(addExpShader, [this.u_s_Layout, this.r_Layout, this.r_Layout]);
    this.sumPipeline = p(sumShader, [this.u_s_Layout, this.r_Layout]);
    this.dividePipeline = p(divideShader, [this.u_s_Layout, this.r_Layout, this.r_Layout]);
    this.transposePipeline = p(transposeShader, [this.u_s_Layout, this.r_Layout]);
    this.fastMatMulPipeline = p(fastMatMulShader, [this.u_s_Layout, this.r_r_Layout]);
    this.fastRowAddPipeline = p(fastRowAddShader, [this.u_s_Layout, this.r_r_Layout]);
  }
}

// (async () => {
//   const GPTModel = new GPT("gpt2", "bpe");
//   await GPTModel.initialize();
//   const prompt = "Hello, my name is";
//   await GPTModel.profile(prompt, 10, 5, 3);
// })();
