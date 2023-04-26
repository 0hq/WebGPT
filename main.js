// TODO: Optimize workgroup size and remove global size.
const workgroup_X = 16; // Dictated by shader.
const workgroup_Y = 16; // Dictated by shader.

class GPT {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device = null;
    this.model = null;
    this.tokenizer = null;
    this.params = null;
    this.minStorageBufferOffsetAlignment = 1;
  }

  getName() {
    return this.folder;
  }

  bufferSizeCalc(dimA, dimB = 1) {
    return alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, this.minStorageBufferOffsetAlignment);
  }

  initBindGroups() {
    this.r_r_r_BindLayout = createBindGroupLayout(this.device, ["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_BindLayout = createBindGroupLayout(this.device, ["read-only-storage", "read-only-storage"]);
    this.r_BindLayout = createBindGroupLayout(this.device, ["read-only-storage"]);
    this.u_r_r_s_BindLayout = createBindGroupLayout(this.device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
    this.u_s_BindLayout = createBindGroupLayout(this.device, ["uniform", "storage"]);
    this.u_s_s_s_BindLayout = createBindGroupLayout(this.device, ["uniform", "storage", "storage", "storage"]);
  }

  initPipelines() {
    this.statsPipeline = createPipeline(this.device, normStatsShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.normPipeline = createPipeline(this.device, normShader, [this.u_s_BindLayout, this.r_r_r_BindLayout, this.r_BindLayout]);
    this.FFNpipeline = createPipeline(this.device, FFNShader, [this.u_r_r_s_BindLayout, this.r_BindLayout]);
    this.GELUpipeline = createPipeline(this.device, GELUShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.splitQKVpipeline = createPipeline(this.device, splitQKVShader, [this.u_s_s_s_BindLayout, this.r_BindLayout]);
    this.attentionWeightsPipeline = createPipeline(this.device, attentionWeightsShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.attentionValuesPipeline = createPipeline(this.device, attentionValuesShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.multiplyPipeline = createPipeline(this.device, multiplyShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.causalMaskPipeline = createPipeline(this.device, causalMaskShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.simpleCausalMaskPipeline = createPipeline(this.device, simpleCausalMaskShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.matmulPipeline = createPipeline(this.device, matMulShader, [this.u_s_BindLayout, this.r_r_BindLayout]);
    this.elementAddPipeline = createPipeline(this.device, elementWiseAdditionShader, [this.u_s_BindLayout, this.r_BindLayout, this.r_BindLayout]);
    this.maxPipeline = createPipeline(this.device, negMaxShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.maskedMaxPipeline = createPipeline(this.device, maskedNegMaxShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.addPipeline = createPipeline(this.device, addShader, [this.u_s_BindLayout, this.r_BindLayout, this.r_BindLayout]);
    this.expPipeline = createPipeline(this.device, expShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.addExpPipeline = createPipeline(this.device, addExpShader, [this.u_s_BindLayout, this.r_BindLayout, this.r_BindLayout]);
    this.sumPipeline = createPipeline(this.device, sumShader, [this.u_s_BindLayout, this.r_BindLayout]);
    this.dividePipeline = createPipeline(this.device, divideShader, [this.u_s_BindLayout, this.r_BindLayout, this.r_BindLayout]);
    this.transposePipeline = createPipeline(this.device, transposeShader, [this.u_s_BindLayout, this.r_BindLayout]);
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    [this.model, this.params] = await this.loadModel(this.folder);
    this.tokenizer = this.loadTokenizer(this.tokenizerType);
    await this.tokenizer.load();

    this.initBindGroups();
    this.initPipelines();

    this.initialized = true;
  }

  loadTokenizer(type) {
    if (type == "bpe") return new GPT2Tokenizer();
    else if (type == "char") return new SimpleTokenizer();
    else throw new Error("Unknown tokenizer type: " + type);
  }

  async loadModel(folder) {
    if (this.initialized) {
      console.error("Model already loaded");
      return;
    }

    // Should be this.device.limits.minStorageBufferOffsetAlignment once bug is fixed.

    console.log("Loading model from folder:", folder);
    const params = await (await fetch(`models/${folder}/params_gpt.json`)).json();
    const { block_size, n_embd, n_head, n_layer, bias, vocab_size } = params;
    params.hidden_size = n_embd * 4;
    params.attentionDotProductScale = 1 / Math.sqrt(n_embd / n_head);
    const { hidden_size } = params;
    console.log("Params:", params);

    if (n_embd % n_head != 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

    console.log("Loading token embeddings...");
    const embeddingWeights = await loadBinaryFile("models/" + folder + "/transformer.wte.weight_gpt.bin");
    const embeddingWeightsBuffer = createBuffer(this.device, this.bufferSizeCalc(vocab_size, n_embd), GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.device.queue.writeBuffer(embeddingWeightsBuffer, 0, embeddingWeights);

    console.log("Loading positional embeddings...");
    const posEmbeddings = await loadBinaryFile("models/" + folder + "/transformer.wpe.weight_gpt.bin");
    const posEmbdBuffer = createBuffer(this.device, this.bufferSizeCalc(block_size, n_embd), GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.device.queue.writeBuffer(posEmbdBuffer, 0, posEmbeddings);

    const layer_buffers = [];
    for (let i = 0; i < n_layer; i++) {
      console.log("Loading layer", i);
      const prefix = `transformer.h.${i}.`;

      console.log("\tLoading attention layer norm...");
      const normAttentionGammaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const normAttentionBetaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(normAttentionGammaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.weight_gpt.bin`));
      this.device.queue.writeBuffer(normAttentionBetaBuffer, 0, await loadBinaryFile(`models/${folder}/${prefix}ln_1.bias_gpt.bin`));

      console.log("\tLoading qkv transform...");
      const qkv_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.weight_gpt.bin`);
      const qkv_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_attn.bias_gpt.bin`) : new Array(3 * n_embd).fill(0);
      const qkvWeightsBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const qkvBiasBuffer = createBuffer(this.device, this.bufferSizeCalc(3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(qkvWeightsBuffer, 0, transposeArray(qkv_weights, 3 * n_embd, n_embd));
      this.device.queue.writeBuffer(qkvBiasBuffer, 0, qkv_bias);

      console.log("\tLoading attention c_proj...");
      const linear_weights = await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.weight_gpt.bin`);
      const linear_bias = bias ? await loadBinaryFile(`models/${folder}/${prefix}attn.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
      const linearWeightsBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const linearBiasBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(linearWeightsBuffer, 0, transposeArray(linear_weights, n_embd, n_embd));
      this.device.queue.writeBuffer(linearBiasBuffer, 0, linear_bias);

      console.log("\tLoading MLP layer norm...");
      const layerNormLinearGamma = await loadBinaryFile(`models/${folder}/${prefix}ln_2.weight_gpt.bin`);
      const layerNormLinearBeta = bias ? await loadBinaryFile(`models/${folder}/${prefix}ln_2.bias_gpt.bin`) : new Array(n_embd).fill(0);
      const normLinearGammaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const normLinearBetaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(normLinearGammaBuffer, 0, layerNormLinearGamma);
      this.device.queue.writeBuffer(normLinearBetaBuffer, 0, layerNormLinearBeta);

      console.log("\tLoading MLP first layer...");
      const firstLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.weight_gpt.bin`);
      const firstLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_fc.bias_gpt.bin`) : new Array(hidden_size).fill(0);
      const firstLayerWeightsBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const firstLayerBiasBuffer = createBuffer(this.device, this.bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(firstLayerWeightsBuffer, 0, transposeArray(firstLayerWeights, hidden_size, n_embd));
      this.device.queue.writeBuffer(firstLayerBiasBuffer, 0, firstLayerBias);

      console.log("\tLoading MLP second layer...");
      const secondLayerWeights = await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.weight_gpt.bin`);
      const secondLayerBias = bias ? await loadBinaryFile(`models/${folder}/${prefix}mlp.c_proj.bias_gpt.bin`) : new Array(n_embd).fill(0);
      const secondLayerWeightsBuffer = createBuffer(this.device, this.bufferSizeCalc(hidden_size, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const secondLayerBiasBuffer = createBuffer(this.device, this.bufferSizeCalc(hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(secondLayerWeightsBuffer, 0, transposeArray(secondLayerWeights, n_embd, hidden_size));
      this.device.queue.writeBuffer(secondLayerBiasBuffer, 0, secondLayerBias);

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
    const normGammaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const normBetaBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(normGammaBuffer, 0, layerNormGamma);
    this.device.queue.writeBuffer(normBetaBuffer, 0, layerNormBeta);

    const output = {
      layer_buffers,
      embeddingWeightsBuffer,
      posEmbdBuffer,
      normGammaBuffer,
      normBetaBuffer,
    };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async *generate(prompt, max_new_tokens, top_k = 10, temperature = 1.0) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    console.log("Starting generation with prompt", prompt);
    let history = this.tokenizer.encode(prompt);

    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = history.slice(-this.params.block_size);

      const startTime = performance.now();
      const logits = await this.run(idx_cond);
      const endTime = performance.now();

      console.log(`Kernel execution time: ${endTime - startTime} ms`);

      const { topKIndices, topKProbs } = selectTopK(logits, top_k);
      const probs = cpuSoftmax(topKProbs, temperature);
      const idx_next = topKIndices[sampleFromDistribution(probs)];

      history = history.concat(idx_next);

      console.log(`Output:\n${this.tokenizer.decode(history)}`);

      yield this.tokenizer.decode([idx_next]);
    }
  }

  async run(idx) {
    const { posEmbdBuffer, layer_buffers, normGammaBuffer, normBetaBuffer, embeddingWeightsBuffer } = this.model;
    const { attentionDotProductScale, n_embd, n_head, n_layer, vocab_size, hidden_size } = this.params;
    const seq_length = idx.length;

    const commandEncoder = this.device.createCommandEncoder();

    const embdOutputBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    for (let i = 0; i < seq_length; i++) {
      commandEncoder.copyBufferToBuffer(
        embeddingWeightsBuffer,
        this.bufferSizeCalc(n_embd) * idx[i],
        embdOutputBuffer,
        this.bufferSizeCalc(n_embd) * i,
        this.bufferSizeCalc(n_embd)
      );
    }

    // Crop the position embeddings to the correct size.
    const posEmbdOutputBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    commandEncoder.copyBufferToBuffer(
      posEmbdBuffer,
      0, // Source offset (starting from the beginning of the buffer)
      posEmbdOutputBuffer, // Destination buffer (cropped buffer)
      0, // Destination offset (starting from the beginning of the cropped buffer)
      this.bufferSizeCalc(seq_length, n_embd) // Number of bytes to copy
    );
    // Residual connection is just elementwise addition, can be used for combining embedding and position embedding.
    const embeddedInputBuffer = this.inlineResidual(commandEncoder, seq_length, n_embd, embdOutputBuffer, posEmbdOutputBuffer);
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

      const attentionOutputBuffer = this.inlineAttention(
        commandEncoder,
        seq_length,
        n_embd,
        attentionDotProductScale,
        layerNormAttentionOutputBuffer,
        n_head,
        buffers.qkvWeightsBuffer,
        buffers.qkvBiasBuffer,
        buffers.linearWeightsBuffer,
        buffers.linearBiasBuffer
      );

      const residualAttentionOutputBuffer = this.inlineResidual(commandEncoder, seq_length, n_embd, attentionOutputBuffer, layerOutputBuffer);

      const layerNormLinearOutputBuffer = this.inlineLayerNorm(
        commandEncoder,
        seq_length,
        n_embd,
        residualAttentionOutputBuffer,
        buffers.normLinearGammaBuffer,
        buffers.normLinearBetaBuffer
      );

      const linearOutputBuffer = this.inlineFFN(
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

    /* 
      Compute Pass Splitting: Possible Approaches.
  
      The limiting factor running larger models is often the maxStorageBufferBindingSize, which prevents you from doing giant matrix multiplications. As far as I know, the best way to solve this is to generate multiple compute passes and split the calculation into smaller sub-matrices. You can simply write back to the buffer with the proper offset and byte size and nothing changes. As long as these operations don't have inter-dependencies, WebGPU should recognize that they can be run in parallel and you shouldn't experience significant performance losses. This needs to be verified!
  
      The question then is to how to divide the operation properly for efficiency. I'm still figuring out how everything works, so I'm unsure what the most efficient way to do this is.
  
      The first thought is that if you have a matrix of elements maxStorageBufferBindingSize * 2, it's straightforward to chop it down the middle. However for non-evenly divisible matrix sizes, you might run into serious memory inefficiencies if you divide by the minimum number of sub-matrices. 
  
      I've implemented/planned a couple different solutions.
  
      (1) Start with the minimum number of groups and calculate wasted memory, then increase # of groups and record the most efficient sizing up to some maximum group number. This calculation can be done when the model is loaded.
  
      (2) Divide by some standard matrix size (maybe a square matrix of rows * rows) and then add one final "overflow matrix" of some irregular size. I really don't know if this is more efficient, still learning, but my gut tells me this might be result in too many groups when fewer could do better.
      
      (3) Assume that the matrix has some decently small factor that fits perfectly and use that. This is the simplest solution, and given that I have 0 clue which option is best until I test, I'm going with this for now.
  
    */

    const slicedEmbedOutputBuffer = createBuffer(this.device, this.bufferSizeCalc(1, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    commandEncoder.copyBufferToBuffer(
      layerNormOutputBuffer, // Source buffer (original position embeddings)
      this.bufferSizeCalc(seq_length - 1, n_embd), // Source offset (starting from the beginning of the buffer)
      slicedEmbedOutputBuffer, // Destination buffer (cropped buffer)
      0, // Destination offset (starting from the beginning of the cropped buffer)
      this.bufferSizeCalc(1, n_embd) // Number of bytes to copy
    );

    const deEmbedOutputBuffer = createBuffer(this.device, this.bufferSizeCalc(1, vocab_size), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    // Assumes that vocab_size has a decent least prime factor.
    const maxStorageBufferSize = this.device.limits.maxStorageBufferBindingSize;
    const totalElements = this.bufferSizeCalc(vocab_size, n_embd);
    var numInstances = Math.ceil(totalElements / maxStorageBufferSize);
    if (numInstances > 1) numInstances = leastPrimeFactor(vocab_size, numInstances);
    var vocabChunkSize = vocab_size / numInstances;

    for (let i = 0; i < numInstances; i++) {
      const deEmbedChunkInputBuffer = createBuffer(this.device, this.bufferSizeCalc(n_embd, vocabChunkSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      // Remember that embeddingWeightsBuffer is transposed.
      commandEncoder.copyBufferToBuffer(
        embeddingWeightsBuffer,
        i * this.bufferSizeCalc(n_embd * vocabChunkSize),
        deEmbedChunkInputBuffer,
        0,
        this.bufferSizeCalc(n_embd, vocabChunkSize)
      );
      // We're doing some buffer tricks here. Since slicedEmbedOutputBuffer is a row matrix, we can just pretend it's a column matrix without any changes to the way it's stored. We then multiply it by the transposed embeddingWeights chunk, resulting in a column vector which, once again, we can pretend is a row vector.
      const deEmbedChunkResultBuffer = this.inlineMatMul(commandEncoder, deEmbedChunkInputBuffer, slicedEmbedOutputBuffer, vocabChunkSize, 1, n_embd);
      commandEncoder.copyBufferToBuffer(
        deEmbedChunkResultBuffer,
        0,
        deEmbedOutputBuffer,
        i * this.bufferSizeCalc(vocabChunkSize),
        this.bufferSizeCalc(vocabChunkSize)
      );
    }

    this.device.queue.submit([commandEncoder.finish()]);

    await deEmbedOutputBuffer.mapAsync(GPUMapMode.READ);
    const output = deEmbedOutputBuffer.getMappedRange();

    return new Float32Array(output);
  }

  inlineSoftmax(commandEncoder, rows, cols, inputBuffer) {
    const dimUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

    const maxResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const maxBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, maxResultBuffer]);

    const addResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const addBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, addResultBuffer]);

    const expResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const expBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, expResultBuffer]);

    const sumResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const sumBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, sumResultBuffer]);

    const divResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const divBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, divResultBuffer]);

    const passEncoder_max = commandEncoder.beginComputePass();
    passEncoder_max.setPipeline(this.maxPipeline);
    passEncoder_max.setBindGroup(0, maxBindGroup);
    passEncoder_max.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_max.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_max.end();

    const passEncoder_add = commandEncoder.beginComputePass();
    passEncoder_add.setPipeline(this.addPipeline);
    passEncoder_add.setBindGroup(0, addBindGroup);
    passEncoder_add.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_add.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [maxResultBuffer]));
    passEncoder_add.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_add.end();

    const passEncoder_exp = commandEncoder.beginComputePass();
    passEncoder_exp.setPipeline(this.expPipeline);
    passEncoder_exp.setBindGroup(0, expBindGroup);
    passEncoder_exp.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [addResultBuffer]));
    passEncoder_exp.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_exp.end();

    const passEncoder_sum = commandEncoder.beginComputePass();
    passEncoder_sum.setPipeline(this.sumPipeline);
    passEncoder_sum.setBindGroup(0, sumBindGroup);
    passEncoder_sum.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [expResultBuffer]));
    passEncoder_sum.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_sum.end();

    const passEncoder_div = commandEncoder.beginComputePass();
    passEncoder_div.setPipeline(this.dividePipeline);
    passEncoder_div.setBindGroup(0, divBindGroup);
    passEncoder_div.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [expResultBuffer]));
    passEncoder_div.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [sumResultBuffer]));
    passEncoder_div.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_div.end();

    return divResultBuffer;
  }

  maskedInlineSoftmax(commandEncoder, rows, cols, inputBuffer) {
    const dimUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

    const maxResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const maxBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, maxResultBuffer]);

    const addExpResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const addExpBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, addExpResultBuffer]);

    const sumResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const sumBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, sumResultBuffer]);

    const divResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const divBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [dimUniformBuffer, divResultBuffer]);

    const passEncoder_max = commandEncoder.beginComputePass();
    passEncoder_max.setPipeline(this.maskedMaxPipeline);
    passEncoder_max.setBindGroup(0, maxBindGroup);
    passEncoder_max.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_max.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_max.end();

    const passEncoder_addExp = commandEncoder.beginComputePass();
    passEncoder_addExp.setPipeline(this.addExpPipeline);
    passEncoder_addExp.setBindGroup(0, addExpBindGroup);
    passEncoder_addExp.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_addExp.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [maxResultBuffer]));
    passEncoder_addExp.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_addExp.end();

    const passEncoder_sum = commandEncoder.beginComputePass();
    passEncoder_sum.setPipeline(this.sumPipeline);
    passEncoder_sum.setBindGroup(0, sumBindGroup);
    passEncoder_sum.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [addExpResultBuffer]));
    passEncoder_sum.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_sum.end();

    const passEncoder_div = commandEncoder.beginComputePass();
    passEncoder_div.setPipeline(this.dividePipeline);
    passEncoder_div.setBindGroup(0, divBindGroup);
    passEncoder_div.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [addExpResultBuffer]));
    passEncoder_div.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [sumResultBuffer]));
    passEncoder_div.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder_div.end();

    return divResultBuffer;
  }

  inlineResidual(commandEncoder, rows, cols, layerOutputBuffer, residualBuffer) {
    const residualUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const residualResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const residualBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [residualUniformBuffer, residualResultBuffer]);
    this.device.queue.writeBuffer(residualUniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.elementAddPipeline);
    passEncoder.setBindGroup(0, residualBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [residualBuffer]));
    passEncoder.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [layerOutputBuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder.end();

    return residualResultBuffer;
  }

  inlineMatMul(commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
    const matmulUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const matmulResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const matMulBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
    this.device.queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.matmulPipeline);
    passEncoder.setBindGroup(0, matMulBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [Abuffer, Bbuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder.end();

    return matmulResultBuffer;
  }

  inlineTranspose(commandEncoder, inputBuffer, rows, cols) {
    const transposeUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const transposeResultBuffer = createBuffer(this.device, this.bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const transposeBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [transposeUniformBuffer, transposeResultBuffer]);
    this.device.queue.writeBuffer(transposeUniformBuffer, 0, new Uint32Array([rows, cols]));

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.transposePipeline);
    passEncoder.setBindGroup(0, transposeBindGroup);
    passEncoder.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
    passEncoder.end();

    return transposeResultBuffer;
  }

  inlineLayerNorm(commandEncoder, seq_length, n_embd, inputBuffer, gammaBuffer, betaBuffer) {
    const statsUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const statsResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const statsBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [statsUniformBuffer, statsResultBuffer]);
    this.device.queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const normUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const normResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const normBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [normUniformBuffer, normResultBuffer]);
    this.device.queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const passEncoder_stats = commandEncoder.beginComputePass();
    passEncoder_stats.setPipeline(this.statsPipeline);
    passEncoder_stats.setBindGroup(0, statsBindGroup);
    passEncoder_stats.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_stats.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y));
    passEncoder_stats.end();

    const passEncoder_norm = commandEncoder.beginComputePass();
    passEncoder_norm.setPipeline(this.normPipeline);
    passEncoder_norm.setBindGroup(0, normBindGroup);
    passEncoder_norm.setBindGroup(1, createBindGroup(this.device, this.r_r_r_BindLayout, [inputBuffer, gammaBuffer, betaBuffer]));
    passEncoder_norm.setBindGroup(2, createBindGroup(this.device, this.r_BindLayout, [statsResultBuffer]));
    passEncoder_norm.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_norm.end();

    return normResultBuffer;
  }

  inlineFFN(
    commandEncoder,
    context,
    n_embed,
    hidden_size,
    inputBuffer,
    firstLayerWeightsBuffer,
    firstLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerBiasBuffer
  ) {
    const firstLayerUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const firstLayerResultBuffer = createBuffer(this.device, this.bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const firstLayerBindGroup = createBindGroup(this.device, this.u_r_r_s_BindLayout, [
      firstLayerUniformBuffer,
      firstLayerBiasBuffer,
      firstLayerWeightsBuffer,
      firstLayerResultBuffer,
    ]);
    this.device.queue.writeBuffer(firstLayerUniformBuffer, 0, new Uint32Array([context, hidden_size, n_embed]));

    const geluUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const geluResultBuffer = createBuffer(this.device, this.bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const geluBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [geluUniformBuffer, geluResultBuffer]);
    this.device.queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([context, hidden_size]));

    const secondLayerUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const secondLayerResultBuffer = createBuffer(this.device, this.bufferSizeCalc(context, n_embed), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const secondLayerBindGroup = createBindGroup(this.device, this.u_r_r_s_BindLayout, [
      secondLayerUniformBuffer,
      secondLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerResultBuffer,
    ]);
    this.device.queue.writeBuffer(secondLayerUniformBuffer, 0, new Uint32Array([context, n_embed, hidden_size]));

    const passEncoder_first = commandEncoder.beginComputePass();
    passEncoder_first.setPipeline(this.FFNpipeline);
    passEncoder_first.setBindGroup(0, firstLayerBindGroup);
    passEncoder_first.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_first.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
    passEncoder_first.end();

    const passEncoder_gelu = commandEncoder.beginComputePass();
    passEncoder_gelu.setPipeline(this.GELUpipeline);
    passEncoder_gelu.setBindGroup(0, geluBindGroup);
    passEncoder_gelu.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [firstLayerResultBuffer]));
    passEncoder_gelu.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
    passEncoder_gelu.end();

    const passEncoder_second = commandEncoder.beginComputePass();
    passEncoder_second.setPipeline(this.FFNpipeline);
    passEncoder_second.setBindGroup(0, secondLayerBindGroup);
    passEncoder_second.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [geluResultBuffer]));
    passEncoder_second.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(n_embed, workgroup_X));
    passEncoder_second.end();

    return secondLayerResultBuffer;
  }

  inlineAttention(
    commandEncoder,
    seq_length,
    n_embd,
    attentionDotProductScale,
    inputBuffer,
    n_head,
    qkvWeightsBuffer,
    qkvBiasBuffer,
    linearWeightsBuffer,
    linearBiasBuffer
  ) {
    const qkvUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const qkvResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const qkvBindGroup = createBindGroup(this.device, this.u_r_r_s_BindLayout, [qkvUniformBuffer, qkvBiasBuffer, qkvWeightsBuffer, qkvResultBuffer]);
    this.device.queue.writeBuffer(qkvUniformBuffer, 0, new Uint32Array([seq_length, 3 * n_embd, n_embd]));

    const splitQKVUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const splitQResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitKResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitVResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const splitQKVBindGroup = createBindGroup(this.device, this.u_s_s_s_BindLayout, [
      splitQKVUniformBuffer,
      splitQResultBuffer,
      splitKResultBuffer,
      splitVResultBuffer,
    ]);
    this.device.queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

    const attentionWeightsUniformBuffer = createBuffer(this.device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const attentionWeightsResultBuffer = createBuffer(
      this.device,
      this.bufferSizeCalc(seq_length, seq_length * n_head),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const attentionWeightsBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
    this.device.queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head, n_embd / n_head, n_embd]));

    const multiplyUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const multiplyResultBuffer = createBuffer(
      this.device,
      this.bufferSizeCalc(seq_length, seq_length * n_head),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const multiplyBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [multiplyUniformBuffer, multiplyResultBuffer]);
    this.device.queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head]));
    this.device.queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

    const causalMaskUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const causalMaskResultBuffer = createBuffer(
      this.device,
      this.bufferSizeCalc(seq_length, seq_length * n_head),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const causalMaskBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
    this.device.queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length])); // Transposes! This is needed for softmax.

    const attentionValuesUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const attentionValuesResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const attentionValuesBindGroup = createBindGroup(this.device, this.u_s_BindLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
    this.device.queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));

    const linearUniformBuffer = createBuffer(this.device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const linearResultBuffer = createBuffer(this.device, this.bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const linearBindGroup = createBindGroup(this.device, this.u_r_r_s_BindLayout, [
      linearUniformBuffer,
      linearBiasBuffer,
      linearWeightsBuffer,
      linearResultBuffer,
    ]);
    this.device.queue.writeBuffer(linearUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_embd]));

    const passEncoder_qkv = commandEncoder.beginComputePass();
    passEncoder_qkv.setPipeline(this.FFNpipeline);
    passEncoder_qkv.setBindGroup(0, qkvBindGroup);
    passEncoder_qkv.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [inputBuffer]));
    passEncoder_qkv.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(3 * n_embd, workgroup_X));
    passEncoder_qkv.end();

    const passEncoder_splitQKV = commandEncoder.beginComputePass();
    passEncoder_splitQKV.setPipeline(this.splitQKVpipeline);
    passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
    passEncoder_splitQKV.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [qkvResultBuffer]));
    passEncoder_splitQKV.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_splitQKV.end();

    const passEncoder_attentionWeights = commandEncoder.beginComputePass();
    passEncoder_attentionWeights.setPipeline(this.attentionWeightsPipeline);
    passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
    passEncoder_attentionWeights.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [splitQResultBuffer, splitKResultBuffer]));
    passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
    passEncoder_attentionWeights.end();

    const passEncoder_multiply = commandEncoder.beginComputePass();
    passEncoder_multiply.setPipeline(this.multiplyPipeline);
    passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
    passEncoder_multiply.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [attentionWeightsResultBuffer]));
    passEncoder_multiply.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
    passEncoder_multiply.end();

    const passEncoder_causalMask = commandEncoder.beginComputePass();
    passEncoder_causalMask.setPipeline(this.causalMaskPipeline);
    passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
    passEncoder_causalMask.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [multiplyResultBuffer]));
    passEncoder_causalMask.dispatchWorkgroups(workgroupCalc(seq_length * n_head, workgroup_Y), workgroupCalc(seq_length, workgroup_X));
    passEncoder_causalMask.end();

    // This is a sloppy-ish solution to the casual mask buffer being processed with every head at once. Obviously, this could be fixed if we just did this in a smarter way but I only realized you could do this at the end. Still learning WebGPU!
    const softmaxOutputBuffer = createBuffer(
      this.device,
      this.bufferSizeCalc(seq_length, seq_length * n_head),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    for (let i = 0; i < n_head; i++) {
      const softmaxInputBuffer = createBuffer(
        this.device,
        this.bufferSizeCalc(seq_length, seq_length),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      );
      commandEncoder.copyBufferToBuffer(
        causalMaskResultBuffer,
        i * this.bufferSizeCalc(seq_length, seq_length),
        softmaxInputBuffer,
        0,
        this.bufferSizeCalc(seq_length, seq_length)
      );
      const softMaxResultBuffer = this.maskedInlineSoftmax(commandEncoder, seq_length, seq_length, softmaxInputBuffer);
      commandEncoder.copyBufferToBuffer(
        softMaxResultBuffer,
        0,
        softmaxOutputBuffer,
        i * this.bufferSizeCalc(seq_length, seq_length),
        this.bufferSizeCalc(seq_length, seq_length)
      );
    }

    const passEncoder_attentionValues = commandEncoder.beginComputePass();
    passEncoder_attentionValues.setPipeline(this.attentionValuesPipeline);
    passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
    passEncoder_attentionValues.setBindGroup(1, createBindGroup(this.device, this.r_r_BindLayout, [softmaxOutputBuffer, splitVResultBuffer]));
    passEncoder_attentionValues.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_attentionValues.end();

    const passEncoder_linear = commandEncoder.beginComputePass();
    passEncoder_linear.setPipeline(this.FFNpipeline);
    passEncoder_linear.setBindGroup(0, linearBindGroup);
    passEncoder_linear.setBindGroup(1, createBindGroup(this.device, this.r_BindLayout, [attentionValuesResultBuffer]));
    passEncoder_linear.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
    passEncoder_linear.end();

    return linearResultBuffer;
  }
}
