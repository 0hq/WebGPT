// ---------------- WebGPU Helper Functions ----------------

async function initializeWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU is not supported");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const queue = device.queue;

  const minStorageBufferOffsetAlignment = 1; // Should be device.limits.minStorageBufferOffsetAlignment but tis was breaking things. Fix later, not breaking just slows performance AFAIK.
  bufferSizeCalc = (dimA, dimB = 1) => alignedSize(dimA * dimB * Float32Array.BYTES_PER_ELEMENT, minStorageBufferOffsetAlignment);

  return { device, queue };
}

function createShader(device, code) {
  return device.createShaderModule({
    code,
  });
}

function createBindGroupLayout(device, string_entries) {
  const entries = string_entries.map((entry, i) => ({
    binding: i,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: entry },
  }));
  return device.createBindGroupLayout({
    entries,
  });
}

function createPipelineLayout(device, bindGroupLayouts) {
  return device.createPipelineLayout({
    bindGroupLayouts,
  });
}

function createComputePipeline(device, shaderModule, pipelineLayout) {
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });
}

function createPipeline(device, shaderString, bindGroupLayouts) {
  const shaderModule = createShader(device, shaderString);
  const pipelineLayout = createPipelineLayout(device, bindGroupLayouts);
  const pipeline = createComputePipeline(device, shaderModule, pipelineLayout);
  return pipeline;
}

function createBindGroup(device, bindGroupLayout, buffers) {
  const entries = buffers.map((buffer, i) => ({
    binding: i,
    resource: {
      buffer,
    },
  }));
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries,
  });
}

function createBuffer(device, size, usage) {
  return device.createBuffer({
    size: size,
    usage: usage,
  });
}

function createOutputBuffer(device, commandEncoder, buffer, rows, cols) {
  const outputBufferSize = bufferSizeCalc(rows, cols);
  const outputBuffer = createBuffer(device, outputBufferSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, outputBufferSize);
  return outputBuffer;
}

// ---------------- Other Helper Functions ----------------

function alignedSize(size, alignment) {
  return Math.ceil(size / alignment) * alignment;
}

const workgroupCalc = (dim, size) => Math.min(Math.ceil(dim / size), 256);

let bufferSizeCalc = (dimA, dimB = 1) => {
  throw new Error("BufferSizeCalc not initialized.");
};

function sampleFromDistribution(probs, top_k) {
  const sortedIndices = Array.from(probs)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .map(({ index }) => index);

  const topKIndices = sortedIndices.slice(0, top_k);
  const topKProbs = topKIndices.map((index) => probs[index]);

  const sumTopKProbs = topKProbs.reduce((a, b) => a + b, 0);
  const normalizedTopKProbs = topKProbs.map((prob) => prob / sumTopKProbs);

  const rand = Math.random();
  let cumulativeProb = 0;
  for (let i = 0; i < top_k; i++) {
    cumulativeProb += normalizedTopKProbs[i];
    if (rand < cumulativeProb) {
      return topKIndices[i];
    }
  }
  return topKIndices[top_k - 1];
}

function cpuSoftmax(logits, temperature = 1.0) {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((expLogit) => expLogit / sumExpLogits);
}

// ----------------------- Matrix Operations -----------------------

function transposeArray(array, input_rows, input_cols) {
  if (array.length !== input_rows * input_cols) {
    console.error("Transpose dims failed, not transposing!");
    // return array;
    throw new Error("Transpose dims failed");
  }

  const transpose = [];
  for (let col = 0; col < input_cols; col++) {
    for (let row = 0; row < input_rows; row++) {
      transpose.push(array[row * input_cols + col]);
    }
  }

  return new Float32Array(transpose);
}

function deEmbedCPU(embeddings, embeddingWeights, seq_length, n_embd, vocab_size) {
  // console.warn("I'm sorry for cheating... De-embedding output with CPU.");

  const predictionEmbeddings = new Float32Array(embeddings).slice((seq_length - 1) * n_embd);
  const logits = [];
  for (let i = 0; i < vocab_size; i++) {
    let dotProduct = 0;
    for (let j = 0; j < n_embd; j++) {
      dotProduct += embeddingWeights[i * n_embd + j] * predictionEmbeddings[j];
    }
    logits.push(dotProduct);
  }

  return logits;
}

function flattenEmbeddings(embeddings, n_embd, seq_length) {
  const flattened = new Float32Array(n_embd * seq_length);
  for (const [i, v] of embeddings.entries()) flattened.set(v, n_embd * i);
  return flattened;
}

function wasted_memory(vocab_size, n_embd, columns) {
  const subMatrixCount = math.ceil(vocab_size / columns);
  const totalElementsInSubMatrices = subMatrixCount * columns * n_embd;
  return totalElementsInSubMatrices - vocab_size * n_embd;
}

function leastPrimeFactor(n, start = 2) {
  for (let i = start; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return i;
  }
  return n;
}
