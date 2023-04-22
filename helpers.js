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

  r_r_r_BindLayout = createBindGroupLayout(device, ["read-only-storage", "read-only-storage", "read-only-storage"]);
  r_r_BindLayout = createBindGroupLayout(device, ["read-only-storage", "read-only-storage"]);
  r_BindLayout = createBindGroupLayout(device, ["read-only-storage"]);
  u_r_r_s_BindLayout = createBindGroupLayout(device, ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  u_s_BindLayout = createBindGroupLayout(device, ["uniform", "storage"]);
  u_s_s_s_BindLayout = createBindGroupLayout(device, ["uniform", "storage", "storage", "storage"]);

  statsPipeline = createPipeline(device, normStatsShader, [u_s_BindLayout, r_BindLayout]);
  normPipeline = createPipeline(device, normShader, [u_s_BindLayout, r_r_r_BindLayout, r_BindLayout]);
  FFNpipeline = createPipeline(device, FFNShader, [u_r_r_s_BindLayout, r_BindLayout]);
  GELUpipeline = createPipeline(device, GELUShader, [u_s_BindLayout, r_BindLayout]);
  splitQKVpipeline = createPipeline(device, splitQKVShader, [u_s_s_s_BindLayout, r_BindLayout]);
  attentionWeightsPipeline = createPipeline(device, attentionWeightsShader, [u_s_BindLayout, r_r_BindLayout]);
  attentionValuesPipeline = createPipeline(device, attentionValuesShader, [u_s_BindLayout, r_r_BindLayout]);
  multiplyPipeline = createPipeline(device, multiplyShader, [u_s_BindLayout, r_BindLayout]);
  causalMaskPipeline = createPipeline(device, causalMaskShader, [u_s_BindLayout, r_BindLayout]);
  matmulPipeline = createPipeline(device, matMulShader, [u_s_BindLayout, r_r_BindLayout]);
  elementAddPipeline = createPipeline(device, elementWiseAdditionShader, [u_s_BindLayout, r_BindLayout, r_BindLayout]);
  maxPipeline = createPipeline(device, negMaxShader, [u_s_BindLayout, r_BindLayout]);
  addPipeline = createPipeline(device, addShader, [u_s_BindLayout, r_BindLayout, r_BindLayout]);
  expPipeline = createPipeline(device, expShader, [u_s_BindLayout, r_BindLayout]);
  sumPipeline = createPipeline(device, sumShader, [u_s_BindLayout, r_BindLayout]);
  dividePipeline = createPipeline(device, divideShader, [u_s_BindLayout, r_BindLayout, r_BindLayout]);

  return { device, queue };
}

let statsPipeline;
let normPipeline;
let FFNpipeline;
let GELUpipeline;
let splitQKVpipeline;
let attentionWeightsPipeline;
let attentionValuesPipeline;
let multiplyPipeline;
let causalMaskPipeline;
let matmulPipeline;
let elementAddPipeline;
let maxPipeline;
let addPipeline;
let expPipeline;
let sumPipeline;
let dividePipeline;

let r_r_r_BindLayout;
let r_r_BindLayout;
let r_BindLayout;
let u_r_r_s_BindLayout;
let u_s_BindLayout;
let u_s_s_s_BindLayout;

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

function sampleFromDistribution(probs) {
  const rand = Math.random();
  let cumulativeProb = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulativeProb += probs[i];
    if (rand < cumulativeProb) {
      return i;
    }
  }
  return probs.length - 1;
}

function cpuSoftmax(logits, temperature = 1.0) {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((expLogit) => expLogit / sumExpLogits);
}

// taken from https://github.com/mourner/quickselect/blob/26a241497b101167ab43c27c077bed4729c6b697/index.js
function quickselectStep(arr, k, left, right, compare) {
  while (right > left) {
    if (right - left > 600) {
      const n = right - left + 1;
      const m = k - left + 1;
      const z = Math.log(n);
      const s = 0.5 * Math.exp(2 * z / 3);
      const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * (m - n / 2 < 0 ? -1 : 1);
      const newLeft = Math.max(left, Math.floor(k - m * s / n + sd));
      const newRight = Math.min(right, Math.floor(k + (n - m) * s / n + sd));
      quickselectStep(arr, k, newLeft, newRight, compare);
    }

    const t = arr[k];
    let i = left;
    let j = right;

    swap(arr, left, k);
    if (compare(arr[right], t) > 0) swap(arr, left, right);

    while (i < j) {
      swap(arr, i, j);
      i++;
      j--;
      while (compare(arr[i], t) < 0) i++;
      while (compare(arr[j], t) > 0) j--;
    }

    if (compare(arr[left], t) === 0) swap(arr, left, j);
    else {
      j++;
      swap(arr, j, right);
    }

    if (j <= k) left = j + 1;
    if (k <= j) right = j - 1;
  }
}

function swap(arr, i, j) {
  const tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}

function selectTopK(probs, top_k) {
  const sortedIndices = Array.from(probs).map((value, index) => ({ value, index }))
  quickselectStep(sortedIndices, top_k, 0, probs.length - 1, (a, b) => b.value - a.value)

  const topKIndices = sortedIndices.slice(0, top_k).map(({ index }) => index);
  const topKProbs = topKIndices.map((index) => probs[index]);
  return { topKIndices, topKProbs };
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

function leastPrimeFactor(n, start = 2) {
  for (let i = start; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return i;
  }
  return n;
}
