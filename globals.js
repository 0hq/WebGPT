const FastMatMulBlock = new FastMatMulBlockClass();
const AttentionBlock = new AttentionBlockClass();
const ResidualBlock = new ResidualBlockClass();
const EmbedBlock = new EmbedBlockClass();
const DeEmbedBlock = new DeEmbedBlockClass();
const GeluBlock = new GeluBlockClass();
const LayerNormBlock = new LayerNormBlockClass();
const SoftmaxBlock = new SoftmaxBlockClass();

// Needed for deletion.
let operations = [FastMatMulBlock, AttentionBlock, ResidualBlock, EmbedBlock, DeEmbedBlock, GeluBlock, LayerNormBlock, SoftmaxBlock];

function initializeOperations(device) {
  for (const operation of operations) operation.initialize(device);
}

function destroyOperationBuffers() {
  for (const operation of operations) operation.destroyBuffers();
}

function clearOperationCache() {
  for (const operation of operations) operation.clearBufferCache();
}

function destroyOperations() {
  for (const operation of operations) operation.destroy();
}

const bufferUsageDict = {
  copy_from: GPUBufferUsage.COPY_SRC,
  copy_to: GPUBufferUsage.COPY_DST,
  storage: GPUBufferUsage.STORAGE,
  uniform: GPUBufferUsage.UNIFORM,
  map_read: GPUBufferUsage.MAP_READ,
};

// ---------------- Helper Functions ----------------

async function fetchBin(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

const wgSize = (dim, size) => Math.min(Math.ceil(dim / size), Infinity);

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

function selectTopK(probs, top_k) {
  const sortedIndices = Array.from(probs)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .map(({ index }) => index);
  const topKIndices = sortedIndices.slice(0, top_k);
  const topKProbs = topKIndices.map((index) => probs[index]);
  return { topKIndices, topKProbs };
}

// ----------------------- Matrix Operations -----------------------

const zeros = (dim) => new Float32Array(dim).fill(0);

function transpose(array, input_rows, input_cols) {
  if (array.length !== input_rows * input_cols) {
    console.log(array.length, input_rows, input_cols);
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

function leastPrimeFactor(n, start = 2) {
  for (let i = start; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return i;
  }
  return n;
}

function formatAsMatrix(floatArray, dimA, dimB) {
  const resultMatrix = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(floatArray.slice(i * dimB, (i + 1) * dimB));
  }
  return resultMatrix;
}
