const NaiveMatMulBlock = new NaiveMatMulBlockClass();
const FastMatMulBlock = new FastMatMulBlockClass();
const FastRowAddBlock = new FastRowAddBlockClass();
const FastFFNBlock = new FastFFNBlockClass();
const AttentionBlock = new AttentionBlockClass();
const ResidualBlock = new ResidualBlockClass();
const EmbedBlock = new EmbedBlockClass();
const DeEmbedBlock = new DeEmbedBlockClass();
const OldDeEmbedBlock = new OldDeEmbedBlockClass();
const GeluBlock = new GeluBlockClass();
const LayerNormBlock = new LayerNormBlockClass();
const TransposeBlock = new TransposeBlockClass();
const SoftmaxBlock = new SoftmaxBlockClass();

// Needed for deletion.
const operations = [
  NaiveMatMulBlock,
  FastMatMulBlock,
  FastRowAddBlock,
  FastFFNBlock,
  AttentionBlock,
  ResidualBlock,
  EmbedBlock,
  DeEmbedBlock,
  OldDeEmbedBlock,
  GeluBlock,
  LayerNormBlock,
  TransposeBlock,
  SoftmaxBlock,
];

function initializeOperations(device) {
  for (const operation of operations) operation.initialize(device);
}

function destroyOperationBuffers() {
  for (const operation of operations) operation.destroyBuffers();
}
