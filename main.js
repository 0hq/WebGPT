let tokenIndex = 0;

let rawModel = null;

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation.json";
const doValidation = true;

(async () => {
  modelParams = await loadModel("state_dict");
  console.log("Params:", modelParams);

  const tokenDict = await (await fetch("models/tokens.json")).json();

  itos = tokenDict.itos;
  stoi = tokenDict.stoi;

  console.log("Tokens:", tokenDict);
  console.log("Unique Tokens:", new Set(Object.values(tokenDict.itos)));

  console.log("Model finished loading.");

  const validateData = await loadValidateModel(validateFile);

  validateModel = validateData;

  console.log("Validate model loaded", validateData);

  generateFromModel("What is the answer to life, the universe, and everything?", 1);

  // cpuTest();
  // cpuTest2();
})();

async function generateFromModel(prompt, max_new_tokens) {
  if (!modelParams || !stoi || !itos) {
    console.log("Model not loaded yet");
    return;
  }

  console.log("Starting generation with prompt", prompt);
  prompt = prompt.split("").map((c) => stoi[c]);
  console.log("Parsed prompt", prompt);

  const context_size = modelParams.params.context_size;
  console.log("block_size", context_size);
  for (let i = 0; i < max_new_tokens; i++) {
    tokenIndex = i;

    // console.log("prompt", prompt);
    const idx_cond = prompt.slice(-context_size);
    // console.log("running inference on sequence", idx_cond);
    const logits = await runInference(idx_cond);

    console.log("Logits", logits);

    // pluck the logits at the final step and scale by desired temperature
    // const logits_scaled = logits; // / temperature;

    // apply softmax to convert logits to (normalized) probabilities
    const probs = simpleSoftmax(logits);

    console.log("Probs", probs);
    console.log("Max prob:", Math.max(...probs));
    console.log("Max prob index:", probs.indexOf(Math.max(...probs)), "char:", itos[probs.indexOf(Math.max(...probs))]);

    // sample from the distribution
    const idx_next = sampleFromDistribution(probs);
    // append sampled index to the running sequence and continue
    // console.log("generated", idx_next);
    prompt = prompt.concat(idx_next);
  }

  console.log("Output ints:", prompt);
  const text = prompt.map((i) => itos[i]).join("");
  console.log("Output:", text);
}

function cpuTest() {
  const inputEmbeddings = validateModel[tokenIndex].tok_pos_emb.data[0];

  const biasEnabled = modelParams.params.biasEnabled;

  const prefix = `transformer.h.${0}.`;

  const layerNormAttentionGamma = rawModel[`${prefix}ln_1.weight`].values.flat().map(parseFloat);
  const layerNormAttentionBeta = biasEnabled ? rawModel[`${prefix}ln_1.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

  console.log(inputEmbeddings, layerNormAttentionBeta, layerNormAttentionGamma);

  // Calculate stats

  // inputEmbeddings dim is 57 x 128

  const average = (array) => array.reduce((a, b) => a + b) / array.length;
  const averageInputEmbeddings = inputEmbeddings.map((row) => average(row));

  console.log("avg", averageInputEmbeddings);

  const variance = (array) => {
    const avg = average(array);
    return average(array.map((a) => (a - avg) ** 2)) + 1e-5;
  };
  const varianceInputEmbeddings = inputEmbeddings.map((row) => variance(row));

  console.log("var", varianceInputEmbeddings);

  const stdevInputEmbeddings = varianceInputEmbeddings.map(Math.sqrt);

  console.log("stdev", stdevInputEmbeddings);

  // let mean = Stats.data[row * 2];
  // let stdev = Stats.data[row * 2 + 1];
  // let output = (Input.data[row * dimX + col] - mean) / stdev;
  // let gamma = Gamma.data[row * 2];
  // let beta = Beta.data[row * 2];
  // let shift = gamma * output + beta;
  // Result.data[row * dimX + col] = shift;

  const output = inputEmbeddings.map((row, rowIdx) => {
    return row.map((col, colIdx) => {
      const mean = averageInputEmbeddings[rowIdx];
      const stdev = stdevInputEmbeddings[rowIdx];
      const gamma = layerNormAttentionGamma[colIdx];
      const beta = layerNormAttentionBeta[colIdx];
      return ((col - mean) / stdev) * gamma + beta;
    });
  });

  console.log("output", output);

  const expectedDict = validateModel[tokenIndex][`block${0}_ln1`];
  const expected = expectedDict.data[0];
  console.log("expected output", expected);
}

function cpuTest2() {
  const inputEmbeddings = validateModel[0].tok_pos_emb.data[0];

  const biasEnabled = modelParams.params.biasEnabled;
  const n_embd = modelParams.params.n_embd;

  const prefix = `transformer.h.${0}.`;

  const layerNormAttentionGamma = rawModel[`${prefix}ln_1.weight`].values.flat().map(parseFloat);
  const layerNormAttentionBeta = biasEnabled ? rawModel[`${prefix}ln_1.bias`].values.flat().map(parseFloat) : new Array(n_embd).fill(0);

  // const layerNormAttentionGamma = new Array(n_embd).fill(1);
  // const layerNormAttentionBeta = new Array(n_embd).fill(0);

  console.log(inputEmbeddings, layerNormAttentionBeta, layerNormAttentionGamma);

  const average = (array) => array.reduce((a, b) => a + b) / array.length;

  // inputEmbeddings dim is 57 x 128
  // Calculate column averages of of 1 x 128
  const averageInputEmbeddings = inputEmbeddings[0].map((_, colIdx) => average(inputEmbeddings.map((row) => row[colIdx])));

  // console.log("avg", averageInputEmbeddings);

  // Calculate column variances of of 1 x 128
  const variance = (array) => {
    const avg = average(array);
    return average(array.map((a) => (a - avg) ** 2)) + 1e-5;
  };
  const varianceInputEmbeddings = inputEmbeddings[0].map((_, colIdx) => variance(inputEmbeddings.map((row) => row[colIdx])));

  // console.log("var", varianceInputEmbeddings);

  // Calculate column stdevs of of 1 x 128
  const stdevInputEmbeddings = varianceInputEmbeddings.map(Math.sqrt);

  // console.log("stdev", stdevInputEmbeddings);

  // Calculate output of 57 x 128
  const output = inputEmbeddings.map((row, rowIdx) => {
    return row.map((col, colIdx) => {
      const mean = averageInputEmbeddings[colIdx];
      const stdev = stdevInputEmbeddings[colIdx];
      const gamma = layerNormAttentionGamma[colIdx];
      const beta = layerNormAttentionBeta[colIdx];
      return ((col - mean) / stdev) * gamma + beta;
    });
  });
  console.log("output", output);

  // Verify that every column is gaussian
  // verifyLayerNorm(output);

  const averageOutput = output[0].map((_, colIdx) => average(output.map((row) => row[colIdx])));
  console.log(averageOutput);

  const stdevOutput = output[0].map((_, colIdx) => variance(output.map((row) => row[colIdx]))).map(Math.sqrt);
  console.log(stdevOutput);

  const expectedDict = validateModel[tokenIndex][`block${0}_ln1`];
  const expected = expectedDict.data[0];
  console.log("expected output", expected);

  const averageExpectedOutput = expected[0].map((_, colIdx) => average(expected.map((row) => row[colIdx])));
  console.log(averageExpectedOutput);

  const stdevExpectedOutput = expected[0].map((_, colIdx) => variance(expected.map((row) => row[colIdx]))).map(Math.sqrt);
  console.log(stdevExpectedOutput);

  const averageExpectedOutputRow = expected.map((row) => average(row));
  console.log(averageExpectedOutputRow);

  const stdevExpectedOutputRow = expected.map((row) => variance(row)).map(Math.sqrt);
  console.log(stdevExpectedOutputRow);
}

function verifyLayerNorm(output, epsilon = 1e-2) {
  const columnMean = (array, colIdx) => array.reduce((a, b) => a + b[colIdx], 0) / array.length;

  const columnStdev = (array, colIdx) => {
    const mean = columnMean(array, colIdx);
    const variance = array.reduce((a, b) => a + (b[colIdx] - mean) ** 2, 0) / array.length;
    return Math.sqrt(variance + 1e-5);
  };

  const numColumns = output[0].length;
  for (let colIdx = 0; colIdx < numColumns; colIdx++) {
    const mean = columnMean(output, colIdx);
    const stdev = columnStdev(output, colIdx);
    if (Math.abs(mean) > epsilon || Math.abs(stdev - 1) > epsilon) {
      console.log(`Column ${colIdx} does not meet the criteria: mean = ${mean}, stdev = ${stdev}`);
      return false;
    }
  }
  console.log("All columns meet the criteria");
  return true;
}
