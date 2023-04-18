let validateIndex = 0;

let rawModel = null;

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation copy.json";
const doValidation = false;

(async () => {
  modelParams = await loadModel("state_dict");
  console.log("Params:", modelParams);

  const tokenDict = await (await fetch("models/tokens.json")).json();

  itos = tokenDict.itos;
  stoi = tokenDict.stoi;

  console.log("Tokens:", tokenDict);
  console.log("Unique Tokens:", new Set(Object.values(tokenDict.itos)));

  console.log("Model finished loading.");

  if (doValidation) {
    const validateData = await loadValidateModel(validateFile);

    validateModel = validateData;

    console.log("Validate model loaded", validateData);
  }

  generateFromModel("BRUTUS:", 200, 1);

  // validateAgainstModel();
})();

async function generateFromModel(prompt, max_new_tokens, top_k = 1) {
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
    validateIndex = i;

    const idx_cond = prompt.slice(-context_size);
    // console.log("idx_cond", idx_cond);
    const logits = await runInference(idx_cond);

    // console.log("Logits", logits);

    const probs = cpuSoftmax(logits, 1.0);

    // console.log("Probs", probs);

    const idx_next = sampleFromDistribution(probs, top_k);
    console.log("Next token", idx_next);
    prompt = prompt.concat(idx_next);
  }

  // console.log("Output ints:", prompt);
  const text = prompt.map((i) => itos[i]).join("");
  console.log(`Output:\n\n${text}`);
}

async function validateAgainstModel() {
  if (!modelParams || !stoi || !itos) {
    console.log("Model not loaded yet");
    return;
  }

  const context_size = modelParams.params.context_size;

  console.log(`Starting validation against ${validateFile}`);
  console.log("Validate model loaded", validateModel);
  console.log("Model params", modelParams);
  console.log("Context size", context_size);

  for (let i = 0; i < validateModel.length; i++) {
    const step = validateModel[i];

    validateIndex = i;

    const idx_cond = step.idx.data[0].slice(-context_size);
    console.log("idx_cond", idx_cond);
    const logits = await runInference(idx_cond);

    // console.log("Logits", logits);
    // console.log("Expected logits", step.logits);

    const probs = cpuSoftmax(logits, 1.0);

    // console.log("Probs", probs);
    // console.log("Expected probs", step.probs);

    const idx_next = sampleFromDistribution(probs, 1);
    console.log("Next token", idx_next);
    console.log("Expected token", sampleFromDistribution(step.probs.data[0], 1));
    if (idx_next !== sampleFromDistribution(step.probs.data[0], 1)) {
      throw new Error("Validation failed");
    }
  }
}

// {
//   "0": 35,
//   "1": 46,
//   "2": 39,
//   "3": 58,
//   "4": 1,
//   "5": 47,
//   "6": 57,
//   "7": 1,
//   "8": 58,
//   "9": 46,
//   "10": 43,
//   "11": 1,
//   "12": 39,
//   "13": 52,
//   "14": 57,
//   "15": 61,
//   "16": 43,
//   "17": 56,
//   "18": 1,
//   "19": 58,
//   "20": 53,
//   "21": 1,
//   "22": 50,
//   "23": 47,
//   "24": 44,
//   "25": 43,
//   "26": 6,
//   "27": 1,
//   "28": 58,
//   "29": 46,
//   "30": 43,
//   "31": 1,
//   "32": 59,
//   "33": 52,
//   "34": 47,
//   "35": 60,
//   "36": 43,
//   "37": 56,
//   "38": 57,
//   "39": 43,
//   "40": 6,
//   "41": 1,
//   "42": 39,
//   "43": 52,
//   "44": 42,
//   "45": 1,
//   "46": 43,
//   "47": 60,
//   "48": 43,
//   "49": 56,
//   "50": 63,
//   "51": 58,
//   "52": 46,
//   "53": 47,
//   "54": 52,
//   "55": 45,
//   "56": 12
// }
