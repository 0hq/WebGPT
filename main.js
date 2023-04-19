let validateIndex = 0;

let itos = null;
let stoi = null;
let modelParams = null;
let bufferSizeCalc = null;

let validateModel = null;
const validateFile = "generation copy.json";
const doValidation = false;

const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const textEncoder = new TextEncoder("utf-8");
const textDecoder = new TextDecoder("utf-8");
let tokenizerData = {
  encoder: null,
  decoder: null,
  bpe_ranks: null,
  byte_encoder: null,
  byte_decoder: null,
  cache: null,
};

(async () => {
  await loadTokenizer();

  // modelParams = await loadGPTModel("gpt2");
  // console.log("Params:", modelParams);
})();

async function loadTokenizer() {
  const bpe_file = await (await fetch("models/vocab.bpe")).text();
  tokenizerData.encoder = await (await fetch("models/gpt_string_to_int.json")).json();

  tokenizerData.decoder = {};
  Object.keys(tokenizerData.encoder).map((x) => {
    tokenizerData.decoder[tokenizerData.encoder[x]] = x;
  });

  const lines = bpe_file.split("\n");

  // bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
  const bpe_merges = lines.slice(1, lines.length - 1).map((x) => {
    return x.split(/(\s+)/).filter(function (e) {
      return e.trim().length > 0;
    });
  });

  tokenizerData.byte_encoder = bytes_to_unicode();
  tokenizerData.byte_decoder = {};
  Object.keys(tokenizerData.byte_encoder).map((x) => {
    tokenizerData.byte_decoder[tokenizerData.byte_encoder[x]] = x;
  });

  tokenizerData.bpe_ranks = dictZip(bpe_merges, range(0, bpe_merges.length));
  tokenizerData.cache = new Map();
}

(async () => {
  // modelParams = await loadFakeGPT2("state_dict");
  // console.log("Params:", modelParams);
  // const tokenDict = await (await fetch("models/tokens.json")).json();
  // itos = tokenDict.itos;
  // stoi = tokenDict.stoi;
  // console.log("Tokens:", tokenDict);
  // console.log("Unique Tokens:", new Set(Object.values(tokenDict.itos)));
  // console.log("Model finished loading.");
  // if (doValidation) {
  //   const validateData = await loadValidateModel(validateFile);
  //   validateModel = validateData;
  //   console.log("Validate model loaded", validateData);
  // }
  // generateFromModel("WILL:", 1, 1);
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

    console.log(`Output:\n\n${prompt.map((i) => itos[i]).join("")}`);
  }

  // console.log("Output ints:", prompt);
  // const text = prompt.map((i) => itos[i]).join("");
  // console.log(`Output:\n\n${text}`);
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
