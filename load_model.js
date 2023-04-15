/*

{
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "block_size": 64,
    "bias": false,
    "vocab_size": 65,
    "dropout": 0
}

*/

async function loadModel(filename) {
  // Load the model from json file
  var model = await (await fetch(`models/${filename}.json`)).json();
  console.log(model);
}

// loadModel("bad_shakespeare");
