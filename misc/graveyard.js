async function benchmarkMatrixMultiplication(device, pipeline) {
  const numTests = 100;
  const maxDimension = 1000;
  const rowsA = 1;
  const colsA = maxDimension;
  const rowsB = colsA;
  const colsB = 1;

  const A = generateRandomMatrix(rowsA, colsA);
  const B = generateRandomMatrix(rowsB, colsB);

  console.log("Running matrix multiplication benchmark...");
  console.log(`Matrix A: ${rowsA}x${colsA}`);
  console.log(`Matrix B: ${rowsB}x${colsB}`);

  const start = performance.now();
  for (let t = 0; t < numTests; t++) {
    const gpuResult = await runMatMul(device, pipeline, A, B);

    // console.log(`Run ${t + 1}: DONE`);
  }
  console.log("DONE");

  const end = performance.now();
  console.log(`Time taken: ${end - start} ms`);
  console.log(`Time per run: ${(end - start) / numTests} ms`);
  console.log(`Runs per second: ${numTests / ((end - start) / 1000)}`);
  console.log(`Ops per second: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000)}`);
  console.log(`GFLOPS: ${(maxDimension ** 2 * numTests) / ((end - start) / 1000) / 1e9}`);
}

async function runMLP(input, weight) {
  const device = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);

  // let layerOutput = input;
  // for (let i = 0; i < weights.length; i++) {
  //   const weight = weights[i];
  //   // const bias = biases[i];

  //   // Perform matrix multiplication
  //   layerOutput = await runMatMul(device, pipeline, layerOutput, weight);

  //   // Add biases
  //   // for (let j = 0; j < layerOutput.length; j++) {
  //   //   layerOutput[j] += bias[j % bias.length];
  //   // }

  //   // Apply the activation function
  //   // if (activation === "relu") {
  //   //   layerOutput = layerOutput.map((value) => Math.max(0, value));
  //   // } else if (activation === "sigmoid") {
  //   //   layerOutput = layerOutput.map((value) => 1 / (1 + Math.exp(-value)));
  //   // }
  // }

  return await runMatMul(device, pipeline, input, weight);
}

/*


  transformer.wte.weight: torch.Size([65, 128])
transformer.wpe.weight: torch.Size([64, 128])
transformer.h.0.ln_1.weight: torch.Size([128])
transformer.h.0.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.0.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.0.ln_2.weight: torch.Size([128])
transformer.h.0.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.0.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.1.ln_1.weight: torch.Size([128])
transformer.h.1.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.1.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.1.ln_2.weight: torch.Size([128])
transformer.h.1.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.1.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.2.ln_1.weight: torch.Size([128])
transformer.h.2.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.2.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.2.ln_2.weight: torch.Size([128])
transformer.h.2.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.2.mlp.c_proj.weight: torch.Size([128, 512])
transformer.h.3.ln_1.weight: torch.Size([128])
transformer.h.3.attn.c_attn.weight: torch.Size([384, 128])
transformer.h.3.attn.c_proj.weight: torch.Size([128, 128])
transformer.h.3.ln_2.weight: torch.Size([128])
transformer.h.3.mlp.c_fc.weight: torch.Size([512, 128])
transformer.h.3.mlp.c_proj.weight: torch.Size([128, 512])
transformer.ln_f.weight: torch.Size([128])
lm_head.weight: torch.Size([65, 128])

With bias:

transformer.wte.weight: torch.Size([65, 64])
transformer.wpe.weight: torch.Size([64, 64])

transformer.h.0.ln_1.weight: torch.Size([64])
transformer.h.0.ln_1.bias: torch.Size([64])
transformer.h.0.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.0.attn.c_attn.bias: torch.Size([192])
transformer.h.0.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.0.attn.c_proj.bias: torch.Size([64])
transformer.h.0.ln_2.weight: torch.Size([64])
transformer.h.0.ln_2.bias: torch.Size([64])
transformer.h.0.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.0.mlp.c_fc.bias: torch.Size([256])
transformer.h.0.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.0.mlp.c_proj.bias: torch.Size([64])

transformer.h.1.ln_1.weight: torch.Size([64])
transformer.h.1.ln_1.bias: torch.Size([64])
transformer.h.1.attn.c_attn.weight: torch.Size([192, 64])
transformer.h.1.attn.c_attn.bias: torch.Size([192])
transformer.h.1.attn.c_proj.weight: torch.Size([64, 64])
transformer.h.1.attn.c_proj.bias: torch.Size([64])
transformer.h.1.ln_2.weight: torch.Size([64])
transformer.h.1.ln_2.bias: torch.Size([64])
transformer.h.1.mlp.c_fc.weight: torch.Size([256, 64])
transformer.h.1.mlp.c_fc.bias: torch.Size([256])
transformer.h.1.mlp.c_proj.weight: torch.Size([64, 256])
transformer.h.1.mlp.c_proj.bias: torch.Size([64])

transformer.ln_f.weight: torch.Size([64])
transformer.ln_f.bias: torch.Size([64])
lm_head.weight: torch.Size([65, 64])

    */
