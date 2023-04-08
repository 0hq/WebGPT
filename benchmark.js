async function benchmark(device, pipeline, N, workgroupSize, it = 1) {
  const A = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));
  const B = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));

  const startTime = performance.now();

  for (let i = 0; i < it; i++) {
    await runMatMul(device, pipeline, A, B);
  }

  const endTime = performance.now();

  const elapsedTime = endTime - startTime;
  const gflops = (2 * N * N * N * it) / (elapsedTime * 1e6);

  console.log(`N: ${N}, Workgroup Size: ${workgroupSize}x${workgroupSize}, Time: ${elapsedTime.toFixed(2)}ms, GFLOPS: ${gflops.toFixed(2)}`);
}

(async () => {
  const device = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);

  const N = 1024;
  const workgroupSize = 16; // Match the workgroup size defined in createMatMulShader
  const it = 1;

  for (let i = 0; i < 5; i++) {
    await benchmark(device, pipeline, N, workgroupSize, it);
  }
})();
