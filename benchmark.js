async function benchmark(device, pipeline, queue, N, it = 1) {
  const A = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));
  const B = new Array(N).fill(null).map(() => new Array(N).fill(null).map(() => Math.random()));

  const startTime = performance.now();

  const { bufferA, bufferB, bufferC, uniformBuffer, dim, masterDimA, masterDimB, bindGroup, numWorkgroupsX, numWorkgroupsY, bufferSizeC } =
    await preMatMulDiscrete(device, queue, pipeline, A, B);

  for (let i = 0; i < it; i++) {
    await runMatMulSameMatrix(device, queue, pipeline, bufferC, bindGroup, numWorkgroupsX, numWorkgroupsY, bufferSizeC);
    // await runMatMulDynamic(device, queue, pipeline, A, B);
  }

  const endTime = performance.now();

  const elapsedTime = endTime - startTime;
  const gflops = (2 * N * N * N * it) / (elapsedTime * 1e6);

  console.log(`Matrix Size: ${N}x${N}, Time: ${elapsedTime.toFixed(2)}ms, GFLOPS: ${gflops.toFixed(2)}`);
}

(async () => {
  const { device, queue } = await initializeWebGPU();
  const pipeline = await createMatMulPipeline(device);

  const N = 512;
  const it = 100;

  for (let i = 0; i < 20; i++) {
    await benchmark(device, pipeline, queue, N, it);
  }
})();
