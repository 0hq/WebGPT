async function initializeWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU is not supported");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const queue = device.queue;

  return { device, queue };
}
