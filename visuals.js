class Visuals {

  initialized = false;

  constructor(model) {
    this.model = model;
    this.device = model.device;
    this.params = model.params;
  }

  init() {
    this.initFoundation();
    this.initUniforms();
    this.initLayoutAndPipeline();
    this.initBuffersAndBindGroup();
    this.updateModelBuffer();

    this.initialized = true;
  }

  initFoundation() {
    const containerEl = document.getElementById("visualsContainer");
    const gpuCanvasEl = document.createElement("canvas");

    containerEl.style.width = this.params.n_embd + "px";
    containerEl.style.height = this.params.n_ctx + "px";
    gpuCanvasEl.style.width = "100%";
    gpuCanvasEl.style.height = "100%";
    gpuCanvasEl.width = this.params.n_embd;
    gpuCanvasEl.height = this.params.n_ctx;

    const gpuContext = gpuCanvasEl.getContext("webgpu");
    const gpuCanvasFormat = navigator.gpu.getPreferredCanvasFormat();

    gpuContext.configure({
      device: this.device,
      format: gpuCanvasFormat,
    });

    containerEl.appendChild(gpuCanvasEl);

    this.containerEl = containerEl;
    this.gpuCanvasFormat = gpuCanvasFormat;
    this.gpuCanvasEl = gpuCanvasEl;
    this.gpuContext = gpuContext;
  }

  updateModelBuffer() {
    this.model.externalBuffer = this.embeddingsBuffer;
  }

  initUniforms() {
    this.uniforms = {
      width: this.model.params.n_embd,
      height: this.model.params.n_ctx,
    };
  }

  initLayoutAndPipeline() {
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform",
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          }
        },
      ]
    });

    this.renderShaderModule = this.device.createShaderModule({
      label: 'visuals',
      code: `
        struct UniformData {
          width: f32,
          height: f32,
        }

        @vertex
        fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
          var positions = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), // bottom left
            vec2<f32>( 1.0, -1.0), // bottom right
            vec2<f32>(-1.0,  1.0), // top left
            vec2<f32>(-1.0,  1.0), // top left
            vec2<f32>( 1.0, -1.0), // bottom right
            vec2<f32>( 1.0,  1.0)  // top right
          );
          return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        }

        @group(0) @binding(0) var<uniform> uniformData: UniformData;
        @group(0) @binding(1) var<storage, read> embeddingsBuffer: array<f32>;

        @fragment
        fn fsMain(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let xNormalized = fragCoord.x / uniformData.width;
          let yNormalized = fragCoord.y / uniformData.height;

          let xIndex = xNormalized * ${this.params.n_embd};
          let yIndex = yNormalized * ${this.params.n_ctx};
          let index = u32(yIndex) * ${this.params.n_embd} + u32(xIndex);

          let vectorValue = embeddingsBuffer[index];

          var outColor = vec4<f32>(0.0);
          outColor = hdrColorMapping(outColor, 1.0, vectorValue * 0.1);

          return outColor;
        }

        fn hdrColorMapping(colorRef: vec4<f32>, hdrThreshold: f32, vectorValue: f32) -> vec4<f32> {
          var color = colorRef;

          if (vectorValue < 0.0) {
            color.b = -vectorValue;
            if (vectorValue < -hdrThreshold) {
              color.g = -vectorValue - hdrThreshold;
            }
          } else {
            color.r = vectorValue;
            if (vectorValue > hdrThreshold) {
              color.g = vectorValue - hdrThreshold;
            }
          }
          color.g = min(color.g, 0.7);
          return color;
        }
      `
    });

    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.renderShaderModule,
        entryPoint: 'vsMain',
        buffers: []
      },
      fragment: {
        module: this.renderShaderModule,
        entryPoint: 'fsMain',
        targets: [
          {
            format: this.gpuCanvasFormat,
          },
        ],
      },
    });
  }

  initBuffersAndBindGroup() {
    const uniformCount = Object.values(this.uniforms).length;

    this.uniformBuffer = this.device.createBuffer({
      size: uniformCount * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.embeddingsBuffer = this.device.createBuffer({
      size: this.model.bufferSize(this.params.n_ctx, this.params.n_embd),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.embeddingsBuffer,
          }
        },
      ]
    });

    this.updateUniforms();
  }

  updateUniforms() {
    this.uniforms.width = this.gpuCanvasEl.width;
    this.uniforms.height = this.gpuCanvasEl.height;

    const uniformArray = new Float32Array([
      this.uniforms.width,
      this.uniforms.height,
    ]);

    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      uniformArray.buffer,
      uniformArray.byteOffset,
      uniformArray.byteLength,
    );
  }

  render(existingCommandEncoder) {
    const commandEncoder = existingCommandEncoder ?? this.device.createCommandEncoder();

    const textureView = this.gpuContext.getCurrentTexture().createView();

    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          loadOp: 'clear',
          loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: 'store',
        }
      ]
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  destroy() {
    this.gpuCanvasEl.remove();
    this.uniformBuffer.destroy();
    this.embeddingsBuffer.destroy();
  }
}