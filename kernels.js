// --------------------- SHADER CODE --------------------- //

// Return maximum value of each row in a matrix times -1.
const negMaxShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    var max_buffer: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
      max_buffer = max(max_buffer, Input.data[row * dimX + i]);
    }

    Result.data[row] = -max_buffer;
  }
`;

// Adds constants [rows, 1] to each row of a matrix [rows, cols].
const addShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(2) @binding(0) var<storage, read> Constants: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] + Constants.data[row];
    }
`;

// Exponentiates each element of a matrix.
const expShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = exp(Input.data[row * dimX + col]);
    }
`;

// Returns the sum of each row of a matrix.
const sumShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        sum = sum + Input.data[row * dimX + i];
    }

    Result.data[row] = sum;
  }
`;

// Divides each element of a matrix by a constant [rows, 1].
const divideShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;
  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(2) @binding(0) var<storage, read> Divisors: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] / Divisors.data[row];
    }
`;

// Multiplies matrix times weights and adds bias.
const FFNShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of A and row dimension of C
    dimX: u32, // col dimension of B and col dimension of C
    dimS: u32, // shared dimension of A and B
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read> Bias: Matrix;
  @group(0) @binding(2) var<storage, read> Weight: Matrix;
  @group(0) @binding(3) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;
    let dimS: u32 = DimBuffer.dimS;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimS; i = i + 1) {
        sum = sum + Input.data[row * dimS + i] * Weight.data[i * dimX + col];
    }

    Result.data[row * dimX + col] = sum + Bias.data[col];
  }
`;

// Masks all values in the matrix that are not causal.
// Currently also transposes the matrix for copying.
const causalMaskShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      let rowMask: u32 = row % dimX;
      let rowNum: u32 = row / dimX;
      if (col > rowMask) {
        Result.data[row * dimX + col] = -1e9;
      } else {
        Result.data[row * dimX + col] = Input.data[rowMask * dimY + col + rowNum * dimX];
      }

    }
`;

// Transpose the matrix.
const transposeShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Result.data[row * dimX + col] = Input.data[col * dimY + row];
  }
`;

// Splits a matrix into Q, K, and V matrices.
const splitQKVShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of Q, K, V
    dimX: u32, // col dimension of Q, K, V
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Q: Matrix;
  @group(0) @binding(2) var<storage, read_write> K: Matrix;
  @group(0) @binding(3) var<storage, read_write> V: Matrix;


  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Q.data[row * dimX + col] = Input.data[row * dimX * 3 + col];
    K.data[row * dimX + col] = Input.data[row * dimX * 3 + dimX + col];
    V.data[row * dimX + col] = Input.data[row * dimX * 3 + 2 * dimX + col];

  }
`;

// Calculates attention weights from Q and K matrices.
const attentionWeightsShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // output row and col dimension, Q & K row dimension (context)
    dimX: u32, // context * heads
    qkvCols: u32, // col dim of Q, K heads
    embedDim: u32, // embedding dimension
  };

  @group(1) @binding(0) var<storage, read> Queries: Matrix;
  @group(1) @binding(1) var<storage, read> Keys: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let qkvCols: u32 = DimBuffer.qkvCols;
    let embedDim: u32 = DimBuffer.embedDim;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / dimY;
    var col_r: u32 = col % dimY;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < qkvCols; i = i + 1) {
        sum = sum + Queries.data[row * embedDim + i + head * qkvCols] * Keys.data[col_r * embedDim + i + head * qkvCols];
    }

    Result.data[row * dimX + col] = sum;
  }
`;

// Calculates attention values from attention weights and V matrix.
const attentionValuesShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // Values row and col dimension, Weights row dimension (context)
    dimX: u32, // Result col dim (n_embd)
    numHeads: u32, // number of heads
    vCols: u32, // col dim of V
  };

  @group(1) @binding(0) var<storage, read> Weights: Matrix;
  @group(1) @binding(1) var<storage, read> Values: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimY: u32 = DimBuffer.dimY;
    let dimX: u32 = DimBuffer.dimX;
    let vCols: u32 = DimBuffer.vCols;

    if (row >= dimY || col >= dimX) {
      return;
    }

    var head: u32 = col / vCols;
    var col_r: u32 = col % dimY;
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimY; i = i + 1) {
        sum = sum +  Values.data[i * dimX + col] * Weights.data[row * dimY + i + head * dimY * dimY];
    }

    Result.data[row * dimX + col] = sum;
  }
`;

// Multiplies every value in a matrix by a single constant.
const multiplyShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
    attentionScale: f32,
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;

      if (row >= DimBuffer.dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = Input.data[row * dimX + col] * DimBuffer.attentionScale;
    }
`;

// Adds two matrices element-wise.
// Obviously super inefficient but i'll be optimizing later, just trying to get this working for now.
const elementWiseAdditionShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Uniforms {
    dimY: u32,
    dimX: u32,
  };

  @group(2) @binding(0) var<storage, read> LayerOutput: Matrix;
  @group(1) @binding(0) var<storage, read> Residual: Matrix;

  @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = dimBuffer.dimX;
    let dimY: u32 = dimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    Result.data[row * dimX + col] = LayerOutput.data[row * dimX + col] + Residual.data[row * dimX + col];
  }
`;

// Multiplies two matrices.
const matMulShader = `
    struct Matrix {
        data: array<f32>,
    }

    struct Uniforms {
      dimY: u32, // row dimension of A and row dimension of C
      dimX: u32, // col dimension of B and col dimension of C
      dimS: u32, // shared dimension of A and B
    };

    @group(1) @binding(0) var<storage, read> A: Matrix;
    @group(1) @binding(1) var<storage, read> B: Matrix;

    @group(0) @binding(1) var<storage, read_write> C: Matrix;
    @group(0) @binding(0) var<uniform> dimBuffer: Uniforms;

    @compute @workgroup_size(16, 16)
    fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row: u32 = global_id.x;
        let col: u32 = global_id.y;
        let dimX: u32 = dimBuffer.dimX;
        let dimY: u32 = dimBuffer.dimY;
        let dimS: u32 = dimBuffer.dimS;

        if (row >= dimY || col >= dimX) {
          return;
        }

        var sum: f32 = 0.0;
        for (var i: u32 = 0; i < dimS; i = i + 1) {
            sum = sum + A.data[row * dimS + i] * B.data[i * dimX + col];
        }

        C.data[row * dimX + col] = sum;
      }
  `;

// Calculates mean and standard deviation per row of a matrix.
const normStatsShader = `
  struct Matrix {
    data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension
    dimX: u32, // col dimension
  };

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;

    if (row >= DimBuffer.dimY || col >= 1) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        sum = sum + Input.data[row * dimX + i];
    }
    var mean: f32 = sum / f32(dimX);

    var variance: f32 = 0.0;
    for (var i: u32 = 0; i < dimX; i = i + 1) {
        variance = variance + (Input.data[row * dimX + i] - mean) * (Input.data[row * dimX + i] - mean);
    }
    variance = variance / f32(dimX);
    var stdev: f32 = sqrt(variance + 1e-5);

    Result.data[row * 2] = mean;
    Result.data[row * 2 + 1] = stdev;
  }
`;

// Adjusts the input matrix by the mean and standard deviation and gamma and beta parameters.
const normShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;
  @group(1) @binding(1) var<storage, read> Gamma: Matrix;
  @group(1) @binding(2) var<storage, read> Beta: Matrix;
  @group(2) @binding(0) var<storage, read> Stats: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row: u32 = global_id.x;
    let col: u32 = global_id.y;
    let dimX: u32 = DimBuffer.dimX;
    let dimY: u32 = DimBuffer.dimY;

    if (row >= dimY || col >= dimX) {
      return;
    }

    let mean = Stats.data[row * 2];
    let stdev = Stats.data[row * 2 + 1];
    let output = (Input.data[row * dimX + col] - mean) / stdev;
    let gamma = Gamma.data[col];
    let beta = Beta.data[col];
    let shift = gamma * output + beta;
    Result.data[row * dimX + col] = shift;
  }
`;

// Squashes all elements of a matrix using the GELU function.
// There's tons of obvious ineffiencies here but I'm pushing them to after this is working.
const GELUShader = `
  struct Matrix {
      data: array<f32>,
  }

  struct Dimensions {
    dimY: u32, // row dimension of input matrix
    dimX: u32, // col dimension of input matrix
  };

  const SQRPI: f32 = 0.7978845608;
  fn gelu(x: f32) -> f32 {
    if (x < -10.0) {
      return 0.0;
    } else if (x > 10.0) {
      return x;
    } else {
      let cdf_approx: f32 = 0.5 * (1.0 + tanh(SQRPI * (x + 0.044715 * pow(x, 3))));
      return x * cdf_approx;
    }
  }

  @group(0) @binding(0) var<uniform> DimBuffer: Dimensions;
  @group(0) @binding(1) var<storage, read_write> Result: Matrix;

  @group(1) @binding(0) var<storage, read> Input: Matrix;

  @compute @workgroup_size(16, 16)
  fn main (@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row: u32 = global_id.x;
      let col: u32 = global_id.y;
      let dimX: u32 = DimBuffer.dimX;
      let dimY: u32 = DimBuffer.dimY;

      if (row >= dimY || col >= dimX) {
        return;
      }

      Result.data[row * dimX + col] = gelu(Input.data[row * dimX + col]);
    }
  `;

// TODO: Optimize workgroup size and set globals per shader.
const workgroup_X = 16; // Dictated by shader.
const workgroup_Y = 16; // Dictated by shader.

// --------------------- PIPELINES --------------------- //

function inlineSoftmax(device, queue, commandEncoder, rows, cols, inputBuffer) {
  const dimUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  queue.writeBuffer(dimUniformBuffer, 0, new Uint32Array([rows, cols]));

  const maxResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const maxBindGroup = createBindGroup(device, u_s_BindLayout, [dimUniformBuffer, maxResultBuffer]);

  const addResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const addBindGroup = createBindGroup(device, u_s_BindLayout, [dimUniformBuffer, addResultBuffer]);

  const expResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const expBindGroup = createBindGroup(device, u_s_BindLayout, [dimUniformBuffer, expResultBuffer]);

  const sumResultBuffer = createBuffer(device, bufferSizeCalc(rows), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const sumBindGroup = createBindGroup(device, u_s_BindLayout, [dimUniformBuffer, sumResultBuffer]);

  const divResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const divBindGroup = createBindGroup(device, u_s_BindLayout, [dimUniformBuffer, divResultBuffer]);

  const passEncoder_max = commandEncoder.beginComputePass();
  passEncoder_max.setPipeline(maxPipeline);
  passEncoder_max.setBindGroup(0, maxBindGroup);
  passEncoder_max.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder_max.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_max.end();

  const passEncoder_add = commandEncoder.beginComputePass();
  passEncoder_add.setPipeline(addPipeline);
  passEncoder_add.setBindGroup(0, addBindGroup);
  passEncoder_add.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder_add.setBindGroup(2, createBindGroup(device, r_BindLayout, [maxResultBuffer]));
  passEncoder_add.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_add.end();

  const passEncoder_exp = commandEncoder.beginComputePass();
  passEncoder_exp.setPipeline(expPipeline);
  passEncoder_exp.setBindGroup(0, expBindGroup);
  passEncoder_exp.setBindGroup(1, createBindGroup(device, r_BindLayout, [addResultBuffer]));
  passEncoder_exp.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_exp.end();

  const passEncoder_sum = commandEncoder.beginComputePass();
  passEncoder_sum.setPipeline(sumPipeline);
  passEncoder_sum.setBindGroup(0, sumBindGroup);
  passEncoder_sum.setBindGroup(1, createBindGroup(device, r_BindLayout, [expResultBuffer]));
  passEncoder_sum.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_sum.end();

  const passEncoder_div = commandEncoder.beginComputePass();
  passEncoder_div.setPipeline(dividePipeline);
  passEncoder_div.setBindGroup(0, divBindGroup);
  passEncoder_div.setBindGroup(1, createBindGroup(device, r_BindLayout, [expResultBuffer]));
  passEncoder_div.setBindGroup(2, createBindGroup(device, r_BindLayout, [sumResultBuffer]));
  passEncoder_div.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder_div.end();

  return divResultBuffer;
}

function inlineResidual(device, queue, commandEncoder, rows, cols, layerOutputBuffer, residualBuffer) {
  const residualUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const residualResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const residualBindGroup = createBindGroup(device, u_s_BindLayout, [residualUniformBuffer, residualResultBuffer]);
  queue.writeBuffer(residualUniformBuffer, 0, new Uint32Array([rows, cols]));

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(elementAddPipeline);
  passEncoder.setBindGroup(0, residualBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, r_BindLayout, [residualBuffer]));
  passEncoder.setBindGroup(2, createBindGroup(device, r_BindLayout, [layerOutputBuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  return residualResultBuffer;
}

function inlineMatMul(device, queue, commandEncoder, Abuffer, Bbuffer, rows, cols, shared) {
  const matmulUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const matmulResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const matMulBindGroup = createBindGroup(device, u_s_BindLayout, [matmulUniformBuffer, matmulResultBuffer]);
  queue.writeBuffer(matmulUniformBuffer, 0, new Uint32Array([rows, cols, shared]));

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(matmulPipeline);
  passEncoder.setBindGroup(0, matMulBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, r_r_BindLayout, [Abuffer, Bbuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  return matmulResultBuffer;
}

function inlineTranspose(device, queue, commandEncoder, inputBuffer, rows, cols) {
  const transposePipeline = createPipeline(device, transposeShader, [u_s_BindLayout, r_BindLayout]);

  const transposeUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const transposeResultBuffer = createBuffer(device, bufferSizeCalc(rows, cols), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const transposeBindGroup = createBindGroup(device, u_s_BindLayout, [transposeUniformBuffer, transposeResultBuffer]);
  queue.writeBuffer(transposeUniformBuffer, 0, new Uint32Array([rows, cols]));

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(transposePipeline);
  passEncoder.setBindGroup(0, transposeBindGroup);
  passEncoder.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder.dispatchWorkgroups(workgroupCalc(rows, workgroup_Y), workgroupCalc(cols, workgroup_X));
  passEncoder.end();

  return transposeResultBuffer;
}

function inlineLayerNorm(device, queue, commandEncoder, seq_length, n_embd, inputBuffer, gammaBuffer, betaBuffer) {
  const statsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const statsResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, 2), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const statsBindGroup = createBindGroup(device, u_s_BindLayout, [statsUniformBuffer, statsResultBuffer]);
  queue.writeBuffer(statsUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const normUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const normResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const normBindGroup = createBindGroup(device, u_s_BindLayout, [normUniformBuffer, normResultBuffer]);
  queue.writeBuffer(normUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const passEncoder_stats = commandEncoder.beginComputePass();
  passEncoder_stats.setPipeline(statsPipeline);
  passEncoder_stats.setBindGroup(0, statsBindGroup);
  passEncoder_stats.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder_stats.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y));
  passEncoder_stats.end();

  const passEncoder_norm = commandEncoder.beginComputePass();
  passEncoder_norm.setPipeline(normPipeline);
  passEncoder_norm.setBindGroup(0, normBindGroup);
  passEncoder_norm.setBindGroup(1, createBindGroup(device, r_r_r_BindLayout, [inputBuffer, gammaBuffer, betaBuffer]));
  passEncoder_norm.setBindGroup(2, createBindGroup(device, r_BindLayout, [statsResultBuffer]));
  passEncoder_norm.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_norm.end();

  return normResultBuffer;
}

function inlineFFN(
  device,
  queue,
  commandEncoder,
  context,
  n_embed,
  hidden_size,
  inputBuffer,
  firstLayerWeightsBuffer,
  firstLayerBiasBuffer,
  secondLayerWeightsBuffer,
  secondLayerBiasBuffer
) {
  const firstLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const firstLayerResultBuffer = createBuffer(device, bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const firstLayerBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [
    firstLayerUniformBuffer,
    firstLayerBiasBuffer,
    firstLayerWeightsBuffer,
    firstLayerResultBuffer,
  ]);
  queue.writeBuffer(firstLayerUniformBuffer, 0, new Uint32Array([context, hidden_size, n_embed]));

  const geluUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const geluResultBuffer = createBuffer(device, bufferSizeCalc(context, hidden_size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const geluBindGroup = createBindGroup(device, u_s_BindLayout, [geluUniformBuffer, geluResultBuffer]);
  queue.writeBuffer(geluUniformBuffer, 0, new Uint32Array([context, hidden_size]));

  const secondLayerUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const secondLayerResultBuffer = createBuffer(device, bufferSizeCalc(context, n_embed), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const secondLayerBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [
    secondLayerUniformBuffer,
    secondLayerBiasBuffer,
    secondLayerWeightsBuffer,
    secondLayerResultBuffer,
  ]);
  queue.writeBuffer(secondLayerUniformBuffer, 0, new Uint32Array([context, n_embed, hidden_size]));

  const passEncoder_first = commandEncoder.beginComputePass();
  passEncoder_first.setPipeline(FFNpipeline);
  passEncoder_first.setBindGroup(0, firstLayerBindGroup);
  passEncoder_first.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder_first.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
  passEncoder_first.end();

  const passEncoder_gelu = commandEncoder.beginComputePass();
  passEncoder_gelu.setPipeline(GELUpipeline);
  passEncoder_gelu.setBindGroup(0, geluBindGroup);
  passEncoder_gelu.setBindGroup(1, createBindGroup(device, r_BindLayout, [firstLayerResultBuffer]));
  passEncoder_gelu.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(hidden_size, workgroup_X));
  passEncoder_gelu.end();

  const passEncoder_second = commandEncoder.beginComputePass();
  passEncoder_second.setPipeline(FFNpipeline);
  passEncoder_second.setBindGroup(0, secondLayerBindGroup);
  passEncoder_second.setBindGroup(1, createBindGroup(device, r_BindLayout, [geluResultBuffer]));
  passEncoder_second.dispatchWorkgroups(workgroupCalc(context, workgroup_Y), workgroupCalc(n_embed, workgroup_X));
  passEncoder_second.end();

  return secondLayerResultBuffer;
}

function inlineAttention(
  device,
  queue,
  commandEncoder,
  seq_length,
  n_embd,
  attentionDotProductScale,
  inputBuffer,
  n_head,
  qkvWeightsBuffer,
  qkvBiasBuffer,
  linearWeightsBuffer,
  linearBiasBuffer
) {
  if (n_embd % n_head != 0) {
    throw new Error("cols must be divisible by n_head");
  }

  const qkvUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const qkvResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, 3 * n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const qkvBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [qkvUniformBuffer, qkvBiasBuffer, qkvWeightsBuffer, qkvResultBuffer]);
  queue.writeBuffer(qkvUniformBuffer, 0, new Uint32Array([seq_length, 3 * n_embd, n_embd]));

  const splitQKVUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const splitQResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitKResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitVResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const splitQKVBindGroup = createBindGroup(device, u_s_s_s_BindLayout, [splitQKVUniformBuffer, splitQResultBuffer, splitKResultBuffer, splitVResultBuffer]);
  queue.writeBuffer(splitQKVUniformBuffer, 0, new Uint32Array([seq_length, n_embd]));

  const attentionWeightsUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionWeightsResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionWeightsBindGroup = createBindGroup(device, u_s_BindLayout, [attentionWeightsUniformBuffer, attentionWeightsResultBuffer]);
  queue.writeBuffer(attentionWeightsUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head, n_embd / n_head, n_embd]));

  const multiplyUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const multiplyResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const multiplyBindGroup = createBindGroup(device, u_s_BindLayout, [multiplyUniformBuffer, multiplyResultBuffer]);
  queue.writeBuffer(multiplyUniformBuffer, 0, new Uint32Array([seq_length, seq_length * n_head]));
  queue.writeBuffer(multiplyUniformBuffer, 8, new Float32Array([attentionDotProductScale]));

  const causalMaskUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const causalMaskResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, seq_length * n_head), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const causalMaskBindGroup = createBindGroup(device, u_s_BindLayout, [causalMaskUniformBuffer, causalMaskResultBuffer]);
  queue.writeBuffer(causalMaskUniformBuffer, 0, new Uint32Array([seq_length * n_head, seq_length])); // Transposes! This is needed for softmax.

  const attentionValuesUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const attentionValuesResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const attentionValuesBindGroup = createBindGroup(device, u_s_BindLayout, [attentionValuesUniformBuffer, attentionValuesResultBuffer]);
  queue.writeBuffer(attentionValuesUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_head, n_embd / n_head]));

  const linearUniformBuffer = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const linearResultBuffer = createBuffer(device, bufferSizeCalc(seq_length, n_embd), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const linearBindGroup = createBindGroup(device, u_r_r_s_BindLayout, [linearUniformBuffer, linearBiasBuffer, linearWeightsBuffer, linearResultBuffer]);
  queue.writeBuffer(linearUniformBuffer, 0, new Uint32Array([seq_length, n_embd, n_embd]));

  const passEncoder_qkv = commandEncoder.beginComputePass();
  passEncoder_qkv.setPipeline(FFNpipeline);
  passEncoder_qkv.setBindGroup(0, qkvBindGroup);
  passEncoder_qkv.setBindGroup(1, createBindGroup(device, r_BindLayout, [inputBuffer]));
  passEncoder_qkv.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(3 * n_embd, workgroup_X));
  passEncoder_qkv.end();

  const passEncoder_splitQKV = commandEncoder.beginComputePass();
  passEncoder_splitQKV.setPipeline(splitQKVpipeline);
  passEncoder_splitQKV.setBindGroup(0, splitQKVBindGroup);
  passEncoder_splitQKV.setBindGroup(1, createBindGroup(device, r_BindLayout, [qkvResultBuffer]));
  passEncoder_splitQKV.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_splitQKV.end();

  const passEncoder_attentionWeights = commandEncoder.beginComputePass();
  passEncoder_attentionWeights.setPipeline(attentionWeightsPipeline);
  passEncoder_attentionWeights.setBindGroup(0, attentionWeightsBindGroup);
  passEncoder_attentionWeights.setBindGroup(1, createBindGroup(device, r_r_BindLayout, [splitQResultBuffer, splitKResultBuffer]));
  passEncoder_attentionWeights.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
  passEncoder_attentionWeights.end();

  const passEncoder_multiply = commandEncoder.beginComputePass();
  passEncoder_multiply.setPipeline(multiplyPipeline);
  passEncoder_multiply.setBindGroup(0, multiplyBindGroup);
  passEncoder_multiply.setBindGroup(1, createBindGroup(device, r_BindLayout, [attentionWeightsResultBuffer]));
  passEncoder_multiply.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(seq_length * n_head, workgroup_X));
  passEncoder_multiply.end();

  const passEncoder_causalMask = commandEncoder.beginComputePass();
  passEncoder_causalMask.setPipeline(causalMaskPipeline);
  passEncoder_causalMask.setBindGroup(0, causalMaskBindGroup);
  passEncoder_causalMask.setBindGroup(1, createBindGroup(device, r_BindLayout, [multiplyResultBuffer]));
  passEncoder_causalMask.dispatchWorkgroups(workgroupCalc(seq_length * n_head, workgroup_Y), workgroupCalc(seq_length, workgroup_X));
  passEncoder_causalMask.end();

  // This is a sloppy-ish solution to the casual mask buffer being processed with every head at once. Obviously, this could be fixed if we just did this in a smarter way but I only realized you could do this at the end. Still learning WebGPU!
  const softmaxOutputBuffer = createBuffer(
    device,
    bufferSizeCalc(seq_length, seq_length * n_head),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  for (let i = 0; i < n_head; i++) {
    const softmaxInputBuffer = createBuffer(
      device,
      bufferSizeCalc(seq_length, seq_length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    commandEncoder.copyBufferToBuffer(
      causalMaskResultBuffer,
      i * bufferSizeCalc(seq_length, seq_length),
      softmaxInputBuffer,
      0,
      bufferSizeCalc(seq_length, seq_length)
    );
    const softMaxResultBuffer = inlineSoftmax(device, queue, commandEncoder, seq_length, seq_length, softmaxInputBuffer);
    commandEncoder.copyBufferToBuffer(
      softMaxResultBuffer,
      0,
      softmaxOutputBuffer,
      i * bufferSizeCalc(seq_length, seq_length),
      bufferSizeCalc(seq_length, seq_length)
    );
  }

  const passEncoder_attentionValues = commandEncoder.beginComputePass();
  passEncoder_attentionValues.setPipeline(attentionValuesPipeline);
  passEncoder_attentionValues.setBindGroup(0, attentionValuesBindGroup);
  passEncoder_attentionValues.setBindGroup(1, createBindGroup(device, r_r_BindLayout, [softmaxOutputBuffer, splitVResultBuffer]));
  passEncoder_attentionValues.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_attentionValues.end();

  const passEncoder_linear = commandEncoder.beginComputePass();
  passEncoder_linear.setPipeline(FFNpipeline);
  passEncoder_linear.setBindGroup(0, linearBindGroup);
  passEncoder_linear.setBindGroup(1, createBindGroup(device, r_BindLayout, [attentionValuesResultBuffer]));
  passEncoder_linear.dispatchWorkgroups(workgroupCalc(seq_length, workgroup_Y), workgroupCalc(n_embd, workgroup_X));
  passEncoder_linear.end();

  return linearResultBuffer;
}
