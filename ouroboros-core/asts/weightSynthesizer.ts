/**
 * @file Direct binary weight dequantization and LoRA synthesis.
 * Decompresses quantized weights in-place in SharedArrayBuffer.
 * @module asts/weightSynthesizer
 */

import type { QuantizationConfig } from '../types';

/**
 * Hardware-accelerated weight dequantizer.
 * Handles Q4_0, Q8_0, and mixed quantization schemes.
 */
export class WeightSynthesizer {
  private sharedBuffer: SharedArrayBuffer;
  private quantConfig: QuantizationConfig;

  constructor(sharedBuffer: SharedArrayBuffer, quantConfig: QuantizationConfig) {
    this.sharedBuffer = sharedBuffer;
    this.quantConfig = quantConfig;
  }

  /**
   * Dequantize Q4_0 block format inline.
   * Q4_0: 32 ints (4 bits each) + 1 fp16 scale per 32-element block
   */
  dequantizeQ4_0(
    inputOffset: number,
    outputOffset: number,
    elementCount: number
  ): void {
    const input = new Uint8Array(this.sharedBuffer, inputOffset);
    const output = new Float32Array(this.sharedBuffer, outputOffset);

    const blockSize = 32;
    const blockByteSize = 18; // 2 bytes scale + 16 bytes data

    let inputIdx = 0;
    let outputIdx = 0;

    for (let blockIdx = 0; blockIdx * blockSize < elementCount; blockIdx++) {
      // Read scale (fp16)
      const scaleBytesLow = input[inputIdx++];
      const scaleBytesHigh = input[inputIdx++];
      const scale = this.fp16ToFloat32(
        (scaleBytesHigh << 8) | scaleBytesLow
      );

      // Dequantize 32 4-bit values
      for (let i = 0; i < 16; i++) {
        const byte = input[inputIdx++];
        const lower4 = (byte & 0x0f) - 8; // 4-bit signed
        const upper4 = ((byte >> 4) & 0x0f) - 8;

        output[outputIdx++] = lower4 * scale;
        if (outputIdx < elementCount) {
          output[outputIdx++] = upper4 * scale;
        }
      }
    }
  }

  /**
   * Dequantize Q8_0 block format inline.
   * Q8_0: 32 ints (8 bits each) + 1 fp16 scale per block
   */
  dequantizeQ8_0(
    inputOffset: number,
    outputOffset: number,
    elementCount: number
  ): void {
    const input = new Int8Array(this.sharedBuffer, inputOffset);
    const output = new Float32Array(this.sharedBuffer, outputOffset);

    const blockSize = 32;
    let inputIdx = 0;
    let outputIdx = 0;

    for (let blockIdx = 0; blockIdx * blockSize < elementCount; blockIdx++) {
      // Read scale (first 2 bytes as fp16)
      const scaleBytes = new Uint16Array(
        this.sharedBuffer,
        inputOffset + blockIdx * 34
      );
      const scale = this.fp16ToFloat32(scaleBytes[0]);

      // Dequantize 32 8-bit values
      const dataOffset = inputOffset + blockIdx * 34 + 2;
      for (let i = 0; i < blockSize && outputIdx < elementCount; i++) {
        const val = new Int8Array(this.sharedBuffer, dataOffset + i, 1)[0];
        output[outputIdx++] = val * scale;
      }
    }
  }

  /**
   * Apply LoRA (Low-Rank Adaptation) corrections.
   * output = base_weight + (LoRA_A @ LoRA_B)
   */
  applyLoRA(
    baseWeightsOffset: number,
    loraAOffset: number,
    loraBOffset: number,
    m: number, // rows
    n: number, // cols
    r: number, // rank
    scale: number = 1.0
  ): void {
    const baseWeights = new Float32Array(this.sharedBuffer, baseWeightsOffset, m * n);
    const loraA = new Float32Array(this.sharedBuffer, loraAOffset, m * r);
    const loraB = new Float32Array(this.sharedBuffer, loraBOffset, r * n);

    // Compute LoRA_A @ LoRA_B into temporary (allocate in buffer tail)
    const tempOffset = baseWeightsOffset + m * n * 4;
    const temp = new Float32Array(this.sharedBuffer, tempOffset, m * n);

    // Matrix multiply: temp = loraA @ loraB
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < r; k++) {
          sum += loraA[i * r + k] * loraB[k * n + j];
        }
        temp[i * n + j] = sum * scale;
      }
    }

    // Add to base weights: base += LoRA correction
    for (let i = 0; i < m * n; i++) {
      baseWeights[i] += temp[i];
    }
  }

  /**
   * Synthesize full tensor from quantized base + LoRA adapters.
   */
  synthesizeWeight(
    tensorName: string,
    quantFormat: string,
    inputOffset: number,
    outputOffset: number,
    size: number,
    hasLoRA: boolean = false,
    loraConfig?: any
  ): void {
    // Step 1: Dequantize
    switch (quantFormat) {
      case 'q4_0':
        this.dequantizeQ4_0(inputOffset, outputOffset, size);
        break;
      case 'q8_0':
        this.dequantizeQ8_0(inputOffset, outputOffset, size);
        break;
      default:
        // Already float, just copy
        const input = new Uint8Array(
          this.sharedBuffer,
          inputOffset,
          size * 4
        );
        const output = new Uint8Array(this.sharedBuffer, outputOffset);
        output.set(input);
    }

    // Step 2: Apply LoRA if present
    if (hasLoRA && loraConfig) {
      this.applyLoRA(
        outputOffset,
        loraConfig.loraAOffset,
        loraConfig.loraBOffset,
        loraConfig.m,
        loraConfig.n,
        loraConfig.r,
        loraConfig.scale || 1.0
      );
    }
  }

  /**
   * Convert fp16 to fp32.
   */
  private fp16ToFloat32(fp16: number): number {
    const sign = (fp16 & 0x8000) >> 15;
    const exponent = (fp16 & 0x7c00) >> 10;
    const mantissa = fp16 & 0x03ff;

    if (exponent === 0) {
      return sign === 0 ? 0 : -0;
    }
    if (exponent === 31) {
      return mantissa === 0
        ? sign === 0
          ? Infinity
          : -Infinity
        : NaN;
    }

    const value =
      Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
    return sign === 0 ? value : -value;
  }
}

export default WeightSynthesizer;
