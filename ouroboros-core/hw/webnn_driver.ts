/**
 * @file WebNN (Web Neural Network) driver for NPU acceleration.
 * Direct MLGraph execution on mobile NPU/GPU.
 * @module hw/webnn_driver
 */

import type { ComputeConfig } from '../types';

/**
 * WebNN executor: maps compute tasks directly to NPU/GPU.
 */
export class WebNNDriver {
  private context: any = null; // MLContext
  private graph: any = null; // MLGraph
  private operands: Map<string, any> = new Map();

  /**
   * Initialize WebNN context (requires navigator.ml).
   */
  async initialize(): Promise<void> {
    const ml = (navigator as any).ml;
    if (!ml) throw new Error('WebNN not available');

    // Create device context
    try {
      this.context = await ml.createContext();
    } catch (e) {
      console.error('Failed to create WebNN context:', e);
      throw e;
    }
  }

  /**
   * Build MLGraph for matrix multiplication.
   */
  buildMatmulGraph(
    inputShape: number[],
    weightsShape: number[],
    outputShape: number[]
  ): void {
    if (!this.context) throw new Error('Context not initialized');

    const builder = this.context.createGraphBuilder();

    // Define operands
    const input = builder.input('input', {
      dataType: 'float32',
      shape: inputShape,
    });

    const weights = builder.constant(
      { dataType: 'float32', shape: weightsShape },
      new Float32Array(weightsShape.reduce((a, b) => a * b, 1))
    );

    // Matrix multiply
    const output = builder.matmul(input, weights);

    // Build graph
    this.graph = builder.build({ output });
    this.operands.set('input', input);
    this.operands.set('weights', weights);
  }

  /**
   * Execute graph with input tensor.
   */
  async execute(
    inputData: Float32Array,
    config: ComputeConfig
  ): Promise<Float32Array> {
    if (!this.graph) throw new Error('Graph not built');

    // Create inputs
    const inputs = {
      input: {
        dataType: 'float32',
        data: inputData,
        dimensions: config.shape || [1, inputData.length],
      },
    };

    // Compute
    const result = await this.context.compute(this.graph, inputs);

    // Extract output
    const outputData = result.outputs.output;
    return new Float32Array(outputData);
  }

  /**
   * Get graph execution stats.
   */
  getStats(): any {
    return {
      graphBuilt: this.graph !== null,
      operandCount: this.operands.size,
    };
  }

  /**
   * Destroy WebNN context.
   */
  destroy(): void {
    this.graph = null;
    this.operands.clear();
    this.context = null;
  }
}

export default WebNNDriver;
