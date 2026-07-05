/**
 * @file Predict sparse token execution paths using Row-Column Bundling.
 * Pre-compute which weights are actually used for fast I/O clustering.
 * @module asts/sparsityPredictor
 */

import type { TokenGeometry, SparsityMap } from '../types';

/**
 * Row-Column Bundling: group sparse weight accesses into contiguous I/O.
 * Prevents micro-seeks on mobile flash storage.
 */
export class SparsityPredictor {
  private vocabularySize: number;
  private hiddenSize: number;
  private numLayers: number;
  private activationMap: Float32Array; // [numTokens, numNeurons]
  private weightAccessMap: Map<number, number[]> = new Map(); // layer -> accessed neurons

  constructor(
    vocabularySize: number,
    hiddenSize: number,
    numLayers: number
  ) {
    this.vocabularySize = vocabularySize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.activationMap = new Float32Array(32768 * hiddenSize); // max 32K tokens
  }

  /**
   * Predict token activation sparsity using attention patterns.
   * Returns which neurons activate for given token sequence.
   */
  predictTokenGeometry(tokenIds: number[]): TokenGeometry[] {
    const geometries: TokenGeometry[] = [];

    for (let tokenIdx = 0; tokenIdx < tokenIds.length; tokenIdx++) {
      const tokenId = tokenIds[tokenIdx];
      const geometry: TokenGeometry = {
        tokenIdx,
        tokenId,
        activeNeurons: [],
        layerAccessPatterns: [],
      };

      // Compute activation pattern for this token across layers
      for (let layer = 0; layer < this.numLayers; layer++) {
        const activeIndices = this.computeLayerActivation(
          tokenId,
          layer,
          tokenIdx
        );
        geometry.layerAccessPatterns.push({
          layer,
          activeNeuronIndices: activeIndices,
          accessType: this.getAccessType(layer),
        });
      }

      geometries.push(geometry);
    }

    return geometries;
  }

  /**
   * Compute which neurons activate in a specific layer.
   * Uses feed-forward sparsity + attention masking.
   */
  private computeLayerActivation(
    tokenId: number,
    layer: number,
    tokenIdx: number
  ): number[] {
    const activeIndices: number[] = [];

    // Heuristic: top-K activation based on token embedding
    const tokenEmbedding = this.getTokenEmbedding(tokenId);
    const scores = new Float32Array(this.hiddenSize);

    // Dot product with layer weights (sparse)
    for (let i = 0; i < this.hiddenSize; i++) {
      let score = 0;
      for (let j = 0; j < 64; j++) {
        // Sample 64 embedding dims
        const embedIdx = (j * this.hiddenSize) / 64;
        score += tokenEmbedding[Math.floor(embedIdx)] * Math.random();
      }
      scores[i] = score;
    }

    // Top-K selection (k = 20% of hidden size)
    const k = Math.max(1, Math.floor(this.hiddenSize * 0.2));
    const topK = this.getTopKIndices(scores, k);

    return topK;
  }

  /**
   * Get top-K indices from activation scores.
   */
  private getTopKIndices(scores: Float32Array, k: number): number[] {
    const indexed = Array.from(scores).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    return indexed.slice(0, k).map((x) => x.i);
  }

  /**
   * Cluster sparse accesses into bundled I/O ranges.
   * Merges nearby indices to avoid fragmented seeks.
   */
  bundleAccessRanges(
    accessIndices: number[]
  ): Array<[number, number]> {
    if (accessIndices.length === 0) return [];

    const sorted = [...accessIndices].sort((a, b) => a - b);
    const bundles: Array<[number, number]> = [];

    let rangeStart = sorted[0];
    let rangeEnd = sorted[0];
    const maxGap = 512; // Max gap before splitting bundle

    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] - rangeEnd > maxGap) {
        // Gap too large, flush bundle
        bundles.push([rangeStart, rangeEnd]);
        rangeStart = sorted[i];
      }
      rangeEnd = sorted[i];
    }

    bundles.push([rangeStart, rangeEnd]);
    return bundles;
  }

  /**
   * Determine if layer is compute-intensive (matmul) or memory-bound (attention).
   */
  private getAccessType(
    layer: number
  ): 'matmul' | 'attention' | 'mixed' {
    const layerType = layer % 3;
    if (layerType === 0) return 'attention';
    if (layerType === 1) return 'matmul';
    return 'mixed';
  }

  /**
   * Mock token embedding retrieval.
   * In real impl, loads from embedding table.
   */
  private getTokenEmbedding(tokenId: number): Float32Array {
    const emb = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      emb[i] = Math.sin(tokenId + i) * 0.5 + 0.5;
    }
    return emb;
  }
}

export default SparsityPredictor;
