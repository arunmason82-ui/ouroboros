/**
 * @file Parse .ouro topology map files into executable byte ranges.
 * Maps token geometry to exact file offsets for contiguous I/O.
 * @module asts/topologyParser
 */

import type { TensorMetadata, TopologyMap } from '../types';

/**
 * Reads GGUF metadata and builds precise offset map.
 * Uses actual tensor sizes to compute exact byte positions.
 */
export class TopologyParser {
  private tensorMap: Map<string, TensorMetadata> = new Map();
  private offsetMap: Map<string, { offset: bigint; size: bigint }> = new Map();
  private totalSize: bigint = BigInt(0);

  /**
   * Parse GGUF metadata structure into offset map.
   * @param metadata Raw metadata from GGUF parser
   */
  parse(metadata: any): TopologyMap {
    const tensors = metadata.tensors || [];
    let currentOffset = BigInt(0);

    for (const tensor of tensors) {
      const tensorName = tensor.name as string;
      const dtype = this.parseDtype(tensor.dtype);
      const shape = (tensor.ne || []) as number[];
      const elementCount = shape.reduce((a, b) => a * b, 1);
      const bytesPerElement = this.getBytesPerElement(dtype);
      const sizeBytes = BigInt(elementCount * bytesPerElement);

      this.tensorMap.set(tensorName, {
        name: tensorName,
        dtype,
        shape,
        size: sizeBytes,
      });

      this.offsetMap.set(tensorName, {
        offset: currentOffset,
        size: sizeBytes,
      });

      currentOffset += sizeBytes;
    }

    this.totalSize = currentOffset;

    return {
      tensors: Array.from(this.tensorMap.values()),
      offsets: Object.fromEntries(
        Array.from(this.offsetMap).map(([k, v]) => [
          k,
          { offset: v.offset.toString(), size: v.size.toString() },
        ])
      ),
      totalSize: this.totalSize.toString(),
    };
  }

  /**
   * Get byte range for specific tensor.
   * Returns [offset, size] for direct file read.
   */
  getRange(tensorName: string): [bigint, bigint] | null {
    const range = this.offsetMap.get(tensorName);
    return range ? [range.offset, range.size] : null;
  }

  /**
   * Get contiguous ranges for row-column bundled weights.
   * Clusters sparse requests into continuous I/O streams.
   */
  getBundledRanges(layerIndices: number[]): Array<[bigint, bigint]> {
    const ranges: Array<[bigint, bigint]> = [];
    let prevEnd = BigInt(0);
    let currentStart = BigInt(0);
    let currentSize = BigInt(0);

    for (const idx of layerIndices) {
      const tensorName = `layer_${idx}`;
      const range = this.offsetMap.get(tensorName);
      if (!range) continue;

      // If gap > 1MB, flush current bundle
      if (range.offset - prevEnd > BigInt(1024 * 1024)) {
        if (currentSize > 0) {
          ranges.push([currentStart, currentSize]);
        }
        currentStart = range.offset;
        currentSize = range.size;
      } else {
        // Merge into current bundle
        currentSize = range.offset + range.size - currentStart;
      }

      prevEnd = range.offset + range.size;
    }

    if (currentSize > 0) {
      ranges.push([currentStart, currentSize]);
    }

    return ranges;
  }

  private parseDtype(dtypeCode: number): string {
    const dtypeMap: Record<number, string> = {
      0: 'f32',
      1: 'f16',
      2: 'q4_0',
      3: 'q4_1',
      4: 'q8_0',
      5: 'q5_0',
      6: 'q5_1',
      7: 'q2_k',
      8: 'q3_k',
      9: 'q4_k',
      10: 'q5_k',
      11: 'q6_k',
      12: 'q8_k',
    };
    return dtypeMap[dtypeCode] || 'unknown';
  }

  private getBytesPerElement(dtype: string): number {
    switch (dtype) {
      case 'f32':
        return 4;
      case 'f16':
        return 2;
      case 'q4_0':
        return 0.5; // 4 bits
      case 'q4_1':
        return 0.5;
      case 'q8_0':
        return 1;
      case 'q5_0':
        return 0.625; // 5 bits
      default:
        return 1;
    }
  }
}

export default TopologyParser;
