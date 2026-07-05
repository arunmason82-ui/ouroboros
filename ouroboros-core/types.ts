/**
 * @file A.S.T.S. architecture type definitions
 * Shared interfaces across all layers
 */

export interface ExecutionContext {
  bufferA: SharedArrayBuffer;
  bufferB: SharedArrayBuffer;
  hardwareTarget: string | null;
  computeWorker: Worker | null;
  ioWorker: Worker | null;
}

export interface AuditResult {
  primaryBackend: 'webgpu' | 'webnn' | 'wasm' | 'cpu';
  capabilities: HardwareCapabilities;
  benchmarks: any[];
  elapsedMs: number;
  isRemoteStreamingMode: boolean;
}

export interface HardwareCapabilities {
  hasWebGPU: boolean;
  hasWebNN: boolean;
  hasWasmSimd: boolean;
  isMobile: boolean;
  cpuCores: number;
  maxMemoryMB: number;
}

export interface TensorMetadata {
  name: string;
  dtype: string;
  shape: number[];
  size: bigint;
}

export interface TopologyMap {
  tensors: TensorMetadata[];
  offsets: Record<string, { offset: string; size: string }>;
  totalSize: string;
}

export interface TokenGeometry {
  tokenIdx: number;
  tokenId: number;
  activeNeurons: number[];
  layerAccessPatterns: Array<{
    layer: number;
    activeNeuronIndices: number[];
    accessType: 'matmul' | 'attention' | 'mixed';
  }>;
}

export interface SparsityMap {
  bundledRanges: Array<[number, number]>;
  accessPatterns: Map<number, TokenGeometry>;
}

export interface ComputeConfig {
  inputOffset?: number;
  weightsOffset?: number;
  outputOffset?: number;
  computeType: 'prefill' | 'decode' | 'attention' | 'matmul';
  layerIdx: number;
  batchSize: number;
  tokenCount: number;
  shape?: number[];
}

export interface ComputeTask {
  computeType: 'prefill' | 'decode' | 'attention' | 'matmul';
  inputBuffer: SharedArrayBuffer;
  outputBuffer: SharedArrayBuffer;
  layerIdx: number;
  batchSize: number;
  tokenCount: number;
  nextLayerIdx?: number;
}

export interface PipelineConfig {
  bufferSizePerLayer: number;
  computeTimeoutMs: number;
  maxQueuedTasks: number;
}

export interface QuantizationConfig {
  format: 'q4_0' | 'q8_0' | 'mixed';
  blockSize: number;
  hasLoRA: boolean;
}
