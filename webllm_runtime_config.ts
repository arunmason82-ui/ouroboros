/**
 * @file WebGPU runtime configuration for LLM inference.
 * Manages GPU memory allocation, cache strategies, and compute pipeline orchestration.
 * Critical for preventing VRAM overflow and maintaining interactive performance.
 * 
 * @module config
 */

import * as tvmjs from '@mlc-ai/web-runtime';
import log from 'loglevel';

/**
 * GPU memory requirements mapping per hardware type.
 * Prevents OOM crashes and thermal throttling on mobile devices.
 */
export interface GPUMemoryConfig {
  // NPU (Neural Processing Unit) - lowest power, limited VRAM (1-2GB)
  npu: {
    maxVram: 1024 * 1024 * 1024, // 1GB
    kvCachePageSize: 8, // smaller pages for limited memory
    maxBatchSize: 1,
    prefillChunkSize: 128,
  };
  
  // Integrated GPU - mid-range VRAM (2-4GB)
  integratedGpu: {
    maxVram: 2 * 1024 * 1024 * 1024, // 2GB
    kvCachePageSize: 16,
    maxBatchSize: 2,
    prefillChunkSize: 512,
  };
  
  // Discrete GPU - high VRAM (6GB+)
  discreteGpu: {
    maxVram: 6 * 1024 * 1024 * 1024, // 6GB
    kvCachePageSize: 32,
    maxBatchSize: 4,
    prefillChunkSize: 2048,
  };
}

/**
 * Detects GPU hardware and returns appropriate configuration.
 * Optimizes memory usage based on device capabilities.
 */
export async function detectGPUCapabilities(): Promise<{
  vendor: string;
  memoryConfig: any;
  maxStorageBufferSize: number;
}> {
  const gpuInfo = await tvmjs.detectGPUDevice();
  
  if (!gpuInfo) {
    throw new Error('WebGPU not available');
  }

  const description = gpuInfo.adapterInfo.description.toLowerCase();
  const maxStorage = gpuInfo.device.limits.maxStorageBufferBindingSize;
  
  let vendor = 'unknown';
  let memoryProfile = 'discreteGpu';
  
  // Detect GPU vendor and set memory profile
  if (description.includes('apple') || description.includes('metal')) {
    vendor = 'Apple';
    memoryProfile = description.includes('m1') ? 'integratedGpu' : 'discreteGpu';
  } else if (description.includes('nvidia')) {
    vendor = 'NVIDIA';
    memoryProfile = 'discreteGpu';
  } else if (description.includes('amd')) {
    vendor = 'AMD';
    memoryProfile = 'discreteGpu';
  } else if (description.includes('intel')) {
    vendor = 'Intel';
    memoryProfile = 'integratedGpu';
  } else if (description.includes('qualcomm') || description.includes('adreno')) {
    vendor = 'Qualcomm';
    memoryProfile = 'npu';
  }

  const memoryConfig = GPUMemoryConfig[memoryProfile as keyof GPUMemoryConfig];
  
  log.info(`Detected GPU: ${vendor}`);
  log.info(`Memory profile: ${memoryProfile}`);
  log.info(`Max storage buffer: ${(maxStorage / 1e9).toFixed(2)}GB`);

  return {
    vendor,
    memoryConfig,
    maxStorageBufferSize: maxStorage,
  };
}

/**
 * KV-Cache configuration for attention mechanisms.
 * Paged cache prevents memory fragmentation during generation.
 */
export interface KVCacheConfig {
  pageSize: number; // elements per page
  maxNumPages: number;
  numLayers: number;
  headDim: number;
  dtype: 'float32' | 'float16';
}

/**
 * Calculates KV cache allocation based on context window and model.
 * Critical for preventing OOM during long-context inference.
 */
export function calculateKVCacheRequirement(
  contextWindowSize: number,
  hiddenSize: number,
  numAttentionHeads: number,
  numLayers: number,
  dtype: 'float32' | 'float16' = 'float16',
): number {
  const bytesPerElement = dtype === 'float32' ? 4 : 2;
  
  // KV cache stores key and value for each token position
  const kvCachePerLayer =
    contextWindowSize * hiddenSize * bytesPerElement * 2; // 2 for K and V
  
  const totalKVCache = kvCachePerLayer * numLayers;
  
  return totalKVCache;
}

/**
 * Validates model configuration against hardware limits.
 * Prevents runtime crashes from exceeding device capabilities.
 */
export function validateModelFitInMemory(
  modelSizeBytes: number,
  kvCacheBytes: number,
  maxVramBytes: number,
): {
  fits: boolean;
  remainingVram: number;
  utilization: number;
} {
  const totalRequired = modelSizeBytes + kvCacheBytes;
  const remainingVram = maxVramBytes - totalRequired;
  const utilization = (totalRequired / maxVramBytes) * 100;
  
  // Keep 10% headroom for compute buffers
  const fits = remainingVram > maxVramBytes * 0.1;
  
  return {
    fits,
    remainingVram,
    utilization,
  };
}

export default {
  detectGPUCapabilities,
  calculateKVCacheRequirement,
  validateModelFitInMemory,
};
