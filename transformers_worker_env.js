/**
 * @file Web Worker environment configuration for parallel model inference.
 * Isolates model loading and inference from main thread to prevent UI blocking.
 * Uses zero-copy techniques when possible for memory efficiency.
 * 
 * @module workers/env
 */

import fs from 'node:fs';
import path from 'node:path';

/**
 * Configuration for worker thread environment.
 * Enables atomic file reads from OPFS (Origin Private File System) on phones.
 */
export const workerEnv = {
  // Enable Origin Private File System for zero-latency synchronous reads
  useOPFS: typeof navigator !== 'undefined' && 'storage' in navigator,
  
  // Enable Cross-Origin Resource Sharing for efficient model streaming
  allowCrossOrigin: true,
  
  // Buffer size for streaming reads (32KB optimal for mobile)
  streamBufferSize: 32 * 1024,
  
  // Enable zero-copy buffer transfers when available
  useSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
};

/**
 * Initializes worker environment for model inference.
 * Sets up file system access and memory management.
 */
export async function initializeWorkerEnvironment() {
  if (typeof self === 'undefined') {
    throw new Error('This module can only run in a Web Worker');
  }

  // Enable high-resolution timing for performance measurement
  if (typeof performance === 'undefined') {
    console.warn('performance API not available in worker');
  }

  return {
    isWorker: true,
    supportedAPIs: {
      opfs: workerEnv.useOPFS,
      sharedArrayBuffer: workerEnv.useSharedArrayBuffer,
      wasmSimd: typeof WebAssembly !== 'undefined' &&
        typeof WebAssembly.instantiate === 'function',
    },
  };
}

/**
 * Creates a message handler for the worker thread.
 * Coordinates between main thread and inference engine.
 */
export function createWorkerMessageHandler(engine) {
  return async function onMessage(event) {
    const { id, method, args } = event.data;
    
    try {
      // Route method calls to engine
      const result = await engine[method]?.(...args);
      
      // Send result back to main thread
      self.postMessage({
        id,
        status: 'success',
        result,
      });
    } catch (error) {
      // Send error back to main thread
      self.postMessage({
        id,
        status: 'error',
        error: error.message,
      });
    }
  };
}

/**
 * Measures performance of worker tasks for optimization.
 */
export function measureWorkerTask(taskName, fn) {
  const start = performance.now();
  const result = fn();
  const duration = performance.now() - start;
  
  console.log(`[Worker] ${taskName} took ${duration.toFixed(2)}ms`);
  
  return result;
}

export default workerEnv;
