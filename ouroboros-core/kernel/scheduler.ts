/**
 * @file Lockless asynchronous pipeline manager for ASTS inference.
 * Implements double-buffering ping-pong pattern to hide I/O latency.
 * GPU computes Layer N while storage fetches Layer N+1 = zero wait states.
 * @module kernel/scheduler
 */

import type { PipelineConfig, ComputeTask, BufferState } from '../types';

/**
 * Atomic state machine for lockless ping-pong buffer swapping.
 * No mutexes or locks - uses atomic operations and memory ordering.
 */
export class LocklessScheduler {
  private bufferA: SharedArrayBuffer;
  private bufferB: SharedArrayBuffer;
  private activeBuffer: Int32Array; // [current_buffer_index]
  private taskQueue: ComputeTask[] = [];
  private isProcessing = false;
  private hardwareReady = false;
  private fetchInFlight = false;
  private lastLayerIdx = -1;
  private config: PipelineConfig;

  constructor(config: PipelineConfig) {
    this.config = config;
    this.bufferA = new SharedArrayBuffer(config.bufferSizePerLayer);
    this.bufferB = new SharedArrayBuffer(config.bufferSizePerLayer);
    this.activeBuffer = new Int32Array(new SharedArrayBuffer(4));
    this.activeBuffer[0] = 0; // Start with buffer A
  }

  /**
   * Get the current active buffer for GPU writes.
   * No locks - just atomic read of buffer index.
   */
  getActiveBuffer(): SharedArrayBuffer {
    const idx = Atomics.load(this.activeBuffer, 0);
    return idx === 0 ? this.bufferA : this.bufferB;
  }

  /**
   * Get the inactive buffer for fetching next layer.
   * Always opposite of active.
   */
  getInactiveBuffer(): SharedArrayBuffer {
    const idx = Atomics.load(this.activeBuffer, 0);
    return idx === 0 ? this.bufferB : this.bufferA;
  }

  /**
   * Enqueue a compute task (prefill, decode, etc.).
   * Tasks fire immediately if GPU is idle, otherwise queue.
   */
  enqueueTask(task: ComputeTask): void {
    this.taskQueue.push(task);
    this.tryDequeueTask();
  }

  /**
   * Try to dequeue and execute the next task.
   * If GPU is processing, abort. Otherwise execute and trigger fetch.
   */
  private async tryDequeueTask(): Promise<void> {
    if (this.isProcessing || this.taskQueue.length === 0) return;

    this.isProcessing = true;
    const task = this.taskQueue.shift()!;

    try {
      // Mark the active buffer ready for GPU compute
      this.hardwareReady = false;

      // Fire compute on GPU/NPU (non-blocking)
      await this.dispatchComputeTask(task);

      // Mark compute complete
      this.hardwareReady = true;

      // If not already fetching, trigger prefetch of next layer
      if (!this.fetchInFlight && task.nextLayerIdx !== undefined) {
        this.fetchInFlight = true;
        this.triggerAsyncFetch(task.nextLayerIdx);
      }

      // Immediately try next task
      this.isProcessing = false;
      this.tryDequeueTask();
    } catch (e) {
      console.error('[Scheduler] Compute task failed:', e);
      this.isProcessing = false;
    }
  }

  /**
   * Dispatch compute to GPU, NPU, or CPU backend.
   * Returns immediately - compute happens asynchronously.
   */
  private async dispatchComputeTask(task: ComputeTask): Promise<void> {
    const activeBuffer = this.getActiveBuffer();

    // Prepare compute command buffer
    const cmd = {
      type: task.computeType,
      inputBuffer: activeBuffer,
      outputBuffer: activeBuffer, // In-place compute
      layerIdx: task.layerIdx,
      batchSize: task.batchSize,
      tokenCount: task.tokenCount,
    };

    // Post to hardware driver worker
    return new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        reject(new Error('Compute task timeout'));
      }, this.config.computeTimeoutMs);

      // Send to backend driver (GPU, NPU, or CPU)
      const msgId = Math.random();
      const handler = (evt: MessageEvent) => {
        if (evt.data.id === msgId) {
          clearTimeout(timeoutHandle);
          self.removeEventListener('message', handler);
          if (evt.data.status === 'error') {
            reject(new Error(evt.data.error));
          } else {
            resolve();
          }
        }
      };
      self.addEventListener('message', handler);
      self.postMessage({ id: msgId, cmd });
    });
  }

  /**
   * Trigger async fetch of next layer into inactive buffer.
   * Uses the inactive buffer while GPU processes current layer.
   */
  private triggerAsyncFetch(layerIdx: number): void {
    if (layerIdx === this.lastLayerIdx) {
      this.fetchInFlight = false;
      return; // Already fetched
    }

    const inactiveBuffer = this.getInactiveBuffer();
    this.lastLayerIdx = layerIdx;

    // Post fetch request to I/O worker
    const msgId = Math.random();
    const handler = (evt: MessageEvent) => {
      if (evt.data.id === msgId) {
        self.removeEventListener('message', handler);
        if (evt.data.status === 'success') {
          // Data is now in inactive buffer, ready to swap
          this.swapBuffers();
        }
        this.fetchInFlight = false;
        this.tryDequeueTask();
      }
    };
    self.addEventListener('message', handler);
    self.postMessage({
      id: msgId,
      cmd: {
        type: 'fetch_layer',
        layerIdx,
        buffer: inactiveBuffer,
      },
    });
  }

  /**
   * Atomic swap of active buffer index.
   * Next task will use the freshly-fetched buffer.
   */
  private swapBuffers(): void {
    const current = Atomics.load(this.activeBuffer, 0);
    const next = current === 0 ? 1 : 0;
    Atomics.store(this.activeBuffer, 0, next);
  }

  /**
   * Get current pipeline stats for telemetry.
   */
  getStats(): {
    queuedTasks: number;
    isProcessing: boolean;
    isFetching: boolean;
    activeBufferIndex: number;
  } {
    return {
      queuedTasks: this.taskQueue.length,
      isProcessing: this.isProcessing,
      isFetching: this.fetchInFlight,
      activeBufferIndex: Atomics.load(this.activeBuffer, 0),
    };
  }
}

export default LocklessScheduler;
