/**
 * @file Low-latency state machine for ASTS lifecycle.
 * Bootstrapping -> Audit -> Range Mapping -> Sparse Calc -> Exec -> Flush
 * No async delays, synchronous where possible.
 * @module kernel/stateMachine
 */

import type { AuditResult, ExecutionContext } from '../types';

type LifecycleState =
  | 'BOOTSTRAPPING'
  | 'AUDITING'
  | 'RANGE_MAPPING'
  | 'SPARSE_CALC'
  | 'EXECUTING'
  | 'FLUSHING'
  | 'READY'
  | 'ERROR';

/**
 * Fast state machine with zero-allocation transitions.
 */
export class STSLifecycleStateMachine {
  private state: LifecycleState = 'BOOTSTRAPPING';
  private executionContext: ExecutionContext | null = null;
  private auditResult: AuditResult | null = null;
  private rangeMap: Map<number, { start: number; size: number }> = new Map();
  private sparseGeometry: Float32Array | null = null;
  private lastError: Error | null = null;
  private stateStartTime = performance.now();

  /**
   * Force synchronous state transition.
   * No promises, no delays.
   */
  transition(toState: LifecycleState): void {
    const prevState = this.state;
    this.state = toState;
    this.stateStartTime = performance.now();
    console.log(`[State] ${prevState} -> ${toState}`);
  }

  /**
   * Execute bootstrap: allocate buffers, init worker threads.
   */
  bootstrap(): void {
    if (this.state !== 'BOOTSTRAPPING') throw new Error('Invalid state');

    this.executionContext = {
      bufferA: new SharedArrayBuffer(16 * 1024 * 1024), // 16MB
      bufferB: new SharedArrayBuffer(16 * 1024 * 1024),
      hardwareTarget: null,
      computeWorker: null,
      ioWorker: null,
    };

    this.transition('AUDITING');
  }

  /**
   * Execute hardware audit: detect GPU/NPU/CPU capabilities.
   */
  async audit(auditor: any): Promise<void> {
    if (this.state !== 'AUDITING') throw new Error('Invalid state');

    try {
      this.auditResult = await auditor.audit();
      this.executionContext!.hardwareTarget = this.auditResult.primaryBackend;
      this.transition('RANGE_MAPPING');
    } catch (e) {
      this.lastError = e as Error;
      this.transition('ERROR');
      throw e;
    }
  }

  /**
   * Build dynamic file byte range map from model topology.
   */
  buildRangeMap(parser: any, metadata: any): void {
    if (this.state !== 'RANGE_MAPPING') throw new Error('Invalid state');

    try {
      const tensors = metadata.tensors || [];
      let currentOffset = 0;

      for (const tensor of tensors) {
        const size = tensor.ne.reduce((a: number, b: number) => a * b, 1);
        this.rangeMap.set(tensor.name, {
          start: currentOffset,
          size,
        });
        currentOffset += size;
      }

      this.transition('SPARSE_CALC');
    } catch (e) {
      this.lastError = e as Error;
      this.transition('ERROR');
      throw e;
    }
  }

  /**
   * Compute sparse activation geometry using Row-Column Bundling.
   */
  calculateSparseGeometry(predictor: any): void {
    if (this.state !== 'SPARSE_CALC') throw new Error('Invalid state');

    try {
      // Allocate geometry buffer
      this.sparseGeometry = new Float32Array(1024);

      // Use predictor to compute dense->sparse token paths
      const geometry = predictor.predict();
      this.sparseGeometry.set(geometry);

      this.transition('EXECUTING');
    } catch (e) {
      this.lastError = e as Error;
      this.transition('ERROR');
      throw e;
    }
  }

  /**
   * Arm execution: ready to receive compute tasks.
   */
  armExecution(): void {
    if (this.state !== 'EXECUTING') throw new Error('Invalid state');
    this.transition('READY');
  }

  /**
   * Flush all active buffers back to storage (end of inference).
   */
  flush(): void {
    if (this.state !== 'READY') return;

    this.transition('FLUSHING');
    // Zero out buffers
    new Uint8Array(this.executionContext!.bufferA).fill(0);
    new Uint8Array(this.executionContext!.bufferB).fill(0);
    this.transition('BOOTSTRAPPING');
  }

  getState(): LifecycleState {
    return this.state;
  }

  getContext(): ExecutionContext | null {
    return this.executionContext;
  }

  getRangeMap(): Map<number, { start: number; size: number }> {
    return this.rangeMap;
  }

  getSparseGeometry(): Float32Array | null {
    return this.sparseGeometry;
  }

  getError(): Error | null {
    return this.lastError;
  }

  getTimeSinceStateChange(): number {
    return performance.now() - this.stateStartTime;
  }
}

export default STSLifecycleStateMachine;
