/**
 * @file Comprehensive A.S.T.S. Test Suite
 * Unit tests, integration tests, and performance benchmarks
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import HardwareAuditor from './ouroboros-core/hw/auditor';
import TopologyParser from './ouroboros-core/asts/topologyParser';
import WeightSynthesizer from './ouroboros-core/asts/weightSynthesizer';
import STSLifecycleStateMachine from './ouroboros-core/kernel/stateMachine';
import LocklessScheduler from './ouroboros-core/kernel/scheduler';

// ============================================
// HARDWARE AUDITOR TESTS
// ============================================

describe('HardwareAuditor', () => {
  let auditor: HardwareAuditor;

  beforeAll(() => {
    auditor = new HardwareAuditor();
  });

  it('should detect available hardware backends', async () => {
    const result = await auditor.audit();
    expect(result).toHaveProperty('primaryBackend');
    expect(['webgpu', 'webnn', 'wasm', 'cpu']).toContain(
      result.primaryBackend
    );
  });

  it('should complete audit within 200ms timeout', async () => {
    const start = performance.now();
    await auditor.audit();
    const duration = performance.now() - start;
    expect(duration).toBeLessThan(200);
  });

  it('should detect mobile devices correctly', async () => {
    const result = await auditor.audit();
    expect(result.capabilities).toHaveProperty('isMobile');
  });
});

// ============================================
// TOPOLOGY PARSER TESTS
// ============================================

describe('TopologyParser', () => {
  let parser: TopologyParser;
  const mockMetadata = {
    tensors: [
      { name: 'embedding', dtype: 0, ne: [32000, 4096] },
      { name: 'layer_0', dtype: 1, ne: [4096, 4096] },
    ],
  };

  beforeAll(() => {
    parser = new TopologyParser();
  });

  it('should parse GGUF metadata correctly', () => {
    const topology = parser.parse(mockMetadata);
    expect(topology.tensors.length).toBe(2);
    expect(topology.offsets).toHaveProperty('embedding');
  });

  it('should calculate correct byte offsets', () => {
    parser.parse(mockMetadata);
    const range = parser.getRange('embedding');
    expect(range).not.toBeNull();
    expect(range![1]).toBeGreaterThan(0n);
  });

  it('should bundle sparse ranges correctly', () => {
    parser.parse(mockMetadata);
    const bundled = parser.getBundledRanges([0, 1]);
    expect(bundled.length).toBeGreaterThan(0);
  });
});

// ============================================
// WEIGHT SYNTHESIZER TESTS
// ============================================

describe('WeightSynthesizer', () => {
  let synthesizer: WeightSynthesizer;
  let sharedBuffer: SharedArrayBuffer;

  beforeAll(() => {
    sharedBuffer = new SharedArrayBuffer(10 * 1024 * 1024); // 10MB
    synthesizer = new WeightSynthesizer(sharedBuffer, {
      format: 'q4_0',
      blockSize: 32,
      hasLoRA: false,
    });
  });

  it('should initialize with valid configuration', () => {
    expect(synthesizer).toBeDefined();
  });

  it('should dequantize Q4_0 format', () => {
    // Create mock Q4_0 data
    const input = new Uint8Array(sharedBuffer, 0, 64);
    input.fill(0x88); // Dummy data

    expect(() => {
      synthesizer.dequantizeQ4_0(0, 1024, 512);
    }).not.toThrow();
  });

  it('should dequantize Q8_0 format', () => {
    const input = new Int8Array(sharedBuffer, 2048, 128);
    input.fill(50);

    expect(() => {
      synthesizer.dequantizeQ8_0(2048, 3072, 512);
    }).not.toThrow();
  });

  it('should apply LoRA corrections', () => {
    expect(() => {
      synthesizer.applyLoRA(
        0,      // baseWeightsOffset
        1024,   // loraAOffset
        2048,   // loraBOffset
        256,    // m (rows)
        256,    // n (cols)
        8,      // r (rank)
        1.0     // scale
      );
    }).not.toThrow();
  });
});

// ============================================
// STATE MACHINE TESTS
// ============================================

describe('STSLifecycleStateMachine', () => {
  let stateMachine: STSLifecycleStateMachine;

  beforeAll(() => {
    stateMachine = new STSLifecycleStateMachine();
  });

  it('should start in BOOTSTRAPPING state', () => {
    expect(stateMachine.getState()).toBe('BOOTSTRAPPING');
  });

  it('should transition through states', () => {
    stateMachine.transition('AUDITING');
    expect(stateMachine.getState()).toBe('AUDITING');

    stateMachine.transition('RANGE_MAPPING');
    expect(stateMachine.getState()).toBe('RANGE_MAPPING');
  });

  it('should allocate execution context on bootstrap', () => {
    const newSM = new STSLifecycleStateMachine();
    newSM.bootstrap();
    const ctx = newSM.getContext();
    expect(ctx).not.toBeNull();
    expect(ctx!.bufferA).toBeDefined();
    expect(ctx!.bufferB).toBeDefined();
  });

  it('should track state transition time', () => {
    const before = stateMachine.getTimeSinceStateChange();
    expect(before).toBeGreaterThanOrEqual(0);
  });
});

// ============================================
// SCHEDULER TESTS
// ============================================

describe('LocklessScheduler', () => {
  let scheduler: LocklessScheduler;

  beforeAll(() => {
    scheduler = new LocklessScheduler({
      bufferSizePerLayer: 16 * 1024 * 1024,
      computeTimeoutMs: 5000,
      maxQueuedTasks: 10,
    });
  });

  it('should return active buffer', () => {
    const buffer = scheduler.getActiveBuffer();
    expect(buffer).toBeInstanceOf(SharedArrayBuffer);
  });

  it('should return inactive buffer (different from active)', () => {
    const active = scheduler.getActiveBuffer();
    const inactive = scheduler.getInactiveBuffer();
    expect(active).not.toBe(inactive);
  });

  it('should provide scheduler stats', () => {
    const stats = scheduler.getStats();
    expect(stats).toHaveProperty('queuedTasks');
    expect(stats).toHaveProperty('isProcessing');
    expect(stats).toHaveProperty('activeBufferIndex');
  });
});

// ============================================
// INTEGRATION TESTS
// ============================================

describe('A.S.T.S. Integration', () => {
  it('should initialize full pipeline', async () => {
    const auditor = new HardwareAuditor();
    const result = await auditor.audit();
    expect(result.primaryBackend).toBeDefined();
  });

  it('should parse and process model metadata', () => {
    const parser = new TopologyParser();
    const metadata = {
      tensors: [
        { name: 'test', dtype: 0, ne: [1024, 1024] },
      ],
    };
    const topology = parser.parse(metadata);
    expect(topology.tensors.length).toBe(1);
  });
});

// ============================================
// PERFORMANCE BENCHMARKS
// ============================================

describe('Performance Benchmarks', () => {
  it('should parse topology in <10ms', () => {
    const parser = new TopologyParser();
    const start = performance.now();
    parser.parse({
      tensors: Array.from({ length: 100 }, (_, i) => ({
        name: `layer_${i}`,
        dtype: i % 2,
        ne: [4096, 4096],
      })),
    });
    const duration = performance.now() - start;
    expect(duration).toBeLessThan(10);
  });

  it('should dequantize 1MB of Q4_0 data in <50ms', () => {
    const buffer = new SharedArrayBuffer(1024 * 1024);
    const synthesizer = new WeightSynthesizer(buffer, {
      format: 'q4_0',
      blockSize: 32,
      hasLoRA: false,
    });

    const start = performance.now();
    synthesizer.dequantizeQ4_0(0, 512 * 1024, 262144);
    const duration = performance.now() - start;
    expect(duration).toBeLessThan(50);
  });
});
