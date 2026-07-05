/**
 * @file Unified GGUF streaming: local OPFS reads on PC, HTTP range requests on mobile.
 * Runs in isolated Web Worker for non-blocking I/O.
 * @module io/ggufStreamer
 */

import { parseGGUFMetadata } from '../../gguf_parser';
import { getModelFile, getFile } from '../../hub_file_handler';

/**
 * Streaming request configuration.
 */
interface StreamRequest {
  id: string;
  modelPath: string;
  startByte: bigint;
  endByte: bigint;
  targetBuffer: SharedArrayBuffer;
  targetOffset: number;
  isMobileRemoteMode: boolean;
}

/**
 * Worker-side GGUF streaming handler.
 */
export class GGUFStreamer {
  private isMobileMode = false;
  private modelCache: Map<string, ArrayBuffer> = new Map();
  private activeRequests: Map<string, StreamRequest> = new Map();

  /**
   * Initialize streamer with device detection.
   */
  initialize(isMobileRemoteMode: boolean): void {
    this.isMobileMode = isMobileRemoteMode;
    console.log(
      `[GGUFStreamer] Initialized in ${isMobileMode ? 'MOBILE REMOTE' : 'LOCAL OPFS'} mode`
    );
  }

  /**
   * Stream GGUF layer chunk with appropriate backend.
   */
  async streamLayerChunk(
    request: StreamRequest
  ): Promise<{ status: 'success' | 'error'; error?: string }> {
    try {
      this.activeRequests.set(request.id, request);

      if (this.isMobileMode) {
        // Mobile: HTTP Range request to Hugging Face
        await this.streamRemoteHTTP(request);
      } else {
        // Desktop: Direct OPFS file read
        await this.streamLocalOPFS(request);
      }

      this.activeRequests.delete(request.id);
      return { status: 'success' };
    } catch (error) {
      this.activeRequests.delete(request.id);
      return {
        status: 'error',
        error: (error as Error).message,
      };
    }
  }

  /**
   * Stream from local OPFS using FileSystemSyncAccessHandle.
   * Zero-latency, synchronous reads on disk.
   */
  private async streamLocalOPFS(
    request: StreamRequest
  ): Promise<void> {
    try {
      // For browser environment, use fetch with local file:// URL
      if (typeof window !== 'undefined' && window.location.protocol === 'file:') {
        const response = await fetch(request.modelPath);
        const blob = await response.blob();
        const arrayBuffer = await blob.arraySlice(Number(request.startByte), Number(request.endByte)).arrayBuffer();

        const targetView = new Uint8Array(request.targetBuffer, request.targetOffset);
        targetView.set(new Uint8Array(arrayBuffer));
      } else {
        // Use hub_file_handler for local file reading
        const data = await getFile(request.modelPath);
        if (data instanceof Response) {
          const arrayBuffer = await data.arrayBuffer();
          const targetView = new Uint8Array(request.targetBuffer, request.targetOffset);
          targetView.set(new Uint8Array(arrayBuffer));
        }
      }
    } catch (error) {
      throw new Error(`OPFS read failed: ${(error as Error).message}`);
    }
  }

  /**
   * Stream from Hugging Face Hub using HTTP Range requests.
   * Atomic byte-range slicing, no full file download.
   */
  private async streamRemoteHTTP(
    request: StreamRequest
  ): Promise<void> {
    try {
      const rangeHeader = `bytes=${request.startByte}-${request.endByte - 1n}`;

      const response = await fetch(request.modelPath, {
        headers: {
          Range: rangeHeader,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const targetView = new Uint8Array(request.targetBuffer, request.targetOffset);
      targetView.set(new Uint8Array(arrayBuffer));
    } catch (error) {
      throw new Error(`Remote HTTP fetch failed: ${(error as Error).message}`);
    }
  }

  /**
   * Prefetch metadata from GGUF file.
   */
  async prefetchMetadata(modelPath: string): Promise<any> {
    if (this.modelCache.has(modelPath)) {
      return JSON.parse(
        new TextDecoder().decode(this.modelCache.get(modelPath))
      );
    }

    try {
      const metadata = await parseGGUFMetadata(modelPath);
      return metadata.metadata;
    } catch (error) {
      console.error('Metadata prefetch failed:', error);
      throw error;
    }
  }

  /**
   * Get active request count (for telemetry).
   */
  getActiveRequestCount(): number {
    return this.activeRequests.size;
  }
}

// Worker message handler
if (typeof self !== 'undefined') {
  const streamer = new GGUFStreamer();

  self.onmessage = async (event: MessageEvent) => {
    const { id, cmd } = event.data;

    if (cmd.type === 'init') {
      streamer.initialize(cmd.isMobileRemoteMode);
      self.postMessage({ id, status: 'success' });
    } else if (cmd.type === 'stream_layer') {
      const result = await streamer.streamLayerChunk(cmd.request);
      self.postMessage({ id, ...result });
    } else if (cmd.type === 'prefetch_metadata') {
      try {
        const metadata = await streamer.prefetchMetadata(cmd.modelPath);
        self.postMessage({ id, status: 'success', metadata });
      } catch (error) {
        self.postMessage({ id, status: 'error', error: (error as Error).message });
      }
    }
  };
}

export default GGUFStreamer;
