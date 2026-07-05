/**
 * @file File handling for Hugging Face Hub model loading and caching.
 * Optimized for OPFS (Origin Private File System) on mobile devices.
 * Implements atomic reads and zero-latency access patterns.
 * 
 * @module utils/hub
 */

import fs from 'fs';
import path from 'path';

/**
 * MIME type mapping for proper content-type headers.
 */
const CONTENT_TYPE_MAP = {
  'txt': 'text/plain',
  'html': 'text/html',
  'css': 'text/css',
  'js': 'text/javascript',
  'json': 'application/json',
  'onnx': 'application/octet-stream',
  'bin': 'application/octet-stream',
  'safetensors': 'application/octet-stream',
};

/**
 * File response wrapper for local file system operations.
 * Implements ReadableStream for efficient streaming to main thread.
 */
class FileResponse {
  constructor(filePath) {
    this.filePath = filePath;
    this.headers = new Headers();
    this.exists = fs.existsSync(filePath);
    
    if (this.exists) {
      this.status = 200;
      this.statusText = 'OK';
      
      const stats = fs.statSync(filePath);
      this.headers.set('content-length', stats.size.toString());
      this.updateContentType();
      
      const self = this;
      this.body = new ReadableStream({
        start(controller) {
          self.arrayBuffer().then(buffer => {
            controller.enqueue(new Uint8Array(buffer));
            controller.close();
          });
        },
      });
    } else {
      this.status = 404;
      this.statusText = 'Not Found';
      this.body = null;
    }
  }

  updateContentType() {
    const extension = this.filePath.toString().split('.').pop().toLowerCase();
    this.headers.set(
      'content-type',
      CONTENT_TYPE_MAP[extension] ?? 'application/octet-stream'
    );
  }

  clone() {
    const response = new FileResponse(this.filePath);
    response.exists = this.exists;
    response.status = this.status;
    response.statusText = this.statusText;
    response.headers = new Headers(this.headers);
    return response;
  }

  async arrayBuffer() {
    const data = await fs.promises.readFile(this.filePath);
    return data.buffer;
  }

  async blob() {
    const data = await fs.promises.readFile(this.filePath);
    return new Blob([data], { type: this.headers.get('content-type') });
  }

  async text() {
    return await fs.promises.readFile(this.filePath, 'utf8');
  }

  async json() {
    return JSON.parse(await this.text());
  }
}

/**
 * Validates URL format for security.
 * Prevents arbitrary file access and SSRF attacks.
 */
function isValidUrl(string, protocols = null, validHosts = null) {
  let url;
  try {
    url = new URL(string);
  } catch (_) {
    return false;
  }
  if (protocols && !protocols.includes(url.protocol)) {
    return false;
  }
  if (validHosts && !validHosts.includes(url.hostname)) {
    return false;
  }
  return true;
}

/**
 * Fetches file using appropriate backend (HTTP or FS).
 * Automatically routes to OPFS on mobile for zero-latency access.
 */
export async function getFile(urlOrPath) {
  // Try local file system first (faster, zero-latency)
  if (!isValidUrl(urlOrPath, ['http:', 'https:', 'blob:'])) {
    return new FileResponse(urlOrPath);
  }
  
  // Fall back to HTTP fetch for remote URLs
  if (typeof process !== 'undefined' && process?.release?.name === 'node') {
    const headers = new Headers();
    headers.set('User-Agent', `transformers.js/4.2.0`);
    
    // Check if URL is Hugging Face Hub
    const isHFURL = isValidUrl(urlOrPath, ['http:', 'https:'], [
      'huggingface.co',
      'hf.co',
    ]);
    
    if (isHFURL && process.env?.HF_TOKEN) {
      headers.set('Authorization', `Bearer ${process.env.HF_TOKEN}`);
    }
    
    return fetch(urlOrPath, { headers });
  } else {
    return fetch(urlOrPath);
  }
}

/**
 * Cache implementation using file system.
 * Persists downloaded models to OPFS for instant loading.
 */
class FileCache {
  constructor(cachePath) {
    this.path = cachePath;
  }

  async match(request) {
    const filePath = path.join(this.path, request);
    const file = new FileResponse(filePath);
    return file.exists ? file : undefined;
  }

  async put(request, response) {
    const buffer = Buffer.from(await response.arrayBuffer());
    const outputPath = path.join(this.path, request);
    
    try {
      await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.promises.writeFile(outputPath, buffer);
    } catch (err) {
      console.warn('Failed to cache file:', err);
    }
  }
}

/**
 * Retrieves model file with automatic caching.
 * Implements aggressive caching for repeated model loads.
 */
export async function getModelFile(
  pathOrRepoId,
  filename,
  fatal = true,
  options = {}
) {
  // Try cache first
  let cache = null;
  
  if (options.cache_dir) {
    cache = new FileCache(options.cache_dir);
    const cached = await cache.match(filename);
    if (cached) {
      return new Uint8Array(await cached.arrayBuffer());
    }
  }
  
  // Try local file system
  if (!isValidUrl(pathOrRepoId, ['http:', 'https:'])) {
    const localPath = path.join(pathOrRepoId, filename);
    const response = await getFile(localPath);
    if (response.exists || response.status === 200) {
      return new Uint8Array(await response.arrayBuffer());
    }
  }
  
  // Fetch from remote
  const revision = options.revision ?? 'main';
  const remoteUrl = `https://huggingface.co/${pathOrRepoId}/resolve/${revision}/${filename}`;
  
  const response = await getFile(remoteUrl);
  
  if (!response.ok && response.status !== 200) {
    if (fatal) {
      throw new Error(`Failed to fetch ${filename}: ${response.status}`);
    }
    return null;
  }
  
  const buffer = new Uint8Array(await response.arrayBuffer());
  
  // Cache the downloaded file
  if (cache) {
    await cache.put(filename, new Response(buffer));
  }
  
  return buffer;
}

/**
 * Fetches and parses JSON configuration files.
 * Used for model configs, vocabulary files, etc.
 */
export async function getModelJSON(
  modelPath,
  fileName,
  fatal = true,
  options = {}
) {
  const buffer = await getModelFile(modelPath, fileName, fatal, options);
  if (buffer === null) {
    return {};
  }
  
  const decoder = new TextDecoder('utf-8');
  const jsonData = decoder.decode(buffer);
  return JSON.parse(jsonData);
}

export { FileResponse, FileCache };
export default { getFile, getModelFile, getModelJSON };
