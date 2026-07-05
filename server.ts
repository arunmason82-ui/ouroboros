/**
 * @file Express.js server: HTTP interface for A.S.T.S. architecture
 * Serves web assets, provides API endpoints, enforces COOP/COEP headers
 * for SharedArrayBuffer support.
 * @module server
 */

import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8080;

// ============================================
// SECURITY HEADERS FOR SharedArrayBuffer
// ============================================
app.use((req: Request, res: Response, next) => {
  // Cross-Origin-Opener-Policy: same-origin (required for SAB)
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  // Cross-Origin-Embedder-Policy: require-corp (required for SAB)
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  // Allow our own origin
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
  next();
});

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ============================================
// API ENDPOINTS
// ============================================

/**
 * Health check endpoint
 */
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

/**
 * Hardware audit endpoint
 */
app.get('/api/audit', (req: Request, res: Response) => {
  res.json({
    message: 'Hardware audit should be run client-side using HardwareAuditor',
    hint: 'Make request from browser with navigator.gpu access',
  });
});

/**
 * Model metadata endpoint
 */
app.post('/api/model/metadata', express.json(), (req: Request, res: Response) => {
  const { modelPath } = req.body;
  if (!modelPath) {
    return res.status(400).json({ error: 'modelPath required' });
  }
  res.json({
    message: 'Metadata fetching delegated to GGUF parser worker',
    modelPath,
  });
});

/**
 * Inference endpoint
 */
app.post('/api/infer', express.json(), (req: Request, res: Response) => {
  const { prompt, modelId, options } = req.body;

  if (!prompt || !modelId) {
    return res.status(400).json({
      error: 'prompt and modelId are required',
    });
  }

  res.json({
    message: 'Inference delegated to worker pool',
    prompt,
    modelId,
    options,
    hint: 'Check browser console for real-time telemetry',
  });
});

/**
 * System info endpoint
 */
app.get('/api/system', (req: Request, res: Response) => {
  res.json({
    node_version: process.version,
    platform: process.platform,
    arch: process.arch,
    memory: process.memoryUsage(),
    uptime: process.uptime(),
  });
});

// ============================================
// STATIC FILE SERVING
// ============================================

app.get('/', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ============================================
// 404 HANDLER
// ============================================

app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: 'Not Found',
    path: req.path,
  });
});

// ============================================
// START SERVER
// ============================================

app.listen(PORT, () => {
  console.log(`[Server] A.S.T.S. listening on http://localhost:${PORT}`);
  console.log(`[COOP] Cross-Origin-Opener-Policy: same-origin`);
  console.log(`[COEP] Cross-Origin-Embedder-Policy: require-corp`);
  console.log(`[SAB] SharedArrayBuffer: ENABLED`);
});

export default app;
