/**
 * @file A.S.T.S. Web UI Application
 * Orchestrates hardware detection, model loading, and inference
 */

const LOG_BUFFER = [];
const MAX_LOGS = 100;

// ============================================
// UTILITY FUNCTIONS
// ============================================

function log(message, type = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = `[${timestamp}] ${message}`;
  console.log(logEntry);

  LOG_BUFFER.push({ message: logEntry, type });
  if (LOG_BUFFER.length > MAX_LOGS) LOG_BUFFER.shift();

  updateLogDisplay();
}

function updateLogDisplay() {
  const logOutput = document.getElementById('logOutput');
  logOutput.innerHTML = LOG_BUFFER.map(
    (entry) =>
      `<div class="log-line ${entry.type}">${entry.message}</div>`
  ).join('');
  logOutput.scrollTop = logOutput.scrollHeight;
}

function updateStat(elementId, value) {
  const element = document.getElementById(elementId);
  if (element) element.textContent = value;
}

// ============================================
// HARDWARE DETECTION
// ============================================

async function initHardware() {
  log('Initializing hardware detection...', 'info');

  try {
    let primaryBackend = 'cpu';
    const capabilities = {
      hasWebGPU: false,
      hasWebNN: false,
      isMobile: /Android|iPhone|iPad/.test(navigator.userAgent),
    };

    // Check WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          capabilities.hasWebGPU = true;
          primaryBackend = 'webgpu';
          log('✓ WebGPU available', 'info');
        }
      } catch (e) {
        log('✗ WebGPU not available', 'warning');
      }
    }

    // Check WebNN (Android)
    if (navigator.ml) {
      capabilities.hasWebNN = true;
      primaryBackend = 'webnn';
      log('✓ WebNN available', 'info');
    }

    // Check SharedArrayBuffer
    const hasSAB = typeof SharedArrayBuffer !== 'undefined';
    log(`SharedArrayBuffer: ${hasSAB ? '✓ Enabled' : '✗ Disabled'}`, 'info');

    updateStat('hardwareTarget', primaryBackend.toUpperCase());
    updateStat(
      'streamingMode',
      capabilities.isMobile ? 'MOBILE (HTTP Range)' : 'DESKTOP (OPFS)'
    );

    return { primaryBackend, capabilities };
  } catch (e) {
    log(`Hardware detection failed: ${e.message}`, 'error');
    updateStat('hardwareTarget', 'ERROR');
  }
}

// ============================================
// MODEL MANAGEMENT
// ============================================

let currentModel = null;

document.getElementById('loadModelBtn').addEventListener('click', async () => {
  const select = document.getElementById('modelSelect');
  const modelId = select.value;

  if (!modelId) {
    log('Please select a model', 'warning');
    return;
  }

  const loadingDiv = document.getElementById('loadingIndicator');
  const loadingText = document.getElementById('loadingText');
  loadingDiv.style.display = 'block';
  loadingText.textContent = `Loading ${modelId}...`;

  try {
    log(`Loading model: ${modelId}`, 'info');

    // Simulate model loading (real implementation would use GGUF streaming)
    await new Promise((resolve) => setTimeout(resolve, 2000));

    currentModel = { id: modelId, loaded: true };
    log(`✓ Model loaded: ${modelId}`, 'info');
    updateStat('bufferStatus', 'Ready');
    document.getElementById('generateBtn').disabled = false;
  } catch (e) {
    log(`Model loading failed: ${e.message}`, 'error');
  } finally {
    loadingDiv.style.display = 'none';
  }
});

// ============================================
// INFERENCE
// ============================================

document.getElementById('generateBtn').addEventListener('click', async () => {
  if (!currentModel) {
    log('No model loaded', 'warning');
    return;
  }

  const prompt = document.getElementById('promptInput').value;
  const maxTokens = parseInt(document.getElementById('maxTokens').value);
  const temperature = parseFloat(document.getElementById('temperature').value);

  if (!prompt) {
    log('Please enter a prompt', 'warning');
    return;
  }

  const generateBtn = document.getElementById('generateBtn');
  generateBtn.disabled = true;

  try {
    log(`Generating with model: ${currentModel.id}`, 'info');
    log(`Prompt: ${prompt.substring(0, 100)}...`, 'info');

    // Simulate inference
    const outputDiv = document.getElementById('outputText');
    let output = '';
    let tokensGenerated = 0;
    const startTime = performance.now();

    for (let i = 0; i < Math.min(maxTokens, 50); i++) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      output += 'token ';
      tokensGenerated++;
      outputDiv.textContent = prompt + '\n\n' + output;
      outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    const endTime = performance.now();
    const elapsed = (endTime - startTime) / 1000;
    const tokensPerSec = tokensGenerated / elapsed;

    log(`✓ Generation complete: ${tokensGenerated} tokens in ${elapsed.toFixed(2)}s`, 'info');
    updateStat('tokensPerSec', tokensPerSec.toFixed(2));
  } catch (e) {
    log(`Inference failed: ${e.message}`, 'error');
  } finally {
    generateBtn.disabled = false;
  }
});

// ============================================
// TELEMETRY
// ============================================

function updateTelemetry() {
  // Simulate telemetry updates
  updateStat('gpuUtil', Math.random() * 100 | 0 + '%');
  updateStat('memoryUsage', (Math.random() * 1000 | 0) + ' MB');
  updateStat('prefillTime', (Math.random() * 500 | 0) + ' ms');
  updateStat('decodeTime', (Math.random() * 100 | 0) + ' ms');
}

setInterval(updateTelemetry, 1000);

// ============================================
// LOGS
// ============================================

document.getElementById('clearLogsBtn').addEventListener('click', () => {
  LOG_BUFFER.length = 0;
  updateLogDisplay();
  log('Logs cleared', 'info');
});

// ============================================
// INITIALIZATION
// ============================================

window.addEventListener('load', async () => {
  log('A.S.T.S. Terminal starting...', 'info');
  await initHardware();
  log('Ready for model loading', 'info');
});
