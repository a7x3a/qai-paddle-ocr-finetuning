"""Lightweight, high-performance web interface for Kurdish PaddleOCR inference.

Provides a zero-dependency, local web UI server for interactive image and full-page recognition
with drag-and-drop, clipboard paste, visual bounding-box overlays, and real-time latency readouts.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import tempfile
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.pipeline.document_reader import DocumentReader, DocumentResult
from src.utils.logger import setup_logger

logger = setup_logger("WebServer")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ku" dir="ltr" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kurdish PaddleOCR - Full Document & Page Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Vazirmatn:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
            kurdish: ['"Vazirmatn"', 'system-ui', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
          },
          colors: {
            brand: {
              50: '#ecfeff',
              100: '#cffafe',
              400: '#22d3ee',
              500: '#06b6d4',
              600: '#0891b2',
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #080c14;
      color: #f1f5f9;
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    }
    .kurdish-text {
      font-family: 'Vazirmatn', system-ui, sans-serif;
      direction: rtl;
    }
    .radial-bg {
      background: radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.12) 0%, rgba(8, 12, 20, 0) 70%);
    }
    .glass-card {
      background: rgba(15, 23, 42, 0.65);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-card:hover {
      border-color: rgba(6, 182, 212, 0.3);
    }
    .dropzone-border {
      background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='16' ry='16' stroke='%23334155' stroke-width='2' stroke-dasharray='8%2c 8' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
    }
    .dropzone-active {
      background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='16' ry='16' stroke='%2306b6d4' stroke-width='2' stroke-dasharray='8%2c 8' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
      background-color: rgba(6, 182, 212, 0.05);
    }
    pre, textarea {
      font-family: 'Vazirmatn', system-ui, sans-serif;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col radial-bg antialiased selection:bg-brand-500 selection:text-black">

  <!-- Header -->
  <header class="border-b border-slate-800/80 sticky top-0 z-50 backdrop-blur-md bg-slate-950/70">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20 text-lg">
          ق
        </div>
        <div>
          <h1 class="font-bold text-base sm:text-lg tracking-tight flex items-center gap-2">
            Kurdish PaddleOCR
            <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-medium">Page & Document OCR</span>
          </h1>
          <p class="text-xs text-slate-400">Detection + Fine-Tuned Kurdish Recognition</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span class="font-mono">DBNet + Rec Active</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 flex flex-col gap-8">

    <!-- Hero / Headline -->
    <div class="flex flex-col gap-2">
      <h2 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Document & Page Text Reader</h2>
      <p class="text-slate-400 text-sm max-w-2xl">
        Upload scanned pages, documents, book pages, receipts, or single text crops. The system automatically detects every text line across the entire page, sorts in Kurdish RTL reading order, and extracts the full text.
      </p>
    </div>

    <!-- Quick Test Samples -->
    <div class="flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <svg class="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Quick Test Samples (Click to Run)
        </span>
      </div>
      <div id="sample-chips" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
        <!-- Sample chips injected by JS -->
      </div>
    </div>

    <!-- Dual Workspace: Upload & Result -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

      <!-- Left Column: Input & Visual Detection View -->
      <div class="lg:col-span-6 flex flex-col gap-4">
        <div class="glass-card rounded-2xl p-6 flex flex-col gap-4">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <svg class="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Document / Page Input
            </h3>
            <span class="text-xs text-slate-500 font-mono">PNG, JPG, PDF-scan</span>
          </div>

          <!-- Drag & Drop Zone -->
          <div id="dropzone" class="dropzone-border rounded-2xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200 group">
            <input type="file" id="file-input" class="hidden" accept="image/*">
            <div class="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center border border-slate-800 text-slate-400 group-hover:text-cyan-400 group-hover:border-cyan-500/40 transition">
              <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div class="text-center flex flex-col gap-1">
              <p class="text-sm font-medium text-slate-200">
                <span class="text-cyan-400 font-semibold underline decoration-cyan-500/40 underline-offset-4">Click to upload</span> or drag a page/image
              </p>
              <p class="text-xs text-slate-500">or press <kbd class="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Ctrl+V</kbd> to paste from clipboard</p>
            </div>
          </div>

          <!-- Image Preview with Toggle for Detection Boxes -->
          <div id="preview-wrapper" class="hidden flex flex-col gap-3 pt-2">
            <div class="flex items-center justify-between text-xs text-slate-400">
              <div class="flex items-center gap-2">
                <button id="toggle-annotated-btn" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 font-medium transition text-[11px] flex items-center gap-1">
                  <span>Show Bounding Boxes</span>
                </button>
              </div>
              <button id="clear-btn" class="hover:text-rose-400 transition text-[11px]">Clear image</button>
            </div>
            <div class="w-full bg-slate-950/80 rounded-xl p-2 flex items-center justify-center border border-slate-800 min-h-[160px] max-h-[380px] overflow-auto">
              <img id="preview-img" src="" alt="Preview" class="max-h-[360px] max-w-full object-contain rounded">
            </div>
            <div id="img-meta" class="text-[11px] font-mono text-slate-500 flex justify-between"></div>
          </div>

        </div>
      </div>

      <!-- Right Column: Extracted Document Content -->
      <div class="lg:col-span-6 flex flex-col gap-4">
        <div class="glass-card rounded-2xl p-6 flex flex-col gap-5 relative overflow-hidden">

          <div class="flex items-center justify-between border-b border-slate-800 pb-4">
            <div class="flex items-center gap-2">
              <h3 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <svg class="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Extracted Document Text
              </h3>
              <span id="lines-count-badge" class="hidden text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                0 lines
              </span>
            </div>
            <div id="latency-badge" class="hidden px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono text-cyan-400">
              0.0 ms
            </div>
          </div>

          <!-- Empty State -->
          <div id="empty-state" class="py-20 flex flex-col items-center justify-center gap-3 text-center text-slate-500">
            <div class="w-12 h-12 rounded-full bg-slate-900/60 flex items-center justify-center border border-slate-800 text-slate-600">
              <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <p class="text-sm font-medium">Ready to read your document</p>
            <p class="text-xs text-slate-600 max-w-xs">Upload a page, receipt, or test sample to extract all text lines.</p>
          </div>

          <!-- Loading State -->
          <div id="loading-state" class="hidden py-20 flex flex-col items-center justify-center gap-3 text-center">
            <div class="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <p class="text-xs font-mono text-cyan-400 tracking-wider uppercase">Detecting text regions & recognizing Kurdish...</p>
          </div>

          <!-- Result Display -->
          <div id="result-state" class="hidden flex flex-col gap-4">

            <!-- Full Document Text Area with Copy -->
            <div class="flex flex-col gap-2">
              <div class="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>FULL EXTRACTED TEXT (RTL)</span>
                <div class="flex items-center gap-2">
                  <button id="copy-btn" class="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 transition text-[11px]">
                    <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <span id="copy-label">Copy All</span>
                  </button>
                  <button id="download-btn" class="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition text-[11px]">
                    <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <span>.txt</span>
                  </button>
                </div>
              </div>
              <div class="w-full rounded-xl bg-slate-950 p-4 border border-slate-800 text-right">
                <textarea id="full-text-area" rows="6" readonly class="w-full bg-transparent border-0 resize-y text-xl sm:text-2xl text-white font-medium kurdish-text focus:outline-none leading-relaxed selection:bg-cyan-500 selection:text-black"></textarea>
              </div>
            </div>

            <!-- Line-by-Line Breakdown Accordion/List -->
            <div class="flex flex-col gap-2 pt-2">
              <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">Detected Lines Breakdown</span>
              <div id="lines-list" class="flex flex-col gap-2 max-h-[260px] overflow-y-auto pr-1">
                <!-- Lines injected here -->
              </div>
            </div>

          </div>

        </div>
      </div>

    </div>

  </main>

  <footer class="border-t border-slate-800/80 py-4 text-center text-xs text-slate-500 font-mono">
    Kurdish PaddleOCR Fine-Tuning Pipeline &bull; Full-Page DBNet Detection + Rec &bull; Accelerated on NVIDIA CUDA
  </footer>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewWrapper = document.getElementById('preview-wrapper');
    const previewImg = document.getElementById('preview-img');
    const imgMeta = document.getElementById('img-meta');
    const clearBtn = document.getElementById('clear-btn');
    const toggleAnnotatedBtn = document.getElementById('toggle-annotated-btn');

    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    const fullTextArea = document.getElementById('full-text-area');
    const linesList = document.getElementById('lines-list');
    const linesCountBadge = document.getElementById('lines-count-badge');
    const latencyBadge = document.getElementById('latency-badge');
    const copyBtn = document.getElementById('copy-btn');
    const copyLabel = document.getElementById('copy-label');
    const downloadBtn = document.getElementById('download-btn');
    const sampleChips = document.getElementById('sample-chips');

    let originalImgUrl = null;
    let annotatedImgUrl = null;
    let showingAnnotated = true;

    // Load sample chips
    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        sampleChips.innerHTML = '';
        data.forEach(s => {
          const btn = document.createElement('button');
          btn.className = "flex flex-col items-center gap-1.5 p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 transition text-left group";
          btn.innerHTML = `
            <div class="w-full h-10 bg-slate-950 rounded flex items-center justify-center overflow-hidden border border-slate-800">
              <img src="/api/sample_image?name=${encodeURIComponent(s.filename)}" class="max-h-8 max-w-full object-contain" alt="sample">
            </div>
            <div class="w-full text-center">
              <span class="kurdish-text text-xs font-semibold text-slate-200 group-hover:text-cyan-400 truncate block">${s.label}</span>
            </div>
          `;
          btn.addEventListener('click', () => {
            selectSample(s);
          });
          sampleChips.appendChild(btn);
        });
      } catch (err) {
        console.error('Failed to load samples:', err);
      }
    }

    async function selectSample(sample) {
      setLoading(true);
      const imgUrl = `/api/sample_image?name=${encodeURIComponent(sample.filename)}`;
      originalImgUrl = imgUrl;
      annotatedImgUrl = null;
      previewImg.src = imgUrl;
      previewWrapper.classList.remove('hidden');
      imgMeta.textContent = sample.filename;

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_path: sample.filename })
        });
        const result = await res.json();
        showResult(result);
      } catch (err) {
        alert('OCR error: ' + err.message);
      } finally {
        setLoading(false);
      }
    }

    // Dropzone Events
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dropzone-active');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dropzone-active');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dropzone-active');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });

    // Paste from clipboard
    window.addEventListener('paste', (e) => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          const blob = item.getAsFile();
          handleFile(blob);
          break;
        }
      }
    });

    function handleFile(file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64Data = e.target.result;
        originalImgUrl = base64Data;
        annotatedImgUrl = null;
        previewImg.src = base64Data;
        previewWrapper.classList.remove('hidden');
        imgMeta.textContent = `${file.name || 'Pasted Document'} (${(file.size / 1024).toFixed(1)} KB)`;
        predictBase64(base64Data);
      };
      reader.readAsDataURL(file);
    }

    async function predictBase64(base64Data) {
      setLoading(true);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_data: base64Data })
        });
        const result = await res.json();
        showResult(result);
      } catch (err) {
        alert('OCR error: ' + err.message);
      } finally {
        setLoading(false);
      }
    }

    function setLoading(isLoading) {
      if (isLoading) {
        emptyState.classList.add('hidden');
        resultState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        latencyBadge.classList.add('hidden');
        linesCountBadge.classList.add('hidden');
      } else {
        loadingState.classList.add('hidden');
      }
    }

    function showResult(res) {
      emptyState.classList.add('hidden');
      resultState.classList.remove('hidden');

      fullTextArea.value = res.full_text || '(No text detected)';
      // Auto adjust height
      const linesCount = (res.lines || []).length;
      fullTextArea.rows = Math.min(10, Math.max(3, linesCount));

      linesCountBadge.textContent = `${linesCount} line${linesCount === 1 ? '' : 's'}`;
      linesCountBadge.classList.remove('hidden');

      if (res.timing_ms) {
        latencyBadge.textContent = `Det: ${res.timing_ms.det_ms}ms | Rec: ${res.timing_ms.rec_ms}ms (${res.timing_ms.total_ms}ms)`;
      }
      latencyBadge.classList.remove('hidden');

      // Setup annotated image toggle
      if (res.annotated_image) {
        annotatedImgUrl = res.annotated_image;
        showingAnnotated = true;
        previewImg.src = annotatedImgUrl;
        toggleAnnotatedBtn.innerHTML = '<span>Show Original Image</span>';
        toggleAnnotatedBtn.classList.remove('hidden');
      } else {
        toggleAnnotatedBtn.classList.add('hidden');
      }

      // Populate lines list
      linesList.innerHTML = '';
      (res.lines || []).forEach(line => {
        const row = document.createElement('div');
        row.className = "flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs";
        const scorePct = (line.score * 100).toFixed(1);
        row.innerHTML = `
          <div class="flex items-center gap-2 overflow-hidden pr-2">
            <span class="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-mono text-[10px] font-bold">
              ${line.line_number}
            </span>
            <span class="kurdish-text text-sm font-semibold text-white truncate text-right">${line.text}</span>
          </div>
          <div class="flex items-center gap-2 flex-shrink-0 font-mono text-[11px]">
            <span class="${line.score >= 0.8 ? 'text-emerald-400' : 'text-amber-400'} font-semibold">${scorePct}%</span>
          </div>
        `;
        linesList.appendChild(row);
      });
    }

    toggleAnnotatedBtn.addEventListener('click', () => {
      showingAnnotated = !showingAnnotated;
      if (showingAnnotated && annotatedImgUrl) {
        previewImg.src = annotatedImgUrl;
        toggleAnnotatedBtn.innerHTML = '<span>Show Original Image</span>';
      } else if (originalImgUrl) {
        previewImg.src = originalImgUrl;
        toggleAnnotatedBtn.innerHTML = '<span>Show Bounding Boxes</span>';
      }
    });

    clearBtn.addEventListener('click', () => {
      previewImg.src = '';
      previewWrapper.classList.add('hidden');
      fileInput.value = '';
      originalImgUrl = null;
      annotatedImgUrl = null;
      emptyState.classList.remove('hidden');
      resultState.classList.add('hidden');
      latencyBadge.classList.add('hidden');
      linesCountBadge.classList.add('hidden');
    });

    copyBtn.addEventListener('click', () => {
      const text = fullTextArea.value.trim();
      if (!text || text.startsWith('(No text')) return;
      navigator.clipboard.writeText(text).then(() => {
        copyLabel.textContent = 'Copied!';
        setTimeout(() => copyLabel.textContent = 'Copy All', 1500);
      });
    });

    downloadBtn.addEventListener('click', () => {
      const text = fullTextArea.value.trim();
      if (!text || text.startsWith('(No text')) return;
      const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'kurdish_ocr_extracted.txt';
      a.click();
    });

    // Initialize
    loadSamples();
  </script>
</body>
</html>
"""


class InferenceRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for web UI and PaddleOCR inference endpoints."""

    reader: DocumentReader = None
    samples_list: list[dict[str, str]] = []
    test_image_dir: Path = None

    def log_message(self, format: str, *args) -> None:
        """Silence default noisy access logs, keep critical ones."""
        pass

    def do_GET(self) -> None:
        """Route GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._send_html(HTML_TEMPLATE)
        elif path == "/api/samples":
            self._send_json(self.samples_list)
        elif path == "/api/sample_image":
            self._serve_sample_image(parsed.query)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        """Route POST inference request."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            self._handle_predict()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_html(self, html_content: str) -> None:
        body = html_content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sample_image(self, query_str: str) -> None:
        params = urllib.parse.parse_qs(query_str)
        fname = params.get("name", [""])[0]
        if not fname:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing image name")
            return

        # Check special sample page
        if fname == "sample_page.jpg":
            cand = Path("data/kurdish_rec/sample_page.jpg").resolve()
            if cand.is_file():
                data = cand.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        if not self.test_image_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Image dir not configured")
            return

        img_path = (self.test_image_dir / fname).resolve()
        if not img_path.is_file() or not str(img_path).startswith(str(self.test_image_dir.resolve())):
            self.send_error(HTTPStatus.NOT_FOUND, "Sample image not found")
            return

        content_type, _ = mimetypes.guess_type(str(img_path))
        data = img_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_predict(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            target_img: Optional[np.ndarray] = None

            if "sample_path" in payload:
                sample_name = payload["sample_path"]
                if sample_name == "sample_page.jpg":
                    cand = Path("data/kurdish_rec/sample_page.jpg").resolve()
                    if cand.is_file():
                        target_img = cv2.imread(str(cand))
                elif self.test_image_dir:
                    cand = (self.test_image_dir / sample_name).resolve()
                    if cand.is_file():
                        target_img = cv2.imread(str(cand))

            elif "image_data" in payload:
                b64_str = payload["image_data"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                target_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if target_img is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "No valid image provided")
                return

            # Execute full document OCR (detection + recognition)
            doc_result = self.reader.read_document(target_img, annotate=True)

            annotated_b64 = None
            if doc_result.annotated_image is not None:
                _, buffer = cv2.imencode(".jpg", doc_result.annotated_image)
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            lines_data = [
                {
                    "text": line.text,
                    "score": line.score,
                    "line_number": line.line_number,
                    "box": line.box,
                }
                for line in doc_result.lines
            ]

            resp = {
                "full_text": doc_result.full_text,
                "lines": lines_data,
                "annotated_image": annotated_b64,
                "timing_ms": doc_result.timing_ms,
                "status": "success",
            }

            self._send_json(resp)

        except Exception as e:
            logger.error(f"Inference request error: {e}", exc_info=True)
            self._send_json({"error": str(e), "status": "error"})


def start_server(
    rec_model_dir: Union[str, Path] = "export/kurdish_final",
    det_model_dir: Union[str, Path] = "assets/base_det_inference",
    host: str = "127.0.0.1",
    port: int = 8501,
    test_label_file: Optional[Union[str, Path]] = "data/kurdish_rec/test_rec.txt",
    test_image_root: Optional[Union[str, Path]] = "data/kurdish_rec/images/test",
    open_browser: bool = True,
) -> None:
    """Launch the Full-Page Kurdish OCR interactive web studio."""
    logger.info("Initializing Full-Page OCR Engine on GPU (DBNet Detector + Kurdish Recognizer)...")
    reader = DocumentReader(rec_model_dir=rec_model_dir, det_model_dir=det_model_dir, use_gpu=True)

    # Warmup once
    dummy_warmup = Path("data/kurdish_rec/sample_page.jpg")
    if dummy_warmup.is_file():
        reader.read_document(dummy_warmup)

    # Load test samples + Full Page sample
    samples: list[dict[str, str]] = []

    # Insert sample page at the first position
    if Path("data/kurdish_rec/sample_page.jpg").is_file():
        samples.append({"filename": "sample_page.jpg", "label": "📄 تەواوی لاپەڕە (Full Page)"})

    lbl_p = Path(test_label_file) if test_label_file else None
    if lbl_p and lbl_p.is_file():
        try:
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_rel = parts[0]
                        label = parts[1]
                        fname = Path(img_rel).name
                        samples.append({"filename": fname, "label": label})
                    if len(samples) >= 6:
                        break
        except Exception as e:
            logger.warning(f"Could not parse sample label file: {e}")

    # Configure handler class
    InferenceRequestHandler.reader = reader
    InferenceRequestHandler.samples_list = samples
    InferenceRequestHandler.test_image_dir = Path(test_image_root).resolve() if test_image_root else None

    server_address = (host, port)
    httpd = HTTPServer(server_address, InferenceRequestHandler)
    url = f"http://{host}:{port}"

    logger.info("=" * 60)
    logger.info(f"  Kurdish Full-Page OCR Studio Running at:")
    logger.info(f"  👉  {url}")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop the server.")

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down web server...")
        httpd.server_close()
        logger.info("Server stopped cleanly.")
