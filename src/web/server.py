"""Lightweight, high-performance web studio for Kurdish PaddleOCR inference.

Provides a full-page and multi-page document processing studio with DBNet text detection,
fine-tuned Kurdish text recognition (SVTR), semantic layout analysis (Titles, Headers,
Paragraphs, Tables, Columns), PDF multi-page navigation, zoom/pan interactive canvas,
bi-directional highlighting, and multi-format exports (Markdown, TXT, JSON, Visuals).
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
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from src.pipeline.document_reader import DocumentReader, DocumentResult
from src.utils.logger import setup_logger

logger = setup_logger("WebStudio")

# In-memory session cache for uploaded multi-page PDFs
PDF_SESSIONS: dict[str, dict[str, Any]] = {}
MAX_SESSIONS = 20

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ku" dir="ltr" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kurdish PaddleOCR - Full Document & Layout Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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
      background-color: #06090e;
      color: #f1f5f9;
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      overflow-x: hidden;
    }
    .kurdish-font {
      font-family: 'Vazirmatn', system-ui, sans-serif;
      direction: rtl;
    }
    .glass-panel {
      background: rgba(13, 19, 33, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-panel:hover {
      border-color: rgba(6, 182, 212, 0.25);
    }
    .dropzone-border {
      background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='16' ry='16' stroke='%23334155' stroke-width='2' stroke-dasharray='8%2c 8' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
    }
    .dropzone-active {
      background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='16' ry='16' stroke='%2306b6d4' stroke-width='2' stroke-dasharray='8%2c 8' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
      background-color: rgba(6, 182, 212, 0.05);
    }
    /* Custom Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.4); }
    ::-webkit-scrollbar-thumb { background: rgba(51, 65, 85, 0.6); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(6, 182, 212, 0.5); }
    .viewport-canvas {
      cursor: grab;
      user-select: none;
    }
    .viewport-canvas:active {
      cursor: grabbing;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-cyan-500 selection:text-black">

  <!-- Top App Navigation -->
  <header class="border-b border-slate-800/80 sticky top-0 z-50 backdrop-blur-md bg-slate-950/80">
    <div class="max-w-[1720px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-sky-600 to-blue-700 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20 text-xl font-kurdish">
          ق
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-extrabold text-base sm:text-lg tracking-tight text-white">Kurdish PaddleOCR Studio</h1>
            <span class="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono font-medium">v2.0 Full-Page</span>
          </div>
          <p class="text-xs text-slate-400">DBNet Full-Page Detection + Fine-Tuned Kurdish SVTR + Layout Structure</p>
        </div>
      </div>

      <!-- Header Center: Quick Status Badges -->
      <div class="hidden md:flex items-center gap-2">
        <div class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 font-mono">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>CUDA Acceleration</span>
        </div>
        <div class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 font-mono">
          <span class="text-cyan-400">RTL</span>
          <span>Column-Aware</span>
        </div>
      </div>

      <!-- Header Right: Export Suite -->
      <div class="flex items-center gap-2">
        <button id="export-md-btn" title="Download Markdown with Layout Headings" class="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 text-xs font-medium text-slate-200 hover:text-cyan-300 transition flex items-center gap-1.5 disabled:opacity-40 disabled:pointer-events-none">
          <svg class="w-3.5 h-3.5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span>.md</span>
        </button>
        <button id="export-txt-btn" title="Download Plain Text" class="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 text-xs font-medium text-slate-200 hover:text-cyan-300 transition flex items-center gap-1.5 disabled:opacity-40 disabled:pointer-events-none">
          <span>.txt</span>
        </button>
        <button id="export-json-btn" title="Download Structured JSON AST" class="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 text-xs font-medium text-slate-200 hover:text-cyan-300 transition flex items-center gap-1.5 disabled:opacity-40 disabled:pointer-events-none">
          <span>.json</span>
        </button>
        <button id="export-img-btn" title="Download Annotated Document Visual" class="px-2.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-black text-xs font-semibold shadow-md shadow-cyan-600/20 transition flex items-center gap-1.5 disabled:opacity-40 disabled:pointer-events-none">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span>Save Image</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Quick Samples Bar -->
  <section class="border-b border-slate-800/60 bg-slate-950/40 px-4 sm:px-6 py-2.5">
    <div class="max-w-[1720px] mx-auto flex items-center gap-3 overflow-x-auto text-xs">
      <span class="text-slate-400 font-semibold uppercase tracking-wider flex items-center gap-1.5 flex-shrink-0 font-mono text-[11px]">
        <svg class="w-3.5 h-3.5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Test Samples:
      </span>
      <div id="sample-chips" class="flex items-center gap-2 flex-nowrap">
        <!-- Dynamically populated chips -->
      </div>
    </div>
  </section>

  <!-- Main Dual Studio Workspace -->
  <main class="flex-1 max-w-[1720px] w-full mx-auto px-4 sm:px-6 py-4 flex flex-col lg:flex-row gap-5 items-stretch">

    <!-- LEFT WORKSPACE: High-Res Document Viewer & Canvas -->
    <div class="lg:w-[54%] flex flex-col gap-3">
      <div class="glass-panel rounded-2xl p-4 flex flex-col gap-3 h-[calc(100vh-140px)] min-h-[600px]">

        <!-- Viewer Toolbar Header -->
        <div class="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
          
          <!-- View Switcher Tabs -->
          <div class="flex items-center p-1 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            <button id="view-layout-btn" class="px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-medium border border-cyan-500/30 transition flex items-center gap-1.5">
              <span>🎨 Layout Blocks</span>
            </button>
            <button id="view-lines-btn" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5">
              <span>📐 Line Boxes</span>
            </button>
            <button id="view-original-btn" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5">
              <span>📄 Clean Page</span>
            </button>
          </div>

          <!-- Zoom & Pan Controls -->
          <div class="flex items-center gap-1.5 bg-slate-950 px-2 py-1 rounded-xl border border-slate-800 text-xs font-mono">
            <button id="zoom-out-btn" class="w-6 h-6 rounded flex items-center justify-center hover:bg-slate-800 text-slate-300 font-bold transition">−</button>
            <span id="zoom-level-text" class="px-1.5 text-slate-400 font-medium">100%</span>
            <button id="zoom-in-btn" class="w-6 h-6 rounded flex items-center justify-center hover:bg-slate-800 text-slate-300 font-bold transition">+</button>
            <div class="w-px h-4 bg-slate-800 mx-0.5"></div>
            <button id="zoom-fit-btn" class="px-2 py-0.5 rounded hover:bg-slate-800 text-slate-300 transition text-[11px]">Fit</button>
            <button id="zoom-reset-btn" class="px-2 py-0.5 rounded hover:bg-slate-800 text-slate-300 transition text-[11px]">1:1</button>
          </div>

          <!-- Document Reset / Upload New -->
          <div class="flex items-center gap-2">
            <button id="upload-new-btn" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 hover:text-white transition flex items-center gap-1">
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span>Upload</span>
            </button>
          </div>

        </div>

        <!-- PDF Multi-Page Navigation Bar (visible if total_pages > 1) -->
        <div id="pdf-nav-bar" class="hidden flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900/90 border border-cyan-500/20 text-xs">
          <div class="flex items-center gap-2">
            <button id="pdf-prev-btn" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-500 hover:text-black font-semibold text-slate-300 transition flex items-center gap-1">
              ◀ Prev Page
            </button>
            <span class="text-slate-400 font-mono">
              Page <span id="pdf-current-page" class="text-cyan-400 font-bold">1</span> of <span id="pdf-total-pages" class="text-white font-bold">1</span>
            </span>
            <button id="pdf-next-btn" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-500 hover:text-black font-semibold text-slate-300 transition flex items-center gap-1">
              Next Page ▶
            </button>
          </div>
          <div class="flex items-center gap-2 text-slate-400 font-mono text-[11px]">
            <span>Fast Interactive PDF Paging</span>
          </div>
        </div>

        <!-- Canvas Container / Interactive Viewport -->
        <div id="viewport-container" class="relative flex-1 bg-slate-950/90 rounded-xl overflow-hidden border border-slate-900 flex items-center justify-center">

          <!-- Dropzone Empty State (shown before file loaded) -->
          <div id="dropzone" class="absolute inset-4 dropzone-border rounded-xl flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-200 group z-10">
            <input type="file" id="file-input" class="hidden" accept="image/*,application/pdf">
            <div class="w-16 h-16 rounded-2xl bg-slate-900/90 flex items-center justify-center border border-slate-800 text-slate-400 group-hover:text-cyan-400 group-hover:border-cyan-500/50 group-hover:scale-105 transition duration-200">
              <svg class="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div class="text-center flex flex-col gap-1.5 max-w-sm">
              <p class="text-base font-semibold text-slate-200">
                <span class="text-cyan-400 underline decoration-cyan-500/40 underline-offset-4">Click to upload</span> or drag document here
              </p>
              <p class="text-xs text-slate-400">Supports full-page scans, multi-page PDFs, book images, or phone photos (PNG, JPG, PDF, WEBP, TIFF)</p>
              <p class="text-[11px] text-slate-500 pt-1 font-mono">Or press <kbd class="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Ctrl+V</kbd> to paste from clipboard</p>
            </div>
          </div>

          <!-- Loading Spinner Overlay -->
          <div id="loading-overlay" class="hidden absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30">
            <div class="w-10 h-10 border-3 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <p id="loading-text" class="text-xs font-mono text-cyan-400 uppercase tracking-wider">Detecting document layout & reading Kurdish text...</p>
          </div>

          <!-- Interactive Zoom/Pan Image Layer -->
          <div id="canvas-wrapper" class="hidden w-full h-full overflow-hidden flex items-center justify-center cursor-grab">
            <img id="document-canvas-img" src="" alt="Document View" class="max-w-none transition-transform duration-75 origin-center pointer-events-none select-none shadow-2xl">
          </div>

        </div>

        <!-- Canvas Footer Status -->
        <div class="flex items-center justify-between text-[11px] font-mono text-slate-500 pt-1">
          <div id="doc-meta-info" class="flex items-center gap-3">
            <span>Ready</span>
          </div>
          <div class="flex items-center gap-2">
            <span>Tip: Drag to pan &bull; Scroll to zoom</span>
          </div>
        </div>

      </div>
    </div>

    <!-- RIGHT WORKSPACE: Structured Reader, Markdown, Lines, JSON -->
    <div class="lg:w-[46%] flex flex-col gap-3">
      <div class="glass-panel rounded-2xl p-4 flex flex-col gap-3 h-[calc(100vh-140px)] min-h-[600px]">

        <!-- Right Header: Tabs & Performance Badges -->
        <div class="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
          
          <!-- Content Navigation Tabs -->
          <div class="flex items-center p-1 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            <button id="tab-formatted-btn" class="px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5">
              <span>📖 Reader</span>
            </button>
            <button id="tab-markdown-btn" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5">
              <span>📝 Markdown</span>
            </button>
            <button id="tab-lines-btn" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5">
              <span>🔍 Lines (<span id="tab-lines-count">0</span>)</span>
            </button>
            <button id="tab-json-btn" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5">
              <span>⚡ JSON</span>
            </button>
          </div>

          <!-- Timing Metric Badges -->
          <div id="timing-badge" class="hidden flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-400">
            <span>0.0 ms</span>
          </div>

        </div>

        <!-- TAB 1: Kurdish Formatted Reader -->
        <div id="tab-formatted-content" class="flex-1 flex flex-col gap-3 overflow-hidden">
          <div class="flex items-center justify-between text-xs text-slate-400 px-1">
            <div class="flex items-center gap-2">
              <span class="font-mono text-[11px] uppercase tracking-wider text-slate-400">STRUCTURED READING VIEW</span>
              <span id="blocks-count-pill" class="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">0 blocks</span>
            </div>
            <div class="flex items-center gap-2">
              <button id="font-decrease-btn" class="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 text-xs font-mono">A-</button>
              <button id="font-increase-btn" class="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 text-xs font-mono">A+</button>
              <button id="copy-formatted-btn" class="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-400 hover:text-cyan-300 text-[11px] flex items-center gap-1">
                <span>Copy</span>
              </button>
            </div>
          </div>

          <!-- Formatted Flow Container -->
          <div id="formatted-flow" class="flex-1 overflow-y-auto pr-2 flex flex-col gap-3 kurdish-font text-right">
            <!-- Empty Placeholder -->
            <div class="py-24 flex flex-col items-center justify-center gap-3 text-center text-slate-600 font-sans">
              <svg class="w-10 h-10 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              <p class="text-sm font-medium text-slate-400">Awaiting Document Input</p>
              <p class="text-xs text-slate-500 max-w-xs">Upload a scanned page or click a sample to see formatted Kurdish layout blocks.</p>
            </div>
          </div>
        </div>

        <!-- TAB 2: Live Markdown Editor -->
        <div id="tab-markdown-content" class="hidden flex-1 flex flex-col gap-2 overflow-hidden">
          <div class="flex items-center justify-between text-xs text-slate-400 px-1 font-mono">
            <span>EXPORTABLE MARKDOWN</span>
            <div class="flex items-center gap-2">
              <button id="copy-markdown-btn" class="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-400 text-[11px]">Copy Markdown</button>
            </div>
          </div>
          <div class="flex-1 rounded-xl bg-slate-950 p-3 border border-slate-900 overflow-hidden flex flex-col">
            <textarea id="markdown-textarea" class="w-full flex-1 bg-transparent border-0 resize-none font-mono text-sm text-slate-200 focus:outline-none leading-relaxed kurdish-font" placeholder="# Kurdish OCR Markdown Output..."></textarea>
          </div>
        </div>

        <!-- TAB 3: Line Inspector -->
        <div id="tab-lines-content" class="hidden flex-1 flex flex-col gap-3 overflow-hidden">
          <div class="flex items-center gap-2">
            <div class="relative flex-1">
              <input type="text" id="lines-search-input" placeholder="Search extracted Kurdish words..." class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 kurdish-font text-right">
            </div>
            <span id="filtered-lines-count" class="text-xs font-mono text-slate-500 flex-shrink-0">0 lines</span>
          </div>

          <div id="lines-table-container" class="flex-1 overflow-y-auto pr-1 flex flex-col gap-2">
            <!-- Injected line rows -->
          </div>
        </div>

        <!-- TAB 4: JSON AST -->
        <div id="tab-json-content" class="hidden flex-1 flex flex-col gap-2 overflow-hidden">
          <div class="flex items-center justify-between text-xs text-slate-400 px-1 font-mono">
            <span>STRUCTURED JSON AST</span>
            <button id="copy-json-btn" class="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-400 text-[11px]">Copy JSON</button>
          </div>
          <div class="flex-1 rounded-xl bg-slate-950 p-3 border border-slate-900 overflow-auto font-mono text-xs text-emerald-400">
            <pre id="json-pre">{}</pre>
          </div>
        </div>

      </div>
    </div>

  </main>

  <script>
    // State Store
    const state = {
      sessionId: null,
      currentPage: 1,
      totalPages: 1,
      viewMode: 'layout', // 'layout' | 'lines' | 'original'
      activeTab: 'formatted', // 'formatted' | 'markdown' | 'lines' | 'json'
      zoom: 1.0,
      pan: { x: 0, y: 0 },
      isDragging: false,
      dragStart: { x: 0, y: 0 },
      fontSizePx: 17,
      resultData: null,
      images: {
        layout: null,
        lines: null,
        original: null,
      }
    };

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadNewBtn = document.getElementById('upload-new-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const canvasWrapper = document.getElementById('canvas-wrapper');
    const documentCanvasImg = document.getElementById('document-canvas-img');
    const docMetaInfo = document.getElementById('doc-meta-info');
    const timingBadge = document.getElementById('timing-badge');

    // View Switchers
    const viewLayoutBtn = document.getElementById('view-layout-btn');
    const viewLinesBtn = document.getElementById('view-lines-btn');
    const viewOriginalBtn = document.getElementById('view-original-btn');

    // Zoom Controls
    const zoomLevelText = document.getElementById('zoom-level-text');
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const zoomFitBtn = document.getElementById('zoom-fit-btn');
    const zoomResetBtn = document.getElementById('zoom-reset-btn');

    // PDF Nav
    const pdfNavBar = document.getElementById('pdf-nav-bar');
    const pdfPrevBtn = document.getElementById('pdf-prev-btn');
    const pdfNextBtn = document.getElementById('pdf-next-btn');
    const pdfCurrentPage = document.getElementById('pdf-current-page');
    const pdfTotalPages = document.getElementById('pdf-total-pages');

    // Tabs
    const tabFormattedBtn = document.getElementById('tab-formatted-btn');
    const tabMarkdownBtn = document.getElementById('tab-markdown-btn');
    const tabLinesBtn = document.getElementById('tab-lines-btn');
    const tabJsonBtn = document.getElementById('tab-json-btn');
    const tabFormattedContent = document.getElementById('tab-formatted-content');
    const tabMarkdownContent = document.getElementById('tab-markdown-content');
    const tabLinesContent = document.getElementById('tab-lines-content');
    const tabJsonContent = document.getElementById('tab-json-content');
    const tabLinesCount = document.getElementById('tab-lines-count');

    // Tab Contents
    const formattedFlow = document.getElementById('formatted-flow');
    const blocksCountPill = document.getElementById('blocks-count-pill');
    const markdownTextarea = document.getElementById('markdown-textarea');
    const linesTableContainer = document.getElementById('lines-table-container');
    const linesSearchInput = document.getElementById('lines-search-input');
    const filteredLinesCount = document.getElementById('filtered-lines-count');
    const jsonPre = document.getElementById('json-pre');
    const sampleChips = document.getElementById('sample-chips');

    // Exports
    const exportMdBtn = document.getElementById('export-md-btn');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const exportImgBtn = document.getElementById('export-img-btn');

    // Font Controls
    const fontIncreaseBtn = document.getElementById('font-increase-btn');
    const fontDecreaseBtn = document.getElementById('font-decrease-btn');
    const copyFormattedBtn = document.getElementById('copy-formatted-btn');
    const copyMarkdownBtn = document.getElementById('copy-markdown-btn');
    const copyJsonBtn = document.getElementById('copy-json-btn');

    // Initialize Samples
    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        sampleChips.innerHTML = '';
        samples.forEach(s => {
          const btn = document.createElement('button');
          btn.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 transition text-xs flex-shrink-0";
          btn.innerHTML = `<span class="kurdish-font font-medium">${s.label}</span>`;
          btn.addEventListener('click', () => runSample(s));
          sampleChips.appendChild(btn);
        });
      } catch (e) {
        console.error('Failed to load samples:', e);
      }
    }

    async function runSample(sample) {
      setLoading(true, `Loading & processing ${sample.label}...`);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_path: sample.filename, page_number: 1 })
        });
        const data = await res.json();
        if (data.status === 'error') {
          alert('Error: ' + data.error);
          return;
        }
        applyOCRResult(data);
      } catch (err) {
        alert('OCR error: ' + err.message);
      } finally {
        setLoading(false);
      }
    }

    // File Upload handling (Image or PDF)
    uploadNewBtn.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dropzone-active');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dropzone-active'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dropzone-active');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        processUploadedFile(e.target.files[0]);
      }
    });

    window.addEventListener('paste', (e) => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          const blob = item.getAsFile();
          processUploadedFile(blob);
          break;
        }
      }
    });

    function processUploadedFile(file) {
      const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
      const reader = new FileReader();

      setLoading(true, isPdf ? 'Parsing PDF pages and extracting layout...' : 'Processing full-page document...');

      reader.onload = async (e) => {
        const b64 = e.target.result;
        try {
          const payload = isPdf 
            ? { pdf_data: b64, page_number: 1, filename: file.name }
            : { image_data: b64, filename: file.name };

          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          if (data.status === 'error') {
            alert('Processing error: ' + data.error);
            return;
          }
          applyOCRResult(data);
        } catch (err) {
          alert('Network/OCR Error: ' + err.message);
        } finally {
          setLoading(false);
        }
      };
      reader.readAsDataURL(file);
    }

    // Page navigation for PDFs
    async function navigatePdfPage(targetPage) {
      if (!state.sessionId) return;
      if (targetPage < 1 || targetPage > state.totalPages) return;

      setLoading(true, `Rendering & recognizing page ${targetPage} of ${state.totalPages}...`);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: state.sessionId, page_number: targetPage })
        });
        const data = await res.json();
        if (data.status === 'error') {
          alert('Page navigation error: ' + data.error);
          return;
        }
        applyOCRResult(data);
      } catch (err) {
        alert('Page navigation error: ' + err.message);
      } finally {
        setLoading(false);
      }
    }

    pdfPrevBtn.addEventListener('click', () => navigatePdfPage(state.currentPage - 1));
    pdfNextBtn.addEventListener('click', () => navigatePdfPage(state.currentPage + 1));

    // Handle OCR Result
    function applyOCRResult(data) {
      state.resultData = data;
      state.sessionId = data.session_id || state.sessionId;
      state.currentPage = data.page_number || 1;
      state.totalPages = data.total_pages || 1;

      state.images.layout = data.annotated_layout_image;
      state.images.lines = data.annotated_image;
      state.images.original = data.original_image;

      // Update Viewport
      dropzone.classList.add('hidden');
      canvasWrapper.classList.remove('hidden');
      updateCanvasView();
      resetZoom();

      // Update PDF nav
      if (state.totalPages > 1) {
        pdfNavBar.classList.remove('hidden');
        pdfCurrentPage.textContent = state.currentPage;
        pdfTotalPages.textContent = state.totalPages;
        pdfPrevBtn.disabled = (state.currentPage <= 1);
        pdfNextBtn.disabled = (state.currentPage >= state.totalPages);
        pdfPrevBtn.classList.toggle('opacity-40', state.currentPage <= 1);
        pdfNextBtn.classList.toggle('opacity-40', state.currentPage >= state.totalPages);
      } else {
        pdfNavBar.classList.add('hidden');
      }

      // Update Meta & Timing
      const timing = data.timing_ms || {};
      timingBadge.textContent = `Det: ${timing.det_ms || 0}ms | Rec: ${timing.rec_ms || 0}ms | Total: ${timing.total_ms || 0}ms`;
      timingBadge.classList.remove('hidden');

      const linesCount = (data.lines || []).length;
      const blocksCount = (data.blocks || []).length;
      const cols = data.column_count || 1;

      docMetaInfo.innerHTML = `
        <span class="text-cyan-400 font-semibold">${linesCount} lines</span> &bull; 
        <span class="text-emerald-400 font-semibold">${blocksCount} blocks</span> &bull; 
        <span>${cols} column${cols > 1 ? 's' : ''}</span>
      `;
      tabLinesCount.textContent = linesCount;
      blocksCountPill.textContent = `${blocksCount} blocks`;

      // Enable Export buttons
      exportMdBtn.disabled = false;
      exportTxtBtn.disabled = false;
      exportJsonBtn.disabled = false;
      exportImgBtn.disabled = false;

      // Populate Formatted Reader
      renderFormattedReader(data.blocks || [], data.lines || []);

      // Populate Markdown Tab
      markdownTextarea.value = data.markdown_text || data.full_text || '';

      // Populate Lines Tab
      renderLinesInspector(data.lines || []);

      // Populate JSON Tab
      jsonPre.textContent = JSON.stringify(data, null, 2);
    }

    function updateCanvasView() {
      if (state.viewMode === 'layout' && state.images.layout) {
        documentCanvasImg.src = state.images.layout;
      } else if (state.viewMode === 'lines' && state.images.lines) {
        documentCanvasImg.src = state.images.lines;
      } else if (state.images.original) {
        documentCanvasImg.src = state.images.original;
      } else if (state.images.layout) {
        documentCanvasImg.src = state.images.layout;
      }
    }

    // View Mode Switching
    viewLayoutBtn.addEventListener('click', () => setViewMode('layout'));
    viewLinesBtn.addEventListener('click', () => setViewMode('lines'));
    viewOriginalBtn.addEventListener('click', () => setViewMode('original'));

    function setViewMode(mode) {
      state.viewMode = mode;
      [viewLayoutBtn, viewLinesBtn, viewOriginalBtn].forEach(b => {
        b.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5";
      });
      if (mode === 'layout') {
        viewLayoutBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
      } else if (mode === 'lines') {
        viewLinesBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
      } else {
        viewOriginalBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
      }
      updateCanvasView();
    }

    // Zoom and Pan Engine
    function applyTransform() {
      documentCanvasImg.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
      zoomLevelText.textContent = `${Math.round(state.zoom * 100)}%`;
    }

    function resetZoom() {
      state.zoom = 1.0;
      state.pan = { x: 0, y: 0 };
      applyTransform();
    }

    zoomInBtn.addEventListener('click', () => {
      state.zoom = Math.min(3.5, state.zoom + 0.15);
      applyTransform();
    });
    zoomOutBtn.addEventListener('click', () => {
      state.zoom = Math.max(0.25, state.zoom - 0.15);
      applyTransform();
    });
    zoomResetBtn.addEventListener('click', resetZoom);
    zoomFitBtn.addEventListener('click', () => {
      const container = document.getElementById('viewport-container');
      if (container && documentCanvasImg.naturalWidth && documentCanvasImg.naturalHeight) {
        const scaleW = (container.clientWidth - 20) / documentCanvasImg.naturalWidth;
        const scaleH = (container.clientHeight - 20) / documentCanvasImg.naturalHeight;
        state.zoom = Math.min(scaleW, scaleH, 1.0);
        state.pan = { x: 0, y: 0 };
        applyTransform();
      }
    });

    // Mouse drag pan
    canvasWrapper.addEventListener('mousedown', (e) => {
      state.isDragging = true;
      state.dragStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
      canvasWrapper.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
      if (!state.isDragging) return;
      state.pan.x = e.clientX - state.dragStart.x;
      state.pan.y = e.clientY - state.dragStart.y;
      applyTransform();
    });
    window.addEventListener('mouseup', () => {
      state.isDragging = false;
      canvasWrapper.style.cursor = 'grab';
    });

    // Wheel zoom
    canvasWrapper.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
      state.zoom = Math.max(0.25, Math.min(4.0, state.zoom * zoomFactor));
      applyTransform();
    }, { passive: false });

    // Render Formatted Reader
    function getScriptBadge(script) {
      if (script === 'KU') {
        return `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">KU</span>`;
      } else if (script === 'EN') {
        return `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">EN</span>`;
      } else if (script === 'AR') {
        return `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">AR</span>`;
      } else if (script === 'NUM') {
        return `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">NUM</span>`;
      }
      return `<span class="text-[9px] px-1.5 py-0.5 rounded font-mono font-bold bg-slate-700 text-slate-300">${script || 'AUTO'}</span>`;
    }

    function renderFormattedReader(blocks, lines) {
      formattedFlow.innerHTML = '';
      if (!blocks.length && !lines.length) {
        formattedFlow.innerHTML = '<p class="text-slate-500 font-sans text-center py-10">No text detected on this page.</p>';
        return;
      }

      // If blocks are available, render structured layout blocks
      if (blocks.length > 0) {
        blocks.forEach(b => {
          const card = document.createElement('div');
          card.className = "group relative rounded-xl p-3.5 transition border";

          let typeBadge = "";
          let textClass = "";

          if (b.block_type === 'title') {
            card.classList.add('bg-purple-950/20', 'border-purple-800/40', 'hover:border-purple-500');
            typeBadge = `<span class="text-[10px] px-2 py-0.5 rounded bg-purple-900/60 text-purple-300 font-mono font-semibold">H1 TITLE</span>`;
            textClass = "text-xl sm:text-2xl font-black text-purple-100 leading-snug";
          } else if (b.block_type === 'header') {
            card.classList.add('bg-sky-950/20', 'border-sky-800/40', 'hover:border-sky-500');
            typeBadge = `<span class="text-[10px] px-2 py-0.5 rounded bg-sky-900/60 text-sky-300 font-mono font-semibold">H2 HEADER</span>`;
            textClass = "text-lg sm:text-xl font-bold text-sky-100 leading-snug";
          } else if (b.block_type === 'table') {
            card.classList.add('bg-amber-950/20', 'border-amber-800/40', 'hover:border-amber-500');
            typeBadge = `<span class="text-[10px] px-2 py-0.5 rounded bg-amber-900/60 text-amber-300 font-mono font-semibold">TABLE</span>`;
            textClass = "text-sm sm:text-base font-medium text-amber-100 leading-relaxed font-mono";
          } else if (b.block_type === 'footer') {
            card.classList.add('bg-slate-900/40', 'border-slate-800', 'hover:border-slate-600');
            typeBadge = `<span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">FOOTER</span>`;
            textClass = "text-xs font-normal text-slate-400";
          } else { // paragraph
            card.classList.add('bg-emerald-950/10', 'border-emerald-800/30', 'hover:border-emerald-500/60');
            typeBadge = `<span class="text-[10px] px-2 py-0.5 rounded bg-emerald-900/50 text-emerald-300 font-mono font-semibold">PARAGRAPH</span>`;
            textClass = "text-base sm:text-lg font-medium text-slate-100 leading-loose";
          }

          const dir = b.direction || (b.script === 'EN' ? 'ltr' : 'rtl');
          const alignClass = dir === 'ltr' ? 'text-left' : 'text-right';
          const fontClass = b.script === 'EN' ? 'font-sans' : 'kurdish-font';

          // Table custom visual rendering
          let contentHtml = '';
          if (b.block_type === 'table' && b.text.includes('|')) {
            const linesList = b.text.split('\n').filter(r => r.trim().startsWith('|'));
            if (linesList.length > 0) {
              let tbl = '<div class="overflow-x-auto my-1"><table class="w-full text-xs sm:text-sm border-collapse ' + alignClass + '">';
              linesList.forEach((rowStr, idx) => {
                const cells = rowStr.split('|').map(c => c.trim()).filter(c => c !== '');
                if (idx === 0) {
                  tbl += '<thead class="bg-amber-900/30 text-amber-300 border-b border-amber-800/40"><tr>' + cells.map(c => `<th class="p-2.5 font-semibold ${alignClass}">${c}</th>`).join('') + '</tr></thead><tbody>';
                } else {
                  tbl += `<tr class="border-b border-white/5 hover:bg-white/5 transition">${cells.map(c => `<td class="p-2.5">${c}</td>`).join('')}</tr>`;
                }
              });
              tbl += '</tbody></table></div>';
              contentHtml = tbl;
            } else {
              contentHtml = b.text.replace(/\n/g, '<br>');
            }
          } else {
            contentHtml = b.text.replace(/\n/g, '<br>');
          }

          card.innerHTML = `
            <div class="flex items-center justify-between pb-2 mb-2 border-b border-white/5 font-sans">
              <div class="flex items-center gap-2">
                ${typeBadge}
                ${getScriptBadge(b.script)}
                <span class="text-[10px] text-slate-500 font-mono">Col ${(b.column_index || 0) + 1} &bull; ${(b.confidence * 100).toFixed(1)}%</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-[10px] font-mono text-slate-500 uppercase">${dir}</span>
                <button onclick="navigator.clipboard.writeText(this.getAttribute('data-copy'))" data-copy="${encodeURIComponent(b.text)}" title="Copy block" class="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-cyan-400 transition text-[11px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700">Copy</button>
              </div>
            </div>
            <div class="${textClass} ${fontClass} ${alignClass}" dir="${dir}" style="font-size: ${state.fontSizePx}px;">
              ${contentHtml}
            </div>
          `;
          // Wire up the copy button decoding
          const copyBtn = card.querySelector('button[data-copy]');
          if (copyBtn) {
            const rawCopy = decodeURIComponent(copyBtn.getAttribute('data-copy'));
            copyBtn.onclick = (e) => {
              e.stopPropagation();
              navigator.clipboard.writeText(rawCopy).then(() => {
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy', 1200);
              });
            };
          }

          formattedFlow.appendChild(card);
        });
      } else {
        // Fallback: render individual lines
        lines.forEach(l => {
          const dir = l.direction || (l.script === 'EN' ? 'ltr' : 'rtl');
          const fontClass = l.script === 'EN' ? 'font-sans' : 'kurdish-font';
          const alignClass = dir === 'ltr' ? 'text-left' : 'text-right';

          const div = document.createElement('div');
          div.className = `p-2.5 rounded bg-slate-900/60 border border-slate-800 text-base text-slate-100 ${fontClass} ${alignClass}`;
          div.dir = dir;
          div.textContent = l.text;
          formattedFlow.appendChild(div);
        });
      }
    }

    // Render Lines Inspector
    function renderLinesInspector(lines) {
      linesTableContainer.innerHTML = '';
      const filter = linesSearchInput.value.trim().toLowerCase();
      let shownCount = 0;

      lines.forEach(l => {
        if (filter && !l.text.toLowerCase().includes(filter)) return;
        shownCount++;

        const row = document.createElement('div');
        row.className = "flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 hover:border-cyan-500/50 transition cursor-pointer text-xs";
        const scorePct = (l.score * 100).toFixed(1);
        const scoreColor = l.score >= 0.8 ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' : 'text-amber-400 bg-amber-500/10 border-amber-500/20';
        const dir = l.direction || (l.script === 'EN' ? 'ltr' : 'rtl');
        const fontClass = l.script === 'EN' ? 'font-sans' : 'kurdish-font';

        row.innerHTML = `
          <div class="flex items-center gap-2 overflow-hidden pr-2">
            <span class="w-6 h-6 rounded-full bg-slate-900 border border-slate-800 text-cyan-400 flex items-center justify-center font-mono text-[10px] font-bold flex-shrink-0">
              ${l.line_number}
            </span>
            ${getScriptBadge(l.script)}
            <span class="${fontClass} text-base font-semibold text-slate-100 truncate" dir="${dir}">${l.text}</span>
          </div>
          <div class="flex items-center gap-2 flex-shrink-0 font-mono text-[11px]">
            <span class="text-slate-500 text-[10px]">${l.word_count || 1}w</span>
            <span class="text-slate-500 text-[10px]">C${(l.column_index || 0) + 1}</span>
            <span class="px-2 py-0.5 rounded-full border ${scoreColor} font-semibold">${scorePct}%</span>
          </div>
        `;
        linesTableContainer.appendChild(row);
      });
      filteredLinesCount.textContent = `${shownCount} lines`;
    }

    linesSearchInput.addEventListener('input', () => {
      if (state.resultData && state.resultData.lines) {
        renderLinesInspector(state.resultData.lines);
      }
    });

    // Tab Switching
    tabFormattedBtn.addEventListener('click', () => switchTab('formatted'));
    tabMarkdownBtn.addEventListener('click', () => switchTab('markdown'));
    tabLinesBtn.addEventListener('click', () => switchTab('lines'));
    tabJsonBtn.addEventListener('click', () => switchTab('json'));

    function switchTab(tab) {
      state.activeTab = tab;
      [tabFormattedBtn, tabMarkdownBtn, tabLinesBtn, tabJsonBtn].forEach(b => {
        b.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5";
      });
      [tabFormattedContent, tabMarkdownContent, tabLinesContent, tabJsonContent].forEach(c => c.classList.add('hidden'));

      if (tab === 'formatted') {
        tabFormattedBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
        tabFormattedContent.classList.remove('hidden');
      } else if (tab === 'markdown') {
        tabMarkdownBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
        tabMarkdownContent.classList.remove('hidden');
      } else if (tab === 'lines') {
        tabLinesBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
        tabLinesContent.classList.remove('hidden');
      } else {
        tabJsonBtn.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30 transition flex items-center gap-1.5";
        tabJsonContent.classList.remove('hidden');
      }
    }

    // Font Sizing in Reader
    fontIncreaseBtn.addEventListener('click', () => {
      state.fontSizePx = Math.min(28, state.fontSizePx + 2);
      if (state.resultData) renderFormattedReader(state.resultData.blocks || [], state.resultData.lines || []);
    });
    fontDecreaseBtn.addEventListener('click', () => {
      state.fontSizePx = Math.max(12, state.fontSizePx - 2);
      if (state.resultData) renderFormattedReader(state.resultData.blocks || [], state.resultData.lines || []);
    });

    // Copy Handlers
    copyFormattedBtn.addEventListener('click', () => {
      const text = (state.resultData && (state.resultData.full_text || state.resultData.markdown_text)) || '';
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        copyFormattedBtn.textContent = 'Copied!';
        setTimeout(() => copyFormattedBtn.textContent = 'Copy', 1500);
      });
    });

    copyMarkdownBtn.addEventListener('click', () => {
      const text = markdownTextarea.value;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        copyMarkdownBtn.textContent = 'Copied!';
        setTimeout(() => copyMarkdownBtn.textContent = 'Copy Markdown', 1500);
      });
    });

    copyJsonBtn.addEventListener('click', () => {
      const text = jsonPre.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        copyJsonBtn.textContent = 'Copied!';
        setTimeout(() => copyJsonBtn.textContent = 'Copy JSON', 1500);
      });
    });

    // Download Suite
    exportMdBtn.addEventListener('click', () => {
      const text = markdownTextarea.value;
      downloadFile(text, `kurdish_ocr_p${state.currentPage}.md`, 'text/markdown');
    });

    exportTxtBtn.addEventListener('click', () => {
      const text = (state.resultData && state.resultData.full_text) || '';
      downloadFile(text, `kurdish_ocr_p${state.currentPage}.txt`, 'text/plain');
    });

    exportJsonBtn.addEventListener('click', () => {
      const text = JSON.stringify(state.resultData || {}, null, 2);
      downloadFile(text, `kurdish_ocr_p${state.currentPage}.json`, 'application/json');
    });

    exportImgBtn.addEventListener('click', () => {
      const imgSrc = (state.viewMode === 'lines' ? state.images.lines : state.images.layout) || state.images.original;
      if (!imgSrc) return;
      const a = document.createElement('a');
      a.href = imgSrc;
      a.download = `kurdish_ocr_annotated_p${state.currentPage}.jpg`;
      a.click();
    });

    function downloadFile(content, filename, type) {
      if (!content) return;
      const blob = new Blob([content], { type: `${type};charset=utf-8` });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
    }

    function setLoading(isLoading, text = 'Processing...') {
      if (isLoading) {
        loadingText.textContent = text;
        loadingOverlay.classList.remove('hidden');
      } else {
        loadingOverlay.classList.add('hidden');
      }
    }

    // Initialize application
    loadSamples();
  </script>
</body>
</html>
"""


class InferenceRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for web studio and PaddleOCR inference endpoints."""

    reader: DocumentReader = None
    samples_list: list[dict[str, str]] = []
    test_image_dir: Path = None

    def log_message(self, format: str, *args) -> None:
        """Silence standard access logs, keep errors."""
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
        """Route POST inference requests."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/api/predict", "/api/pdf_page"}:
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

        # Check special sample page and document
        if fname in {"sample_page.jpg", "sample_document.pdf"}:
            cand = Path("data/kurdish_rec") / fname
            if cand.is_file():
                data = cand.read_bytes()
                mime = "application/pdf" if fname.endswith(".pdf") else "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
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
            session_id: Optional[str] = payload.get("session_id")
            page_number: int = int(payload.get("page_number", 1))
            total_pages: int = 1
            doc_result: Optional[DocumentResult] = None

            # Case 1: Navigating an existing cached PDF session
            if session_id and session_id in PDF_SESSIONS:
                cached = PDF_SESSIONS[session_id]
                pdf_bytes = cached["bytes"]
                doc_result, total_pages = self.reader.read_pdf_page(
                    pdf_input=pdf_bytes,
                    page_number=page_number,
                    annotate=True,
                )
                target_img = doc_result.annotated_image

            # Case 2: New PDF Upload via Base64
            elif "pdf_data" in payload:
                b64_str = payload["pdf_data"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                pdf_bytes = base64.b64decode(b64_str)

                new_session_id = uuid.uuid4().hex
                if len(PDF_SESSIONS) >= MAX_SESSIONS:
                    PDF_SESSIONS.pop(next(iter(PDF_SESSIONS)))
                PDF_SESSIONS[new_session_id] = {"bytes": pdf_bytes, "created_at": time.time()}
                session_id = new_session_id

                doc_result, total_pages = self.reader.read_pdf_page(
                    pdf_input=pdf_bytes,
                    page_number=page_number,
                    annotate=True,
                )
                target_img = doc_result.annotated_image

            # Case 3: Sample PDF
            elif payload.get("sample_path") == "sample_document.pdf":
                pdf_path = Path("data/kurdish_rec/sample_document.pdf").resolve()
                if not pdf_path.is_file():
                    raise FileNotFoundError("sample_document.pdf not found")
                pdf_bytes = pdf_path.read_bytes()

                new_session_id = uuid.uuid4().hex
                if len(PDF_SESSIONS) >= MAX_SESSIONS:
                    PDF_SESSIONS.pop(next(iter(PDF_SESSIONS)))
                PDF_SESSIONS[new_session_id] = {"bytes": pdf_bytes, "created_at": time.time()}
                session_id = new_session_id

                doc_result, total_pages = self.reader.read_pdf_page(
                    pdf_input=pdf_bytes,
                    page_number=page_number,
                    annotate=True,
                )
                target_img = doc_result.annotated_image

            # Case 4: Sample Image
            elif "sample_path" in payload:
                sample_name = payload["sample_path"]
                if sample_name == "sample_page.jpg":
                    cand = Path("data/kurdish_rec/sample_page.jpg").resolve()
                    if cand.is_file():
                        target_img = cv2.imread(str(cand))
                elif self.test_image_dir:
                    cand = (self.test_image_dir / sample_name).resolve()
                    if cand.is_file():
                        target_img = cv2.imread(str(cand))

            # Case 5: New Image Upload via Base64
            elif "image_data" in payload:
                b64_str = payload["image_data"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                target_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # If image-based and not already processed as PDF
            if doc_result is None:
                if target_img is None:
                    self.send_error(HTTPStatus.BAD_REQUEST, "No valid image or PDF document provided")
                    return
                doc_result = self.reader.read_document(target_img, annotate=True, page_number=page_number)
                total_pages = 1

            # Prepare Base64 outputs for views
            annotated_line_b64 = None
            if doc_result.annotated_image is not None:
                _, buf = cv2.imencode(".jpg", doc_result.annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                annotated_line_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            annotated_layout_b64 = None
            if doc_result.annotated_layout_image is not None:
                _, buf = cv2.imencode(".jpg", doc_result.annotated_layout_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                annotated_layout_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            original_b64 = None
            if target_img is not None:
                _, buf = cv2.imencode(".jpg", target_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                original_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            resp = {
                "status": "success",
                "session_id": session_id,
                "page_number": doc_result.page_number,
                "total_pages": total_pages,
                "column_count": doc_result.column_count,
                "full_text": doc_result.full_text,
                "markdown_text": doc_result.markdown_text or doc_result.full_text,
                "blocks": [b.to_dict() for b in doc_result.blocks],
                "lines": [
                    {
                        "text": line.text,
                        "score": line.score,
                        "line_number": line.line_number,
                        "column_index": line.column_index,
                        "box": line.box,
                        "script": line.script,
                        "direction": line.direction,
                        "word_count": line.word_count,
                    }
                    for line in doc_result.lines
                ],
                "annotated_layout_image": annotated_layout_b64,
                "annotated_image": annotated_line_b64,
                "original_image": original_b64 or annotated_layout_b64,
                "timing_ms": doc_result.timing_ms,
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

    # 1. Full Page Kurdish Document (A4 high-res)
    if Path("data/kurdish_rec/sample_page.jpg").is_file():
        samples.append({"filename": "sample_page.jpg", "label": "📄 تەواوی لاپەڕە (Full Page A4)"})

    # 2. Multi-page PDF Kurdish Document
    if Path("data/kurdish_rec/sample_document.pdf").is_file():
        samples.append({"filename": "sample_document.pdf", "label": "📑 پەڕتووکی کوردی (2-Page PDF)"})

    # 3. Add a few line crop samples
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
    httpd = ThreadingHTTPServer(server_address, InferenceRequestHandler)
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
