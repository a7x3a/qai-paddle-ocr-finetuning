.PHONY: help install prepare smoke benchmark_base pilot train report clean

PYTHON ?= python

help:
	@echo "================================================================================"
	@echo " PaddleOCR Recognition Fine-Tuning Pipeline (Production HPC / Kurdish MLOps)    "
	@echo "================================================================================"
	@echo "Targets:"
	@echo "  install        Install repository dependencies from requirements.txt"
	@echo "  prepare        Preprocess raw annotations, crop polygons, build dictionary"
	@echo "  smoke          Run environment, CUDA, VRAM, and gradient smoke test"
	@echo "  benchmark_base Benchmark un-finetuned baseline foundation model"
	@echo "  pilot          Execute pilot convergence run and quick evaluation"
	@echo "  train          Run production fine-tuning on complete dataset"
	@echo "  report         Benchmark all saved checkpoints and generate leaderboard"
	@echo "  clean          Remove temporary runtime artifacts and Python caches"
	@echo "================================================================================"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) main.py prepare-data

smoke:
	$(PYTHON) main.py smoke-test

benchmark_base:
	$(PYTHON) main.py benchmark-base

pilot:
	$(PYTHON) main.py pilot-run

train:
	$(PYTHON) main.py train

export:
	$(PYTHON) main.py export

report:
	$(PYTHON) main.py benchmark-all --checkpoints-dir output/production_run/checkpoints

infer:
	$(PYTHON) main.py infer --split test --count 5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf output/temp_runtime 2>/dev/null || true
