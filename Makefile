.PHONY: help setup prepare smoke benchmark_base pilot train export report pipeline read serve infer clean

UV := $(shell where uv 2>nul || which uv 2>/dev/null)
PYTHON ?= $(if $(UV),uv run python,python)

help:
	@echo "================================================================================"
	@echo " PaddleOCR Recognition Fine-Tuning Pipeline (Kurdish Production HPC / MLOps)   "
	@echo "================================================================================"
	@echo "Targets:"
	@echo "  setup          Automated setup of PaddleOCR, dependencies, models & dataset"
	@echo "  prepare        Download dataset from Hugging Face & extract shards to data/"
	@echo "  smoke          Run environment, CUDA, VRAM, and gradient smoke test"
	@echo "  benchmark_base Benchmark un-finetuned baseline foundation model"
	@echo "  pilot          Execute pilot convergence run and quick evaluation"
	@echo "  train          Run production fine-tuning on complete dataset"
	@echo "  pipeline       Run end-to-end flow: Smoke -> Base -> Pilot -> Train -> Export"
	@echo "  export         Export best checkpoint to deployable inference model"
	@echo "  report         Benchmark all saved checkpoints and generate leaderboard"
	@echo "  read           OCR full-page document, PDF, or book (INPUT=path)"
	@echo "  serve          Launch interactive web studio browser interface"
	@echo "  infer          Run inference on sample images with ground-truth comparison"
	@echo "  clean          Remove temporary runtime artifacts and Python caches"
	@echo "================================================================================"

setup:
	$(PYTHON) main.py setup

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

pipeline:
	$(PYTHON) main.py pipeline

export:
	$(PYTHON) main.py export

report:
	$(PYTHON) main.py benchmark-all --checkpoints-dir output/production_run/checkpoints

read:
	$(PYTHON) main.py read --input $(INPUT)

serve:
	$(PYTHON) main.py serve

infer:
	$(PYTHON) main.py infer --split test --count 5

benchmark_unseen:
	$(PYTHON) main.py benchmark-unseen --count 1000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf output/temp_runtime 2>/dev/null || true
