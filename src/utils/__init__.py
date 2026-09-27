"""Utility modules: logging, configuration injection, dynamic VRAM tuning, and visual tables."""

from .config_injector import PaddleConfigInjector, auto_detect_gpu_batch_size
from .logger import setup_logger
from .visualizer import Visualizer

__all__ = [
    "setup_logger",
    "PaddleConfigInjector",
    "auto_detect_gpu_batch_size",
    "Visualizer",
]
