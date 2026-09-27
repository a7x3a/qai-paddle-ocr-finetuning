"""Data ingestion, normalization, polygon cropping, and dataset indexing modules."""

from .normalizer import KurdishTextNormalizer, normalize_kurdish_text
from .processor import DatasetProcessor

__all__ = ["KurdishTextNormalizer", "normalize_kurdish_text", "DatasetProcessor"]
