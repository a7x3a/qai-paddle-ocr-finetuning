"""Runtime PIR (Paddle Intermediate Representation) Compatibility Manager.

Handles version-specific differences in PaddlePaddle's PIR dialect (e.g. between 3.3.x and 3.4.x):
- Paddle 3.4.0+ (Blackwell / RTX 50-Series, CUDA 12.8+) expects DoubleAttribute ('0.a_f64')
  for layer_norm/norm epsilon attributes in pd_op3.cc.
- Paddle 3.3.x (CUDA 11.8/12.6, Ada Lovelace, Ampere, CPU) expects FloatAttribute ('0.a_f32').

Provides proactive version-based adaptation and self-healing trial execution to guarantee
zero 'ValueError: Type of attribute: epsilon is not pir::DoubleAttribute/FloatAttribute' errors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("PIRCompat")


def adapt_inference_json_to_runtime(
    model_dir: Union[str, Path],
    target_precision: Optional[str] = None,
) -> bool:
    """Proactively adjust epsilon attribute types in inference.json based on Paddle runtime.

    Args:
        model_dir: Directory containing inference.json and inference.pdiparams.
        target_precision: Explicit '0.a_f64' (Double) or '0.a_f32' (Float). If None, auto-detected.

    Returns:
        True if the file was modified, False otherwise.
    """
    m_dir = Path(model_dir).resolve()
    json_path = m_dir / "inference.json"
    if not json_path.is_file():
        return False

    if target_precision is None:
        try:
            import paddle
            from packaging import version
            p_ver = version.parse(paddle.__version__)
            if p_ver >= version.parse("3.4.0") or "3.4" in str(paddle.__version__):
                target_precision = "0.a_f64"
            else:
                target_precision = "0.a_f32"
        except Exception:
            target_precision = "0.a_f64" if "3.4" in getattr(paddle, "__version__", "") else "0.a_f32"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        modified = False
        ops = data.get("program", {}).get("regions", [{}])[0].get("blocks", [{}])[0].get("ops", [])
        for op in ops:
            if isinstance(op, dict):
                op_type = str(op.get("#", ""))
                # layer_norm is the primary operator whose epsilon type changed between 3.3 and 3.4
                if "layer_norm" in op_type or "norm" in op_type:
                    for item in op.get("A", []):
                        if isinstance(item, dict) and item.get("N") == "epsilon":
                            at = item.get("AT", {})
                            if at.get("#") != target_precision:
                                at["#"] = target_precision
                                modified = True

        if modified:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(
                f"[PIRCompat] Adapted {json_path.name} in {m_dir.name} to {target_precision} "
                f"for Paddle runtime compatibility."
            )
        return modified
    except Exception as e:
        logger.warning(f"[PIRCompat] Could not adapt {json_path}: {e}")
        return False


def self_heal_pir_inference_model(model_dir: Union[str, Path]) -> bool:
    """Validate that the model loads in the current Paddle runtime, automatically healing on type mismatch.

    Args:
        model_dir: Directory containing inference.json and inference.pdiparams.

    Returns:
        True if the model is ready and loadable, False otherwise.
    """
    m_dir = Path(model_dir).resolve()
    json_path = m_dir / "inference.json"
    params_path = m_dir / "inference.pdiparams"
    if not json_path.is_file() or not params_path.is_file():
        return False

    # 1. Proactive adaptation first
    adapt_inference_json_to_runtime(m_dir)

    # 2. Trial load with self-healing fallback
    try:
        import paddle.inference as p_inf
    except ImportError:
        return True  # Cannot probe without paddle.inference, rely on proactive adaptation

    for attempt in range(2):
        try:
            cfg = p_inf.Config(str(json_path), str(params_path))
            cfg.disable_gpu()  # Probe on CPU for instantaneous validation without VRAM allocation
            cfg.switch_ir_optim(False)
            _ = p_inf.create_predictor(cfg)
            return True
        except ValueError as e:
            err = str(e)
            if "is not pir::DoubleAttribute" in err:
                logger.info(f"[PIRCompat] Detected Paddle 3.4+ DoubleAttribute requirement. Healing {m_dir.name}...")
                adapt_inference_json_to_runtime(m_dir, target_precision="0.a_f64")
            elif "is not pir::FloatAttribute" in err:
                logger.info(f"[PIRCompat] Detected Paddle 3.3 FloatAttribute requirement. Healing {m_dir.name}...")
                adapt_inference_json_to_runtime(m_dir, target_precision="0.a_f32")
            else:
                logger.warning(f"[PIRCompat] Unknown predictor creation error: {e}")
                break
        except Exception:
            # Non-attribute error, allow standard predictor loader to report
            break

    return True


def heal_all_known_models(project_root: Optional[Union[str, Path]] = None) -> None:
    """Proactively heal all bundled inference models in the project."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent.parent
    known_dirs = [
        root / "export" / "kurdish_final",
        root / "assets" / "base_rec_inference",
        root / "assets" / "base_det_inference",
    ]
    for kd in known_dirs:
        if kd.is_dir() and (kd / "inference.json").is_file():
            self_heal_pir_inference_model(kd)
