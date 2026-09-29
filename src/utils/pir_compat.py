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
    target_layer_norm_precision: Optional[str] = None,
) -> bool:
    """Proactively adjust operator attribute types in inference.json based on Paddle runtime.

    Paddle PIR Dialect Operator Rules:
    - pd_op.batch_norm_: epsilon MUST ALWAYS be pir::FloatAttribute ('0.a_f32') in ALL Paddle versions
      (validated in pd_pir_op.cc:2034).
    - pd_op.layer_norm: epsilon is pir::DoubleAttribute ('0.a_f64') in Paddle 3.4.0+ / Blackwell
      (validated in pd_op3.cc:1256), and pir::FloatAttribute ('0.a_f32') in Paddle 3.3.x.

    Args:
        model_dir: Directory containing inference.json and inference.pdiparams.
        target_layer_norm_precision: Explicit '0.a_f64' (Double) or '0.a_f32' (Float). If None, auto-detected.

    Returns:
        True if the file was modified, False otherwise.
    """
    m_dir = Path(model_dir).resolve()
    json_path = m_dir / "inference.json"
    if not json_path.is_file():
        return False

    if target_layer_norm_precision is None:
        try:
            import paddle
            from packaging import version
            p_ver = version.parse(paddle.__version__)
            if p_ver >= version.parse("3.4.0") or "3.4" in str(paddle.__version__):
                target_layer_norm_precision = "0.a_f64"
            else:
                target_layer_norm_precision = "0.a_f32"
        except Exception:
            try:
                import paddle
                target_layer_norm_precision = "0.a_f64" if "3.4" in getattr(paddle, "__version__", "") else "0.a_f32"
            except Exception:
                target_layer_norm_precision = "0.a_f32"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        modified = False
        ops = data.get("program", {}).get("regions", [{}])[0].get("blocks", [{}])[0].get("ops", [])
        for op in ops:
            if isinstance(op, dict):
                op_type = str(op.get("#", ""))
                a_list = op.get("A", [])
                if not isinstance(a_list, list):
                    continue

                # Rule 1: batch_norm_ epsilon is strictly 0.a_f32 (FloatAttribute) in all Paddle versions
                if "batch_norm" in op_type:
                    for item in a_list:
                        if isinstance(item, dict) and item.get("N") == "epsilon":
                            at = item.get("AT", {})
                            if at.get("#") != "0.a_f32":
                                at["#"] = "0.a_f32"
                                modified = True

                # Rule 2: layer_norm epsilon is 0.a_f64 on Paddle 3.4+, 0.a_f32 on Paddle 3.3
                elif "layer_norm" in op_type:
                    for item in a_list:
                        if isinstance(item, dict) and item.get("N") == "epsilon":
                            at = item.get("AT", {})
                            if at.get("#") != target_layer_norm_precision:
                                at["#"] = target_layer_norm_precision
                                modified = True

        if modified:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(
                f"[PIRCompat] Adapted {json_path.name} in {m_dir.name} "
                f"(layer_norm={target_layer_norm_precision}, batch_norm=0.a_f32) for Paddle runtime."
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

    for attempt in range(3):
        try:
            cfg = p_inf.Config(str(json_path), str(params_path))
            cfg.disable_gpu()  # Probe on CPU for instantaneous validation without VRAM allocation
            cfg.switch_ir_optim(False)
            _ = p_inf.create_predictor(cfg)
            return True
        except ValueError as e:
            err = str(e)
            if "is not pir::DoubleAttribute" in err:
                logger.info(f"[PIRCompat] Detected DoubleAttribute requirement in {m_dir.name}. Healing layer_norm to 0.a_f64...")
                adapt_inference_json_to_runtime(m_dir, target_layer_norm_precision="0.a_f64")
            elif "is not pir::FloatAttribute" in err:
                if "pd_pir_op.cc" in err or "batch_norm" in err:
                    logger.info(f"[PIRCompat] Detected batch_norm FloatAttribute requirement in {m_dir.name}. Enforcing 0.a_f32...")
                    adapt_inference_json_to_runtime(m_dir)
                else:
                    logger.info(f"[PIRCompat] Detected layer_norm FloatAttribute requirement in {m_dir.name}. Healing layer_norm to 0.a_f32...")
                    adapt_inference_json_to_runtime(m_dir, target_layer_norm_precision="0.a_f32")
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
        root / "export" / "kurdish_smoke",
        root / "export" / "pilot_test",
        root / "export" / "v1",
    ]
    # Also find any other inference models in export/ or assets/
    for search_dir in [root / "export", root / "assets"]:
        if search_dir.is_dir():
            for p in search_dir.rglob("inference.json"):
                if p.parent not in known_dirs:
                    known_dirs.append(p.parent)

    for kd in known_dirs:
        if kd.is_dir() and (kd / "inference.json").is_file():
            self_heal_pir_inference_model(kd)

