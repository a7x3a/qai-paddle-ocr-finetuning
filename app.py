"""Quick launcher for Kurdish PaddleOCR Interactive Web Studio."""

import argparse
from pathlib import Path
from src.web.server import start_server


def main():
    parser = argparse.ArgumentParser(description="Kurdish PaddleOCR Interactive Web Studio")
    parser.add_argument("--model-dir", "--rec-model-dir", dest="model_dir", type=str, default="export/kurdish_final", help="Path to exported recognition model")
    parser.add_argument("--det-model-dir", type=str, default="assets/base_det_inference", help="Path to detection model")
    parser.add_argument("--port", type=int, default=8501, help="Web server port (default: 8501)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    start_server(
        rec_model_dir=args.model_dir,
        det_model_dir=args.det_model_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
