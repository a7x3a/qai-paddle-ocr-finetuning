"""Local browser UI for manual Kurdish OCR checkpoint testing."""

from __future__ import annotations

import cgi
import json
import mimetypes
import shutil
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_ROOT = PROJECT_ROOT / "PaddleOCR"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PREDICTOR = PROJECT_ROOT / "scripts" / "_run_predictor.py"
DICTIONARY = PROJECT_ROOT / "data" / "kurdish_rec" / "kurdish_dict.txt"
BASE_DICTIONARY = PROJECT_ROOT / "PaddleOCR" / "ppocr" / "utils" / "dict" / "ppocrv5_arabic_dict.txt"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "kurdish_arabic_ppocr_v5"
EXPORT_ROOT = PROJECT_ROOT / "export"
HTML = PROJECT_ROOT / "scripts" / "manual_ui.html"


def checkpoint_options() -> list[dict[str, str]]:
    options = []
    for prefix in ("best_accuracy", "latest"):
        path = OUTPUT_ROOT / f"{prefix}.pdparams"
        if path.is_file():
            options.append({"label": f"fine-tuned: {prefix}", "path": str(path.relative_to(PROJECT_ROOT)), "dictionary": str(DICTIONARY.relative_to(PROJECT_ROOT))})
    base_export = EXPORT_ROOT / "base_arabic_ppocr_v5"
    if (base_export / "inference.pdiparams").is_file():
        options.append({"label": "base: Arabic PP-OCRv5", "path": str(base_export.relative_to(PROJECT_ROOT)), "dictionary": str(BASE_DICTIONARY.relative_to(PROJECT_ROOT))})
    for path in sorted(EXPORT_ROOT.glob("**/inference.pdiparams")):
        directory = path.parent
        if directory == base_export:
            continue
        dictionary = BASE_DICTIONARY if "base_arabic" in directory.name else DICTIONARY
        options.append({"label": f"export: {directory.relative_to(EXPORT_ROOT)}", "path": str(directory.relative_to(PROJECT_ROOT)), "dictionary": str(dictionary.relative_to(PROJECT_ROOT))})
    return options


def resolve_model(requested: str, temp_root: Path) -> Path:
    requested_path = (PROJECT_ROOT / requested).resolve()
    if (requested_path / "inference.pdiparams").is_file():
        return requested_path
    if not requested_path.is_file() and requested_path.with_suffix(".pdparams").is_file():
        requested_path = requested_path.with_suffix("")
    if requested_path.suffix == ".pdparams":
        requested_path = requested_path.with_suffix("")
    if not (requested_path.with_suffix(".pdparams")).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {requested}")
    export_dir = EXPORT_ROOT / "manual_ui" / requested_path.name
    if not (export_dir / "inference.pdiparams").is_file():
        command = [
            str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "export_kurdish.ps1"),
        ]
        # PowerShell is used only for the existing export wrapper.
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROJECT_ROOT / "scripts" / "export_kurdish.ps1"),
             "-Checkpoint", str(requested_path.relative_to(PROJECT_ROOT)), "-Output", str(export_dir.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
    return export_dir


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/checkpoints":
            self.send_json({"checkpoints": checkpoint_options()})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/test":
            self.send_error(404)
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
            )
            checkpoint = form.getfirst("checkpoint", "")
            dictionary = form.getfirst("dictionary", str(DICTIONARY.relative_to(PROJECT_ROOT)))
            cpu = form.getfirst("cpu", "false") == "true"
            ground_truth = form.getfirst("ground_truth", "")
            upload = form["image"]
            if not checkpoint or not getattr(upload, "filename", None):
                raise ValueError("Choose a checkpoint and image first.")
            with tempfile.TemporaryDirectory(prefix="qai-ocr-ui-") as temp_dir:
                temp_root = Path(temp_dir)
                image_path = temp_root / Path(upload.filename).name
                with image_path.open("wb") as image_file:
                    shutil.copyfileobj(upload.file, image_file)
                model_dir = resolve_model(checkpoint, temp_root)
                request = {"model_dir": str(model_dir), "dictionary": str((PROJECT_ROOT / dictionary).resolve()), "batch_size": 1, "use_gpu": not cpu, "images": [str(image_path)]}
                result = subprocess.run([str(VENV_PYTHON), str(PREDICTOR)], input=json.dumps(request, ensure_ascii=False), text=True, encoding="utf-8", capture_output=True, cwd=PADDLE_ROOT, check=False)
                if result.returncode:
                    raise RuntimeError(result.stderr or result.stdout)
                lines = [line for line in result.stdout.splitlines() if line.strip()]
                prediction = json.loads(lines[-1])[0]
                response = {"checkpoint": checkpoint, "filename": upload.filename, "prediction": prediction, "ground_truth": ground_truth}
                if ground_truth:
                    response["exact_match"] = ground_truth == prediction.get("text", "")
                self.send_json(response)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[manual-ui] {format % args}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Manual OCR UI: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
