"""Report writers: JSON for tooling, Markdown for reading, CSV for per-sample review."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

HEADLINE = (
    ("cer", "CER", "lower"),
    ("wer", "WER", "lower"),
    ("exact_match_accuracy", "Exact match", "higher"),
    ("character_accuracy", "Char accuracy", "higher"),
    ("word_accuracy", "Word accuracy", "higher"),
    ("similarity", "Similarity", "higher"),
    ("mean_confidence", "Mean confidence", "higher"),
)

CSV_FIELDS = (
    "image",
    "domain",
    "script",
    "ground_truth",
    "prediction",
    "char_errors",
    "word_errors",
    "confidence",
    "exact_match",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _overall_table(summary: dict) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, label, _ in HEADLINE:
        if key not in summary:
            continue
        value = summary[key]
        lines.append(f"| {label} | {_percent(value) if key != 'mean_confidence' else f'{value:.4f}'} |")
    return "\n".join(lines)


def _breakdown_table(title: str, breakdown: dict) -> str:
    if not breakdown:
        return ""
    lines = [
        f"### {title}",
        "",
        "| Group | Samples | CER | WER | Exact match | Mean confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in breakdown.items():
        if not summary.get("samples"):
            continue
        lines.append(
            f"| {name} | {summary['samples']} | {_percent(summary['cer'])} | {_percent(summary['wer'])} "
            f"| {_percent(summary['exact_match_accuracy'])} | {summary['mean_confidence']:.4f} |"
        )
    return "\n".join(lines)


def _comparison_table(comparison: dict) -> str:
    if not comparison:
        return ""
    lines = [
        "## Comparison vs baseline",
        "",
        "| Metric | Baseline | Candidate | Delta | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label, _ in HEADLINE:
        row = comparison.get(key)
        if not row:
            continue
        formatter = (lambda v: f"{v:.4f}") if key == "mean_confidence" else _percent
        lines.append(
            f"| {label} | {formatter(row['baseline'])} | {formatter(row['candidate'])} "
            f"| {row['delta'] * 100:+.4f}pp | {row['verdict']} |"
        )
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_markdown(path: Path, title: str, payload: dict, comparison: dict | None = None) -> None:
    summary = payload["overall"]
    blocks = [
        f"# {title}",
        "",
        f"- Model: `{payload.get('model_dir', 'n/a')}`",
        f"- Checkpoint: `{payload.get('checkpoint', 'n/a')}`",
        f"- Split: `{payload.get('split', 'n/a')}`",
        f"- Samples: `{summary.get('samples', 0)}`",
        f"- Generated: `{payload.get('generated_at', now())}`",
        "",
        "## Overall",
        "",
        _overall_table(summary),
        "",
        _comparison_table(comparison or {}),
        _breakdown_table("By domain", payload.get("by_domain", {})),
        "",
        _breakdown_table("By script", payload.get("by_script", {})),
        "",
        _breakdown_table("By ground-truth label length", payload.get("by_label_length", {})),
        "",
        _breakdown_table("By image size", payload.get("by_image_size", {})),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def write_html(path: Path, title: str, payload: dict, comparison: dict | None = None) -> None:
    summary = payload["overall"]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        ":root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--muted:#8b93a7;--good:#3fb950;--bad:#f85149;--acc:#58a6ff}",
        "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);",
        "font:15px/1.55 ui-sans-serif,system-ui,'Segoe UI',Roboto,sans-serif}",
        "header{padding:28px 32px 18px;border-bottom:1px solid var(--line)}",
        "h1{margin:0 0 6px;font-size:20px;letter-spacing:-.01em}",
        ".sub{color:var(--muted);font-size:13px}",
        "main{padding:24px 32px 48px;max-width:1180px}",
        ".cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:28px}",
        ".card{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}",
        ".card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em}",
        ".card .v{font-size:26px;font-variant-numeric:tabular-nums;margin-top:4px;font-weight:600}",
        "section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:18px}",
        "h2{font-size:14px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}",
        "table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}",
        "th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);font-size:13.5px}",
        "th:first-child,td:first-child{text-align:left}",
        "th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}",
        "tr:last-child td{border-bottom:none}",
        ".good{color:var(--good)}.bad{color:var(--bad)}.acc{color:var(--acc)}",
        "code{background:#11141a;padding:1px 6px;border-radius:4px;font-size:12.5px}",
        "</style></head><body>",
        f"<header><h1>{html.escape(title)}</h1>",
        f"<div class='sub'><code>{html.escape(str(payload.get('model_dir', 'n/a')))}</code> &middot; "
        f"split <b>{html.escape(str(payload.get('split', 'n/a')))}</b> &middot; "
        f"{summary.get('samples', 0)} samples &middot; {html.escape(str(payload.get('generated_at', '')))}</div></header>",
        "<main><div class='cards'>",
    ]
    for key, label, _ in HEADLINE:
        if key not in summary:
            continue
        value = summary[key]
        shown = f"{value:.4f}" if key == "mean_confidence" else _percent(value)
        parts.append(f"<div class='card'><div class='k'>{html.escape(label)}</div><div class='v'>{shown}</div></div>")
    parts.append("</div>")

    if comparison:
        parts.append("<section><h2>Comparison vs baseline</h2><table><tr><th>Metric</th><th>Baseline</th>"
                     "<th>Candidate</th><th>Delta</th><th>Verdict</th></tr>")
        for key, label, _ in HEADLINE:
            row = comparison.get(key)
            if not row:
                continue
            cls = "good" if row["verdict"] == "improved" else ("bad" if row["verdict"] == "regressed" else "")
            fmt = (lambda v: f"{v:.4f}") if key == "mean_confidence" else _percent
            parts.append(
                f"<tr><td>{html.escape(label)}</td><td>{fmt(row['baseline'])}</td>"
                f"<td>{fmt(row['candidate'])}</td><td class='{cls}'>{row['delta'] * 100:+.4f}pp</td>"
                f"<td class='{cls}'>{row['verdict']}</td></tr>"
            )
        parts.append("</table></section>")

    for heading, key in (
        ("By domain", "by_domain"),
        ("By script", "by_script"),
        ("By ground-truth label length", "by_label_length"),
        ("By image size", "by_image_size"),
    ):
        breakdown = payload.get(key) or {}
        if not breakdown:
            continue
        parts.append(f"<section><h2>{html.escape(heading)}</h2><table>"
                     "<tr><th>Group</th><th>Samples</th><th>CER</th><th>WER</th>"
                     "<th>Exact match</th><th>Confidence</th></tr>")
        for name, item in breakdown.items():
            if not item.get("samples"):
                continue
            parts.append(
                f"<tr><td>{html.escape(str(name))}</td><td>{item['samples']}</td>"
                f"<td>{_percent(item['cer'])}</td><td>{_percent(item['wer'])}</td>"
                f"<td>{_percent(item['exact_match_accuracy'])}</td>"
                f"<td>{item['mean_confidence']:.4f}</td></tr>"
            )
        parts.append("</table></section>")

    parts.append("</main></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_all(
    report_dir: Path,
    stem: str,
    title: str,
    payload: dict,
    rows: Sequence[dict] | None = None,
    comparison: dict | None = None,
) -> dict[str, Path]:
    written = {
        "json": report_dir / f"{stem}.json",
        "md": report_dir / f"{stem}.md",
        "html": report_dir / f"{stem}.html",
    }
    write_json(written["json"], payload)
    write_markdown(written["md"], title, payload, comparison)
    write_html(written["html"], title, payload, comparison)
    if rows is not None:
        written["csv"] = report_dir / f"{stem}.csv"
        write_csv(written["csv"], rows)
    return written
