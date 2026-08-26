"""
Deepfake Detector — Master Entry Point & CLI

Commands
────────
  # 1. Launch the Streamlit Web Interface:
  python -m app.main ui
  streamlit run app/ui/streamlit_app.py

  # 2. Launch the FastAPI REST API Server:
  python -m app.main api [--host 0.0.0.0] [--port 8000]

  # 3. Analyze a single media file:
  python -m app.main predict path/to/media.mp4

  # 4. Batch analyze a directory:
  python -m app.main batch path/to/folder/ [--output results.json] [--recursive]

  # 5. Check hardware & model status:
  python -m app.main health
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

# Ensure standard output can print Unicode characters safely on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_single_predict(file_path: str, output_file: str | None = None) -> None:
    """Analyze a single file and print formatted output."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found — {file_path}")
        sys.exit(1)

    from app.analyzer.media_analyzer import MediaAnalyzer

    print(f"[*] Analyzing: {path.name}")
    print("    Loading neural network models...")

    analyzer = MediaAnalyzer(device="auto")
    report = analyzer.analyze(path)

    verdict_badge = {
        "REAL": "[✅ REAL]",
        "FAKE": "[❌ FAKE]",
        "UNCERTAIN": "[⚠️ UNCERTAIN]",
    }.get(report.verdict, "[?]")

    print()
    print("=" * 60)
    print(f"  {verdict_badge}  Verdict: {report.verdict}")
    print("=" * 60)
    print(f"  Confidence (Fake Probability): {report.confidence * 100:.2f}%")
    print(f"  Media Type:                    {report.media_type.capitalize()}")
    print(f"  Processing Time:               {report.processing_time_ms:.1f} ms")
    print()

    for modality, score in report.scores.items():
        if score is not None:
            bar_len = int(score * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            print(f"  {modality:>10s}: [{bar}] {score * 100:.2f}%")

    print()
    if report.metadata:
        print("  Metadata:")
        for k, v in report.metadata.items():
            if k != "error":
                print(f"    • {k}: {v}")
    print()

    if output_file:
        out_path = Path(output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[*] Report saved to {out_path.resolve()}")


def run_batch_predict(dir_path: str, output_file: str | None = None, recursive: bool = False) -> None:
    """Batch analyze a folder of media files."""
    path = Path(dir_path)
    if not path.is_dir():
        print(f"Error: Directory not found — {dir_path}")
        sys.exit(1)

    from app.analyzer.media_analyzer import MediaAnalyzer

    analyzer = MediaAnalyzer(device="auto")
    print(f"[*] Scanning directory: {path}")
    reports = analyzer.analyze_directory(path, recursive=recursive)

    if not reports:
        print("No supported media files found.")
        return

    print()
    print(f"{'File Name':<35} | {'Verdict':<10} | {'Fake Prob':<10} | {'Type':<8} | {'Latency':<8}")
    print("-" * 80)

    for r in reports:
        fname = r.metadata.get("file_name", "unknown")
        if len(fname) > 32:
            fname = fname[:29] + "..."
        print(f"{fname:<35} | {r.verdict:<10} | {r.confidence * 100:>8.2f}% | {r.media_type:<8} | {r.processing_time_ms:>6.0f}ms")

    print("-" * 80)
    total = len(reports)
    fakes = sum(1 for r in reports if r.verdict == "FAKE")
    reals = sum(1 for r in reports if r.verdict == "REAL")
    uncertains = sum(1 for r in reports if r.verdict == "UNCERTAIN")

    print(f"Total: {total} | Fakes: {fakes} | Real: {reals} | Uncertain: {uncertains}")
    print()

    if output_file:
        out_path = Path(output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in reports], f, indent=2)
        print(f"[*] Batch report saved to {out_path.resolve()}")


def run_ui() -> None:
    """Launch the Streamlit web interface."""
    ui_script = _ROOT / "app" / "ui" / "streamlit_app.py"
    print(f"[*] Starting Streamlit Web App from {ui_script}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_script)])


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI REST server via uvicorn."""
    import uvicorn
    print(f"[*] Starting FastAPI REST server on http://{host}:{port}...")
    print(f"    Interactive Swagger docs: http://localhost:{port}/docs")
    uvicorn.run("app.api.server:app", host=host, port=port, reload=False)


def run_health() -> None:
    """Print hardware and model health status."""
    from app.analyzer.media_analyzer import MediaAnalyzer
    analyzer = MediaAnalyzer()
    status = analyzer.get_system_status()
    print(json.dumps(status, indent=2))


def cli() -> None:
    """Master command-line parser."""
    parser = argparse.ArgumentParser(
        description="Deepfake Media Detector — Image, Video, and Audio Deepfake Analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. predict
    predict_parser = subparsers.add_parser("predict", help="Analyze a single media file")
    predict_parser.add_argument("file", help="Path to image, video, or audio file")
    predict_parser.add_argument("-o", "--output", help="Save JSON report to output file")

    # 2. batch
    batch_parser = subparsers.add_parser("batch", help="Batch analyze a directory of media files")
    batch_parser.add_argument("directory", help="Directory containing media files")
    batch_parser.add_argument("-o", "--output", help="Save JSON report to output file")
    batch_parser.add_argument("-r", "--recursive", action="store_true", help="Scan directory recursively")

    # 3. ui
    subparsers.add_parser("ui", help="Launch Streamlit web application")

    # 4. api
    api_parser = subparsers.add_parser("api", help="Start FastAPI REST server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")

    # 5. health
    subparsers.add_parser("health", help="Check hardware and model status")

    # Handle direct file argument without 'predict' keyword
    if len(sys.argv) > 1 and sys.argv[1] not in ["predict", "batch", "ui", "api", "health", "-h", "--help"]:
        run_single_predict(sys.argv[1])
        return

    args = parser.parse_args()

    if args.command == "predict":
        run_single_predict(args.file, args.output)
    elif args.command == "batch":
        run_batch_predict(args.directory, args.output, args.recursive)
    elif args.command == "ui":
        run_ui()
    elif args.command == "api":
        run_api(args.host, args.port)
    elif args.command == "health":
        run_health()
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()