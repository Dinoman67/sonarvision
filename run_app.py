#!/usr/bin/env python3
"""
YOLO-ESI Debris Analysis Web Application Runner
================================================
Starts the unified FastAPI application serving both the REST API
and the built React/TypeScript frontend.

Usage:
    python run_app.py [--port 8000] [--host 0.0.0.0] [--model-path /path/to/model.onnx]
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run YOLO-ESI Debris Analysis Web App")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--model-path", type=str, default=None, help="Custom path to YOLO-ESI ONNX model")
    args = parser.parse_args()

    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    print("\n" + "=" * 70)
    print("🌊 YOLO-ESI DEBRIS ANALYSIS WEB APPLICATION")
    print("   Environmental Remote Sensing & Side-Scan Sonar Analysis")
    print("=" * 70)
    print(f"  Host       : http://{args.host}:{args.port}")
    print(f"  Model Path : {os.getenv('MODEL_PATH', '/home/ashish/Downloads/yolo_esi_fp16.onnx')}")
    print(f"  Frontend   : Integrated React + TypeScript + Tailwind UI")
    print("=" * 70 + "\n")

    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
