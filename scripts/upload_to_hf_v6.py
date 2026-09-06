#!/usr/bin/env python3
"""
Upload SonarVision v6 Models & Dataset to Hugging Face (Restricted / Private)
=============================================================================

Usage:
    # 1. Provide token directly:
    python scripts/upload_to_hf_v6.py --token hf_xxxxxxxxxxxxxxxxxxxxxxxxx

    # 2. Or set environment variable:
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
    python scripts/upload_to_hf_v6.py

    # Optional flags:
    --only-model       (Upload only the model repository)
    --only-dataset     (Upload only the dataset repository)
    --username <name>  (Override HF username or organization)
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Push SonarVision v6 Models & Dataset to Hugging Face Private Repositories"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face User Access Token (with 'write' permission). If not provided, reads $HF_TOKEN."
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Hugging Face username or organization (defaults to authenticated account name)."
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="Custom model repo ID (e.g., username/sonarvision-yolov8-esi-v6). Defaults to <username>/sonarvision-yolov8-esi-v6"
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="Custom dataset repo ID (e.g., username/sonarvision-multisource-v6). Defaults to <username>/sonarvision-multisource-v6"
    )
    parser.add_argument(
        "--only-model",
        action="store_true",
        help="Upload only the model repository"
    )
    parser.add_argument(
        "--only-dataset",
        action="store_true",
        help="Upload only the dataset repository"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Set repository visibility to public (Default is restricted/private)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_folder = project_root / "hf_export_v6" / "model"
    dataset_folder = project_root / "hf_export_v6" / "dataset"

    is_private = not args.public
    visibility_str = "Private / Restricted 🔒" if is_private else "Public 🌐"

    print("=" * 70)
    print("🌊 SonarVision v6 Hugging Face Exporter (Restricted / Private)")
    print("=" * 70)

    # 1. Validate Token
    if not args.token:
        print("\n❌ Error: No Hugging Face token provided!")
        print("   Please provide your Hugging Face write token using:")
        print("   python scripts/upload_to_hf_v6.py --token <YOUR_WRITE_TOKEN>")
        print("\n   Or export it in your shell:")
        print("   export HF_TOKEN=\"<YOUR_WRITE_TOKEN>\"")
        print("\n   👉 Generate a token at: https://huggingface.co/settings/tokens (Role: Write)\n")
        sys.exit(1)

    api = HfApi(token=args.token)

    # 2. Authenticate
    try:
        user_info = api.whoami()
        username = args.username or user_info["name"]
        print(f"✅ Authenticated as: {user_info.get('name')} ({user_info.get('email', 'No public email')})")
        print(f"   Target Namespace: {username}")
        print(f"   Repository Visibility: {visibility_str}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please verify that your token is valid and has 'write' permissions.")
        sys.exit(1)

    model_repo_id = args.model_repo or f"{username}/sonarvision-yolov8-esi-v6"
    dataset_repo_id = args.dataset_repo or f"{username}/sonarvision-multisource-v6"

    # 3. Upload Model
    if not args.only_dataset:
        print("\n" + "-" * 70)
        print(f"📦 [1/2] Preparing Model Repository: {model_repo_id}")
        if not model_folder.exists():
            print(f"❌ Model staging directory not found at: {model_folder}")
            sys.exit(1)

        try:
            print(f"   Creating/verifying repository (private={is_private})...")
            create_repo(
                repo_id=model_repo_id,
                token=args.token,
                private=is_private,
                repo_type="model",
                exist_ok=True
            )
            print(f"   ✅ Model repo ready: https://huggingface.co/{model_repo_id}")

            print(f"   Uploading model weights, modules, and README from {model_folder}...")
            api.upload_folder(
                folder_path=str(model_folder),
                repo_id=model_repo_id,
                repo_type="model",
                commit_message="Release YOLOv8-ESI v6 weights, modules, and evaluation documentation"
            )
            print(f"   🎉 Successfully uploaded Model repository: https://huggingface.co/{model_repo_id}")
        except Exception as e:
            print(f"   ❌ Error during model upload: {e}")
            if not args.only_model:
                print("   Continuing to dataset upload...")

    # 4. Upload Dataset
    if not args.only_model:
        print("\n" + "-" * 70)
        print(f"🗂️  [2/2] Preparing Dataset Repository: {dataset_repo_id}")
        if not dataset_folder.exists():
            print(f"❌ Dataset staging directory not found at: {dataset_folder}")
            sys.exit(1)

        try:
            print(f"   Creating/verifying dataset repository (private={is_private})...")
            create_repo(
                repo_id=dataset_repo_id,
                token=args.token,
                private=is_private,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"   ✅ Dataset repo ready: https://huggingface.co/datasets/{dataset_repo_id}")

            print(f"   Uploading 924MB dataset archive and metadata from {dataset_folder}...")
            print("   (This may take several minutes depending on network bandwidth)")
            api.upload_folder(
                folder_path=str(dataset_folder),
                repo_id=dataset_repo_id,
                repo_type="dataset",
                commit_message="Upload SonarVision Multi-Source v6 Dataset (5,558 SSS images, 4 classes, leakage-free)"
            )
            print(f"   🎉 Successfully uploaded Dataset repository: https://huggingface.co/datasets/{dataset_repo_id}")
        except Exception as e:
            print(f"   ❌ Error during dataset upload: {e}")

    print("\n" + "=" * 70)
    print("✨ All tasks processed!")
    if not args.only_dataset:
        print(f"• Model URL:   https://huggingface.co/{model_repo_id}")
    if not args.only_model:
        print(f"• Dataset URL: https://huggingface.co/datasets/{dataset_repo_id}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
