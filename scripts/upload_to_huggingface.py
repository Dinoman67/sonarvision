#!/usr/bin/env python3
"""
Upload YOLOv8-ESI to Hugging Face Private Repository
===================================================

Usage:
    python scripts/upload_to_huggingface.py --token <YOUR_HF_TOKEN> [--repo-id <USERNAME>/yolo-esi]
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Push YOLOv8-ESI model to Hugging Face Private Repository")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face Access Token (with write permission)")
    parser.add_argument("--repo-id", type=str, default=None, help="Target HF Repo ID (e.g., username/yolo-esi). If omitted, defaults to <your_username>/yolo-esi")
    parser.add_argument("--folder", type=str, default="hf_yolo_esi_model", help="Folder containing files to upload")
    parser.add_argument("--private", action="store_true", default=True, help="Create repository as private (default: True)")
    
    args = parser.parse_args()
    
    if not args.token:
        print("❌ Error: No Hugging Face token provided. Provide via --token <token> or set HF_TOKEN environment variable.")
        sys.exit(1)
        
    api = HfApi(token=args.token)
    
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Authenticated with Hugging Face as: {username} ({user_info.get('email', 'no email')})")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
        
    repo_id = args.repo_id or f"{username}/yolo-esi"
    
    print(f"🚀 Target Repository: {repo_id} (Private: {args.private})")
    
    # Create private repository if it doesn't exist
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=args.token,
            private=args.private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to create/verify repository {repo_id}: {e}")
        sys.exit(1)
        
    folder_path = Path(args.folder).resolve()
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        sys.exit(1)
        
    print(f"📦 Uploading contents of {folder_path} to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial release: YOLOv8-ESI private model weights, modules, and documentation"
        )
        print(f"🎉 Successfully uploaded YOLOv8-ESI model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload folder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
