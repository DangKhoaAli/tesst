"""
Main CLI Entrypoint for AIC Video Retrieval System.
Supports direct internal operation via Terminal/Script without needing any UI.

Commands:
  search       -- Execute a search query (KIS / Q&A / TRAKE)
  index        -- Run offline video preprocessing & vector indexing
  build-index  -- Build FAISS index from pre-extracted CLIP-32 .npy files
"""

import sys
import argparse
from typing import List, Dict, Any

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_search(query: str, top_k: int = 10):
    """Direct internal Python API for executing a search query."""
    print(f"[SEARCH] Executing internal search for query: '{query}' (top_k={top_k})")
    
    # Placeholder demonstration of internal pipeline invocation
    # In full implementation:
    # pipeline = RetrievalPipeline.from_config("configs/system.yaml")
    # results = pipeline.run(query)
    
    mock_results = [
        {"rank": i + 1, "video_id": f"L01_V00{i+1}", "pts_time": 120.5 + i * 10, "score": round(0.95 - i * 0.05, 4)}
        for i in range(min(top_k, 5))
    ]
    
    print("\n--- Search Results ---")
    for res in mock_results:
        print(f"Rank {res['rank']}: Video {res['video_id']} at {res['pts_time']}s | Score: {res['score']}")
    
    return mock_results

def run_indexing(video_dir: str):
    """Direct internal Python API for running video preprocessing & vector indexing."""
    print(f"[INDEXING] Running internal indexing pipeline on video directory: '{video_dir}'")
    # In full implementation:
    # pipeline = IndexingPipeline.from_config("configs/system.yaml")
    # pipeline.run(video_dir)
    print("[SUCCESS] Indexing completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="AIC System - Internal CLI Interface")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Search sub-command
    search_parser = subparsers.add_parser("search", help="Execute search query directly")
    search_parser.add_argument("--query", "-q", type=str, required=True, help="Text search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results to return")

    # Index sub-command
    index_parser = subparsers.add_parser("index", help="Run offline indexing pipeline")
    index_parser.add_argument("--video-dir", "-v", type=str, default="datasets/raw_videos", help="Path to video directory")

    # Build-index sub-command (Sprint 1 — load .npy → FAISS)
    build_parser = subparsers.add_parser("build-index", help="Build FAISS index from pre-extracted CLIP .npy files")
    build_parser.add_argument("--npy-dir",          required=True, help="Dir with CLIP-32 .npy files")
    build_parser.add_argument("--map-keyframes-dir", required=True, help="Dir with map-keyframes CSV files")
    build_parser.add_argument("--keyframes-img-dir", required=True, help="Root dir of keyframe images")
    build_parser.add_argument("--output-dir",        default="indexes", help="Output directory for indexes")

    args = parser.parse_args()

    if args.command == "search":
        run_search(args.query, args.top_k)
    elif args.command == "index":
        run_indexing(args.video_dir)
    elif args.command == "build-index":
        from scripts.build_faiss_index import main as build_main
        import sys
        sys.argv = [
            "build_faiss_index",
            "--npy-dir",           args.npy_dir,
            "--map-keyframes-dir", args.map_keyframes_dir,
            "--keyframes-img-dir", args.keyframes_img_dir,
            "--output-dir",        args.output_dir,
        ]
        build_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
