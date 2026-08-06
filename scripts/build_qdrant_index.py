"""
Build Qdrant Text Index from extracted OCR / Caption / ASR JSON files.

Usage:
    python scripts/build_qdrant_index.py \\
        --ocr-dir      datasets/ocr \\
        --captions-dir datasets/captions \\
        --asr-dir      datasets/subtitles \\
        --qdrant-url   http://localhost:6333 \\
        --overwrite

Usage on Kaggle:
    !python AIC_System/scripts/build_qdrant_index.py \\
        --ocr-dir      /kaggle/working/ocr \\
        --captions-dir /kaggle/working/captions \\
        --asr-dir      /kaggle/working/subtitles \\
        --qdrant-url   http://localhost:6333 \\
        --overwrite
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.qdrant_db import QdrantDB
from src.embeddings.text.bge import BGEEncoder
from src.utils.logger import get_logger

logger = get_logger("build_qdrant_index")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Qdrant text index from JSON features")
    parser.add_argument("--ocr-dir",      default=None, help="Dir with OCR JSON files")
    parser.add_argument("--captions-dir", default=None, help="Dir with Caption JSON files")
    parser.add_argument("--asr-dir",      default=None, help="Dir with ASR JSON files")
    parser.add_argument("--qdrant-url",   default="http://localhost:6333")
    parser.add_argument("--overwrite",    action="store_true", help="Recreate collections")
    parser.add_argument("--batch-size",   type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.ocr_dir, args.captions_dir, args.asr_dir]):
        logger.error("Provide at least one of --ocr-dir, --captions-dir, --asr-dir")
        sys.exit(1)

    # ---- Load BGE-M3 ----
    logger.info("Loading BGE-M3 encoder...")
    encoder = BGEEncoder()
    encoder.load()

    # ---- Connect Qdrant ----
    logger.info(f"Connecting to Qdrant: {args.qdrant_url}")
    db = QdrantDB(url=args.qdrant_url)
    db.connect()
    db.create_collections(overwrite=args.overwrite)

    t_start = time.time()

    # ---- Index each modality ----
    if args.captions_dir and Path(args.captions_dir).exists():
        logger.info("=" * 50)
        logger.info("Indexing CAPTIONS (English)...")
        n = db.index_from_json(
            args.captions_dir, collection="captions",
            text_field="caption_en", bge_encoder=encoder,
            batch_size=args.batch_size,
        )
        logger.info(f"Captions: {n:,} vectors indexed")

    if args.ocr_dir and Path(args.ocr_dir).exists():
        logger.info("=" * 50)
        logger.info("Indexing OCR text...")
        n = db.index_from_json(
            args.ocr_dir, collection="ocr",
            text_field="texts", bge_encoder=encoder,
            batch_size=args.batch_size,
        )
        logger.info(f"OCR: {n:,} vectors indexed")

    if args.asr_dir and Path(args.asr_dir).exists():
        logger.info("=" * 50)
        logger.info("Indexing ASR transcripts...")
        n = db.index_from_json(
            args.asr_dir, collection="asr",
            text_field="asr_text", bge_encoder=encoder,
            batch_size=args.batch_size,
        )
        logger.info(f"ASR: {n:,} vectors indexed")

    logger.info("=" * 50)
    logger.info(f"Done in {time.time() - t_start:.1f}s")

    # Print collection stats
    for alias in ["captions", "ocr", "asr"]:
        try:
            count = db.collection_count(db.COLLECTIONS[alias])
            logger.info(f"  {alias}: {count:,} points in Qdrant")
        except Exception:
            pass


if __name__ == "__main__":
    main()
