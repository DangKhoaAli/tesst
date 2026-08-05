"""
Query Runner CLI — batch-process AIC queries from a JSON file.

Reads a query JSON file, runs the RetrievalPipeline for each query,
and writes BTC-standard submission CSV files.

Usage (local):
    python scripts/run_queries.py \\
        --queries datasets/queries/kis_queries.json \\
        --index-dir indexes \\
        --output-dir outputs/submission

Usage (Kaggle Notebook):
    !python AIC_System/scripts/run_queries.py \\
        --queries /kaggle/input/aic-queries/kis_queries.json \\
        --index-dir /kaggle/input/aic-indexes \\
        --output-dir /kaggle/working/submission

Input JSON format (array of query objects):
    [
        {
            "query_id": "q001",
            "type": "textual_kis",
            "text": "Tìm video về diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời"
        },
        {
            "query_id": "q002",
            "type": "qa",
            "description": "Trong video lễ trao giải âm nhạc...",
            "question": "Có bao nhiêu người lên sân khấu nhận giải lớn nhất?"
        }
    ]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.evaluation.submission_formatter import SubmissionFormatter
from src.common.enums import QueryType
from src.reasoning.query_classifier import QueryClassifier
from src.utils.logger import get_logger

logger = get_logger("run_queries")


def parse_args():
    parser = argparse.ArgumentParser(description="Run AIC queries and produce submission CSV")
    parser.add_argument("--queries",    required=True,
                        help="Path to queries JSON file")
    parser.add_argument("--index-dir",  default="indexes",
                        help="Dir containing faiss_visual.index + keyframe_master.parquet")
    parser.add_argument("--output-dir", default="outputs/submission",
                        help="Output directory for submission CSV files")
    parser.add_argument("--top-k",      type=int, default=100,
                        help="Number of retrieval candidates (default: 100)")
    parser.add_argument("--clip-model", default="ViT-B-32",
                        help="CLIP model name (default: ViT-B-32)")
    parser.add_argument("--device",     default=None,
                        help="Compute device: cuda / cpu (auto-detect if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------
    # Load queries
    # --------------------------------------------------------
    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error(f"Query file not found: {queries_path}")
        sys.exit(1)

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries from {queries_path}")

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------
    logger.info("Loading RetrievalPipeline...")
    t0 = time.time()
    pipeline = RetrievalPipeline.from_index_dir(
        index_dir=args.index_dir,
        clip_model=args.clip_model,
        device=args.device,
        top_k_retrieval=args.top_k,
    )
    logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")

    # --------------------------------------------------------
    # Run queries
    # --------------------------------------------------------
    formatter  = SubmissionFormatter(output_dir=args.output_dir)
    classifier = QueryClassifier()

    t_total = time.time()
    errors = 0

    for i, query_dict in enumerate(queries):
        qid   = str(query_dict.get("query_id", i))
        qtype = classifier.classify(query_dict)
        t_q   = time.time()

        try:
            evidence = pipeline.run(query_dict, query_id=qid)
        except Exception as e:
            logger.error(f"Query {qid} failed: {e}")
            errors += 1
            continue

        if evidence is None:
            logger.warning(f"No result for query_id={qid}")
            continue

        # Add to appropriate submission bucket
        if qtype == QueryType.TEXTUAL_KIS:
            formatter.add_kis(qid, evidence)

        elif qtype == QueryType.QA:
            answer = query_dict.get("expected_answer", "")  # placeholder; Sprint 4 fills this
            formatter.add_qa(qid, evidence, answer=answer)

        elif qtype == QueryType.TRAKE:
            # TRAKE returns multi-event result; stub for now
            formatter.add_trake(qid, evidence.video_id, {1: evidence.frame_idx})

        elapsed = time.time() - t_q
        logger.info(
            f"[{i+1}/{len(queries)}] {qid} ({qtype.value}): "
            f"{evidence.video_id} frame_idx={evidence.frame_idx} "
            f"pts={evidence.pts_time:.2f}s  ({elapsed:.2f}s)"
        )

    # --------------------------------------------------------
    # Save submission files
    # --------------------------------------------------------
    formatter.save_kis()
    formatter.save_qa()
    formatter.save_trake()
    formatter.save_json()

    stats = formatter.stats()
    total_time = time.time() - t_total
    avg_time = total_time / max(len(queries), 1)

    logger.info("=" * 60)
    logger.info(f"Done in {total_time:.1f}s  (avg {avg_time:.2f}s/query)")
    logger.info(f"KIS:   {stats['kis']} results")
    logger.info(f"Q&A:   {stats['qa']} results")
    logger.info(f"TRAKE: {stats['trake']} results")
    logger.info(f"Errors: {errors}")
    logger.info(f"Output: {Path(args.output_dir).resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
