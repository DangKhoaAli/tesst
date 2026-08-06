# AIC Video Retrieval System

Hệ thống truy xuất video cho cuộc thi **AIC-HCMC**, xử lý 3 loại query chính thức:
- **Dạng 1 — Textual KIS**: Tìm kiếm khung hình theo mô tả văn bản
- **Dạng 2 — Q&A**: Trả lời câu hỏi về nội dung video
- **Dạng 3 — TRAKE**: Định vị chuỗi sự kiện theo thứ tự thời gian

---

## Kiến trúc hệ thống

```
CLIP .npy ──────► FAISS Index (visual)  ─┐
                                          ├──► RRF Fusion ──► FrameSelector ──► frame_idx
PaddleOCR ──┐                            │
Qwen2.5-VL ─┼──► BGE-M3 ──► Qdrant ────┘
Whisper ────┘

Q&A: + Qwen2.5-VL VQA (score_relevance → answer_question)
TRAKE: + Video-level RRF Phase 1 + Per-event alignment Phase 2
```

---

## Cấu trúc thư mục

```
AIC_System/
├── configs/
│   ├── system.yaml          # Toàn bộ config hệ thống
│   └── kaggle_paths.yaml    # Đường dẫn Kaggle dataset
├── datasets/
│   └── queries/
│       └── sample_queries.json   # Query mẫu (KIS + QA + TRAKE)
├── notebooks/
│   ├── kaggle_01_build_index.ipynb     # Build FAISS index từ .npy
│   ├── kaggle_02_extract_ocr.ipynb     # PaddleOCR trên keyframes
│   ├── kaggle_03_extract_captions.ipynb # Qwen2.5-VL captions
│   ├── kaggle_04_extract_asr.ipynb     # Faster-Whisper ASR
│   ├── kaggle_05_run_queries.ipynb     # Chạy queries → submission CSV
│   ├── kaggle_06_build_qdrant.ipynb    # Build Qdrant text index
│   └── kaggle_07_evaluate.ipynb        # Đánh giá Recall@K, MRR
├── scripts/
│   ├── build_faiss_index.py     # CLI build FAISS
│   ├── build_qdrant_index.py    # CLI build Qdrant
│   ├── run_queries.py           # CLI batch query runner
│   └── kaggle_setup.sh          # Bash setup cho Kaggle
├── src/
│   ├── common/                  # types, enums, constants
│   ├── database/                # faiss_db, qdrant_db
│   ├── embeddings/              # clip (visual), bge (text)
│   ├── evaluation/              # submission_formatter, evaluator
│   ├── evidence/                # frame_selector
│   ├── feature_extractors/      # ocr, asr, captioner
│   ├── fusion/                  # reciprocal_rank (RRF)
│   ├── llm/                     # qwen_client, prompt_templates, response_parser
│   ├── pipeline/                # retrieval_pipeline, qa_pipeline, trake_pipeline
│   ├── reasoning/               # query_parser, query_classifier
│   ├── retrieval/               # visual_retriever, text_retriever, base
│   ├── storage/                 # metadata_store
│   └── utils/                   # logger
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Thứ tự chạy trên Kaggle

### Bước 1 — Chạy 1 lần duy nhất (xây dựng index)
```
Notebook 01: FAISS index       (~15 min)
Notebook 02: OCR extraction    (~3 hours)
Notebook 03: Captions          (~8 hours)  [không bắt buộc ngay]
Notebook 04: ASR extraction    (~4 hours)  [không bắt buộc ngay]
Notebook 06: Build Qdrant      (~30 min)   [sau khi 02/03/04 xong]
```

### Bước 2 — Chạy mỗi khi có query (thi đấu)
```
Notebook 05: Run queries → submission_kis.csv / submission_qa.csv / submission_trake.csv
```

---

## Chạy local (nếu có data)

```powershell
# Build FAISS index
python main.py build-index \
    --npy-dir "D:\data\clip-features-32" \
    --map-keyframes-dir "D:\data\map-keyframes" \
    --keyframes-img-dir "D:\data\keyframes" \
    --output-dir "indexes"

# Chạy queries
python scripts/run_queries.py \
    --queries "datasets/queries/sample_queries.json" \
    --index-dir "indexes" \
    --output-dir "outputs/submission"
```

---

## Format query JSON

```json
[
  {
    "query_id": "q001",
    "type": "textual_kis",
    "text": "Mô tả sự kiện bạn muốn tìm"
  },
  {
    "query_id": "q002",
    "type": "qa",
    "description": "Mô tả bối cảnh sự kiện",
    "question": "Câu hỏi cụ thể?"
  },
  {
    "query_id": "q003",
    "type": "trake",
    "activity": "Tên hoạt động",
    "events": [
      {"id": 1, "name": "Tên bước", "description": "...", "hint": "..."},
      {"id": 2, "name": "Tên bước", "description": "...", "hint": "..."}
    ]
  }
]
```

---

## Output format (BTC submission)

```csv
# submission_kis.csv
query_id,video_id,frame_idx
q001,L21_V001,1500

# submission_qa.csv
query_id,video_id,frame_idx,answer
q002,L21_V002,900,5

# submission_trake.csv
query_id,video_id,event_1_frame_idx,event_2_frame_idx,event_3_frame_idx,event_4_frame_idx
q003,L21_V003,450,900,1200,1500
```
