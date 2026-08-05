AIC-System/
│
├── configs/                     # 1. Quản lý toàn bộ cấu hình hệ thống (Hierarchical YAML)
│   ├── system.yaml              # Cấu hình phần cứng (CUDA devices, num_workers, seed, paths)
│   ├── retrieval.yaml           # Cấu hình Top-K search, strategy weights, fusion parameters
│   ├── embedding.yaml           # Thông số các model embedding (CLIP, SigLIP, BGE-M3, batch size)
│   ├── ocr.yaml                 # Cấu hình PaddleOCR / Florence-2 (Language, min confidence)
│   ├── asr.yaml                 # Cấu hình Whisper (Model size: large-v3, beam_size, language)
│   ├── detector.yaml            # Cấu hình YOLOv11 / Grounding DINO (Confidence thresholds)
│   ├── llm.yaml                 # Cấu hình LLM/VLM (API key, Gemini/Ollama endpoints, temperature)
│   ├── qdrant.yaml              # Thông số Qdrant Vector DB (Host, port, collection schemas)
│   ├── faiss.yaml               # Thông số FAISS index (Index type: IVF-PQ/HNSW, metric)
│   ├── evaluation.yaml         # Đường dẫn bộ Groundtruth & cấu hình metrics (Recall@K, mAP)
│   └── logging.yaml             # Format log, log level, rotation policy
│
├── datasets/                    # 2. Data Lake & Tiền xử lý (Gitignored ngoại trừ .gitkeep)
│   ├── raw_videos/              # Video gốc đầu vào (.mp4, .mkv)
│   ├── keyframes/               # Ảnh keyframe đã cắt chia theo Video ID / Shot ID
│   ├── shots/                   # Phân đoạn shot (Chứa JSON thông tin shot boundary)
│   ├── scenes/                  # Phân đoạn cảnh lớn (Scene grouping)
│   ├── metadata/                # Metadata thô từ BTC hoặc sinh tự động
│   ├── subtitles/               # Phụ đề / Transcript từ ASR (.json, .srt)
│   ├── ocr/                     # Dữ liệu chữ OCR bóc tách được theo keyframe (.json)
│   ├── objects/                 # Kết quả phát hiện vật thể (Bounding boxes & labels)
│   ├── captions/                # Mô tả tự động sinh bởi VLM cho từng keyframe
│   ├── embeddings/              # File npy / hdf5 lưu cache embedding vector
│   ├── queries/                 # Danh sách các câu hỏi test & câu hỏi thi chính thức
│   └── groundtruth/             # Bộ nhãn đáp án chuẩn để benchmark đánh giá
│
├── indexes/                     # 3. Nơi lưu trữ Index Vector & Text (Gitignored)
│   ├── clip/                    # FAISS / Annoy Index cho CLIP visual vector
│   ├── siglip/                  # Vector Index cho SigLIP visual vector
│   ├── ocr/                     # BM25 / Sparse Text Index cho dữ liệu chữ OCR
│   ├── text/                    # Sparse / Dense Index cho Captions & Transcripts
│   ├── object/                  # Inverted Index cho thẻ vật thể & vị trí
│   ├── multimodal/              # Joint Index đa phương thức
│   ├── faiss/                   # Các file chỉ mục `.index` binary của FAISS
│   └── qdrant/                  # Local storage snapshot của Qdrant DB
│
├── models/                      # 4. Lưu Checkpoint / Trọng số mô hình AI (Chỉ chứa weights, KHÔNG chứa code)
│   ├── clip/                    # Checkpoint OpenCLIP / ViT-L-14
│   ├── siglip/                  # Checkpoint SigLIP / SigLIP2
│   ├── whisper/                 # Checkpoint Whisper Large-v3 / Faster-Whisper
│   ├── paddleocr/               # Trọng số PaddleOCR detection & recognition
│   ├── grounding_dino/          # Trọng số Grounding DINO
│   ├── sam2/                    # Trọng số Segment Anything 2
│   ├── yolov11/                 # Trọng số YOLOv11 object detector
│   ├── florence2/               # Trọng số Florence-2 vision-language model
│   └── llm/                     # Local GGUF / HuggingFace Weights (Qwen2.5-VL, LLaVA)
│
├── outputs/                     # 5. Nơi lưu kết quả đầu ra & File nộp bài
│   ├── retrieval/               # Kết quả tìm kiếm dạng JSON thô cho từng query
│   ├── rerank/                  # Kết quả sau khi qua tầng Re-ranking
│   ├── evaluation/              # Báo cáo đánh giá (CSV, JSON metrics summary)
│   ├── prediction/              # Kết quả dự đoán khung hình / timestamp
│   ├── submission/              # File `.csv` / `.json` định dạng chuẩn nộp BTC AI Challenge
│   └── visualization/           # Ảnh heatmap, biểu đồ phân tích kết quả, UI screenshots
│
├── logs/                        # 6. Nhật ký hoạt động của hệ thống
│   ├── api/                     # Access log & Request payload log từ FastAPI
│   ├── preprocessing/           # Log quá trình tách khung hình, OCR, ASR
│   ├── retrieval/               # Log chi tiết quá trình truy xuất (Query latency, top-k score)
│   ├── benchmark/               # Log đo đạc hiệu năng (Profiling GPU VRAM, Memory, Latency)
│   └── errors/                  # Log lỗi hệ thống & Stack traces
│
├── tests/                       # 7. Bộ kiểm thử tự động (Automated Testing Suite)
│   ├── conftest.py              # Pytest fixtures (Sample keyframes, Mock Vector DB, Fake queries)
│   ├── unit/                    # Kiểm thử đơn vị từng module độc lập
│   │   ├── test_frame_extractor.py
│   │   ├── test_ocr_extractor.py
│   │   ├── test_embedder.py
│   │   ├── test_fusion.py
│   │   └── test_query_parser.py
│   ├── integration/             # Kiểm thử tích hợp luồng
│   │   ├── test_indexing_pipeline.py
│   │   └── test_retrieval_pipeline.py
│   └── performance/             # Kiểm thử tải & Latency benchmark
│       ├── test_search_latency.py
│       └── test_vram_usage.py
│
├── notebooks/                   # 8. Không gian thử nghiệm & Khám phá dữ liệu (EDA)
│   ├── 01_eda_dataset.ipynb     # Phân tích thống kê video, độ dài, độ phân giải
│   ├── 02_test_ocr_models.ipynb # So sánh độ chính xác PaddleOCR vs Florence-2
│   ├── 03_test_embeddings.ipynb # Kiểm thử CLIP vs SigLIP vs DINOv2
│   ├── 04_fusion_experiments.ipynb# Thử nghiệm các hệ số trọng số RRF & Weighted Sum
│   └── 05_prompt_engineering.ipynb# Thử nghiệm Prompt cho LLM Query Parser
│
├── experiments/                 # 9. Quản lý các thí nghiệm Ablation Study
│   ├── configs/                 # File cấu hình riêng cho từng đợt thử nghiệm
│   └── runs/                    # Lưu thông số WandB / MLflow / TensorBoard
│
├── docs/                        # 10. Tài liệu kỹ thuật chi tiết
│   ├── architecture.md          # Tài liệu tổng quan kiến trúc
│   ├── pipeline.md              # Chi tiết luồng Offline Indexing & Online Retrieval
│   ├── api.md                   # Tài liệu hướng dẫn sử dụng REST API (OpenAPI spec)
│   ├── benchmark.md             # Báo cáo so sánh hiệu năng qua các phiên bản
│   ├── experiment.md            # Sổ tay ghi chép kết quả thí nghiệm
│   └── setup.md                 # Hướng dẫn cài đặt môi trường, driver CUDA
│
├── scripts/                     # 11. Các Script tự động hóa & Utility CLI
│   ├── download_models.py       # Tải checkpoint tự động từ HuggingFace
│   ├── download_dataset.py      # Tải dữ liệu video thi từ server
│   ├── build_indexes.py         # Script chạy đánh index toàn bộ dataset
│   ├── run_benchmark.py         # Script chạy đánh giá tự động trên bộ groundtruth
│   └── format_submission.py     # Conversion script ra file nộp bài CSV
│
├── src/                         # 12. MÃ NGUỒN CHÍNH (Core Application Source Code)
│   ├── api/                     # FastAPI Application Layer
│   ├── ui/                      # Streamlit UI Layer
│   ├── pipeline/                # Pipelines (Indexing, Retrieval, Evaluation)
│   ├── orchestrator/            # Workflow Engine & Intent Routing
│   ├── preprocessing/           # Video Frame Extractor, Shot Boundary, Keyframe Selection
│   ├── feature_extractors/      # Services: OCR, ASR, Captioner, Object Detector
│   ├── embeddings/              # Encoders: Visual (CLIP/SigLIP/DINOv2), Text (BGE/E5)
│   ├── retrieval/               # Retrievers: Visual, Text, OCR, Object, Fusion
│   ├── fusion/                  # Score Fusion algorithms (RRF, Weighted Sum)
│   ├── reranking/               # Rerankers: Cross-Encoder, LLM Reranker, Temporal Reranker
│   ├── reasoning/               # Query Parser, Intent Classifier, Spatial/Temporal Reasoner
│   ├── llm/                     # LLM/VLM Client, Prompt Manager, Structured JSON Parsers
│   ├── evidence/                # Shot Aggregator, Timestamp Window Selector, Explanation
│   ├── storage/                 # Metadata Store (SQLite/Parquet), Multi-level Cache
│   ├── database/                # Vector DB Adapters (Qdrant, FAISS)
│   ├── evaluation/              # Latency Profiler, Metrics Calculation
│   ├── plugins/                 # Dynamic Plugin Loader & Model Registry
│   ├── common/                  # Core Data Contracts (Dataclasses, Enums, Exceptions)
│   └── utils/                   # Image, Video, Logger, Timer Utilities
│
├── .gitignore                   # Loại trừ datasets/, models/, indexes/, outputs/, logs/
├── Dockerfile                   # Docker build container hỗ trợ CUDA
├── docker-compose.yml           # Compose chạy Qdrant, Redis, FastAPI, Streamlit
├── Makefile                     # Shortcut commands (make index, make test, make run)
├── pyproject.toml               # Quản lý dependencies & package build (Poetry / Hatch)
├── requirements.txt             # Danh sách thư viện Python
├── main.py                      # CLI Entrypoint cho toàn bộ hệ thống
└── README.md                    # Hướng dẫn tổng quan dự án



Cấu Trúc Chi Tiết Thư Mục
src/
├── api/                         # REST API Services (FastAPI)
│   ├── __init__.py
│   ├── main.py                  # Điểm khởi chạy FastAPI app
│   ├── routes/                  # API Routers
│   │   ├── search.py            # Endpoints cho tìm kiếm video/keyframe
│   │   ├── indexing.py          # Endpoints cho offline pipeline trigger
│   │   └── health.py            # System diagnostic & GPU memory status
│   ├── schemas/                 # Pydantic schemas cho request/response
│   │   ├── search_schema.py
│   │   └── indexing_schema.py
│   └── dependencies.py          # Injection cho Pipelines & DB Instances
│
├── ui/                          # User Interface (Streamlit / Web App)
│   ├── app.py                   # Entry point giao diện Streamlit
│   ├── pages/                   # Các trang tính năng
│   │   ├── 1_🔍_Search.py       # Giao diện tìm kiếm đa phương thức (Text, Image, Hybrid)
│   │   ├── 2_⚙️_Indexing.py     # Dashboard quản lý và tiền xử lý video
│   │   └── 3_📊_Benchmark.py    # Trang đánh giá Recall@K, mAP, Latency
│   └── components/              # UI components tái sử dụng
│       ├── video_player.py      # Player xem video đúng timestamp
│       ├── keyframe_grid.py     # Grid hiển thị kết quả keyframe top-K
│       └── filter_bar.py        # Thanh lọc theo OCR, Object, Event
│
├── pipeline/                    # Quản lý luồng thực thi end-to-end
│   ├── base.py                  # Abstract Base Pipeline
│   ├── preprocessing_pipeline.py# Luồng: Raw Video -> Shot Detection -> Keyframes
│   ├── indexing_pipeline.py     # Luồng: Keyframes -> OCR/ASR/Embedding -> Vector DB
│   ├── retrieval_pipeline.py    # Luồng: Query Parsing -> Multi-retriever -> Fusion -> Rerank -> Evidence
│   └── evaluation_pipeline.py   # Luồng đánh giá tự động trên bộ Groundtruth
│
├── orchestrator/                # Lớp điều phối thông minh
│   ├── workflow_engine.py       # Động hóa việc chạy DAG (Directed Acyclic Graph) pipeline
│   └── strategy_router.py       # Định tuyến chiến lược tìm kiếm dựa trên Intent của Query
│
├── preprocessing/               # Tiền xử lý dữ liệu video đầu vào
│   ├── frame_extractor.py       # Tách khung hình bằng OpenCV / PyAV với FPS tùy chỉnh
│   ├── shot_detector.py         # Phát hiện chuyển cảnh (Content-aware Shot Boundary Detection)
│   ├── keyframe_selector.py     # Chọn keyframe đại diện (Middle frame / Clustering / Score)
│   ├── duplicate_removal.py     # Lọc khung hình trùng lặp (Perceptual Hash / Cosine Distance)
│   └── metadata_builder.py     # Tổng hợp JSON metadata chuẩn cho từng Keyframe/Shot
│
├── feature_extractors/          # Dịch vụ trích xuất đặc trưng chuyên sâu
│   ├── base.py                  # Interface chuẩn cho mọi Feature Extractor
│   ├── ocr_extractor.py         # Trích xuất văn bản trong ảnh (PaddleOCR / EasyOCR)
│   ├── asr_extractor.py         # Trích xuất giọng nói / phụ đề (Whisper / Faster-Whisper)
│   ├── object_detector.py       # Phát hiện vật thể (YOLOv11 / Florence-2 / Grounding DINO)
│   └── captioner.py             # Sinh mô tả tự động (LLaVA / Florence-2 / Qwen2-VL)
│
├── embeddings/                  # Trích xuất Vector Embeddings
│   ├── base.py                  # Interface cho Embedder (Encode Text / Encode Image)
│   ├── visual/                  # Visual Embedders
│   │   ├── clip.py              # CLIP (OpenCLIP, ViT-L/14)
│   │   ├── siglip.py            # SigLIP / SigLIP2
│   │   └── dinov2.py            # DINOv2 cho đặc trưng chi tiết ảnh
│   ├── text/                    # Text Embedders cho OCR/Captions
│   │   ├── bge.py               # BGE-M3 (Multilingual, Multi-granularity)
│   │   └── e5.py                # Multilingual-E5
│   └── multimodal/              # Kết hợp đa thức (Joint Embeddings)
│
├── retrieval/                   # Các Module tìm kiếm đơn miền (Single-domain Retrievers)
│   ├── base.py                  # Abstract Base Retriever
│   ├── visual_retriever.py      # Tìm kiếm tương đồng vector ảnh (Visual similarity)
│   ├── text_retriever.py        # Tìm kiếm văn bản trên OCR / Captions / Transcripts (BM25 / Vector)
│   ├── object_retriever.py      # Tìm kiếm theo thẻ vật thể & bounding box
│   └── fusion_retriever.py      # Kết hợp kết quả từ nhiều đơn vị tìm kiếm
│
├── fusion/                      # Chiến lược tổng hợp kết quả (Score Fusion)
│   ├── base.py
│   ├── reciprocal_rank.py       # Reciprocal Rank Fusion (RRF)
│   ├── weighted_sum.py          # Chuẩn hóa score (Min-Max/Z-Score) và tính tổng có trọng số
│   └── hybrid_fusion.py         # Hybrid Dense-Sparse Fusion
│
├── reranking/                   # Tái xếp hạng kết quả (Re-ranking)
│   ├── base.py
│   ├── clip_reranker.py         # Cross-modal fine-grained scoring
│   ├── cross_encoder.py         # Text-Text cross encoder cho OCR/Caption
│   ├── llm_reranker.py          # Dùng VLM/LLM đánh giá mối liên quan giữa Query & Keyframe
│   └── temporal_reranker.py     # Thống kê liên tục về mặt thời gian của các shot liền kề
│
├── reasoning/                   # Suy luận câu hỏi nâng cao (Advanced Reasoning)
│   ├── query_parser.py          # Phân tích câu hỏi ra cấu trúc (Objects, Action, OCR, Time Window)
│   ├── query_classifier.py      # Phân loại Intent (Visual-heavy, OCR-heavy, Temporal, Event)
│   ├── temporal_reasoner.py     # Suy luận thứ tự hành động (vd: "A xảy ra trước B")
│   └── spatial_reasoner.py      # Suy luận vị trí không gian (vd: "vật nằm bên trái xe bus")
│
├── llm/                         # Tích hợp Large Language / Vision Models
│   ├── client.py                # Client gọi API (Gemini / Local Ollama / vLLM)
│   ├── prompt_templates.py      # Quản lý hệ thống Prompt (Query parsing, Verification, QA)
│   └── response_parser.py       # Structured Output Parser (Pydantic validation từ JSON LLM)
│
├── evidence/                    # Trích xuất bằng chứng xác thực (Evidence Verification)
│   ├── frame_selector.py        # Gộp các keyframe gần kề thành Shot/Scene chứng cứ
│   ├── timestamp_selector.py    # Xกำหนด khoảng thời gian (Start Time - End Time) chính xác
│   └── explanation.py           # Sinh giải thích lý do chọn Video/Timestamp này
│
├── storage/                     # Quản lý lưu trữ Data & Cache
│   ├── metadata_store.py        # SQLite / JSON storage cho Shot & Keyframe Metadata
│   ├── artifact_store.py        # Đường dẫn lưu ảnh Keyframe, Video Clips
│   └── cache.py                 # Redis / Memory Cache cho Embedding & Kết quả query phổ biến
│
├── database/                    # Adapter kết nối Cơ sở dữ liệu Vector
│   ├── base.py                  # Abstract VectorDB Interface
│   ├── qdrant_db.py             # Client Qdrant (Hỗ trợ Named Vectors: visual, ocr, caption)
│   └── faiss_db.py              # FAISS Index Wrapper cho tìm kiếm siêu tốc cục bộ
│
├── evaluation/                  # Module Benchmark & Đánh giá hiệu năng
│   ├── metrics.py               # Cài đặt Recall@K, Precision@K, mAP, MRR, NDCG
│   ├── latency_profiler.py      # Đo thời gian phản hồi từng chặng (Parsing -> Search -> Rerank)
│   └── evaluator.py             # Chạy kiểm thử tự động trên bộ Test Query & Groundtruth
│
├── plugins/                     # Plugin Kiến trúc Mở rộng (Open Architecture)
│   ├── plugin_manager.py        # Tải động (Dynamic loading) các module mô hình tùy chỉnh
│   └── registry.py              # Registry Pattern đăng ký Models, Retrievers, Extractors
│
├── common/                      # Khái niệm & Chuẩn dữ liệu dùng chung (Core Data Contracts)
│   ├── types.py                 # Dataclass chuẩn: VideoItem, KeyframeItem, QueryIntent, SearchResult
│   ├── enums.py                 # System Enum: ModelType, IndexType, SearchStrategy
│   ├── exceptions.py            # Hệ thống Exception tùy chỉnh
│   └── constants.py             # Hằng số mặc định hệ thống
│
└── utils/                       # Utility Functions
    ├── image_utils.py           # Crop, Resize, Format conversion
    ├── video_utils.py           # FFmpeg wrapper, FPS calculation, Timecode conversion
    ├── logger.py                # Structlog / Loguru logger config
    └── timer.py                 # Decorator & Context Manager đo thời gian thực thi code
