# Vietnamese Legal Assistant

Ứng dụng hỏi đáp pháp luật tiếng Việt sử dụng:

- Elasticsearch cho BM25/lexical retrieval.
- Qdrant cho dense retrieval.
- VietLegal Harrier cho embedding.
- ViRanker để rerank tài liệu.
- Qwen3.5-9B chạy qua vLLM hoặc SGLang.
- FastAPI phục vụ API và giao diện web.

## Cấu trúc chính

```text
src/main/
├── Agents/                         # LangGraph, retrieval, reranker, generation
├── Web/                            # FastAPI và giao diện web
├── deploy/                         # Script start/stop service
├── data -> Legal_assistant/data    # Dữ liệu dùng chung
├── dbs -> Legal_assistant/dbs      # Elasticsearch và Qdrant dùng chung
├── model_cache -> .../model_cache  # Model dùng chung
├── demo.py                         # Entrypoint FastAPI
├── build_hybrid_index.py           # Build Elasticsearch/Qdrant index
└── requirements.txt
```

Trong workspace hiện tại, `data/`, `dbs/` và `model_cache/` là symlink tới:

```text
/home/uet/cuongdam/Legal_assistant/
```

Không cần copy lại các thư mục lớn này. Nếu chuyển project sang máy hoặc đường dẫn
khác, cần tạo lại symlink hoặc đặt dữ liệu/model đúng tên thư mục tương ứng.

## Yêu cầu

- Linux.
- Python 3.12.
- NVIDIA GPU và CUDA.
- Java 8 trở lên cho VnCoreNLP.
- `curl`, `bash` và `nvidia-smi`.

Cài dependencies nếu môi trường chưa có:

```bash
cd /home/uet/cuongdam/K4-Day08-RAG-Pipeline-B1-1/src/main

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các script mặc định tìm Python/vLLM trong:

```text
/home/uet/miniconda3/envs/gemma
```

Có thể đổi bằng biến `CONDA_ENV_PREFIX` hoặc `PYTHON_BIN`.

## Chạy toàn bộ dự án

Đi vào thư mục project:

```bash
cd /home/uet/cuongdam/K4-Day08-RAG-Pipeline-B1-1/src/main
```

Khởi động Elasticsearch, Qdrant, vLLM và FastAPI theo đúng thứ tự:

```bash
bash deploy/start_backend_stack.sh --vllm
```

Script sẽ:

1. Khởi động hoặc bỏ qua Elasticsearch nếu port `9201` đã hoạt động.
2. Khởi động hoặc bỏ qua Qdrant nếu port `6333` đã hoạt động.
3. Khởi động hoặc bỏ qua vLLM nếu port `8006` đã có model server.
4. Khởi động FastAPI backend trên port `8010`.
5. Preload embedding model, reranker và VnCoreNLP.

Nếu muốn dùng cấu hình runtime của project gốc mà không copy file chứa secret:

```bash
LEGAL_ASSISTANT_ENV_FILE=/home/uet/cuongdam/Legal_assistant/legal_assistant.env \
  bash deploy/start_backend_stack.sh --vllm
```

Ví dụ chỉ định GPU và port trực tiếp:

```bash
LLM_GPU_ID=1 \
BACKEND_GPU_ID=0 \
LLM_PORT=8006 \
BACKEND_PORT=8010 \
CHECKPOINTER_BACKEND=none \
LANGSMITH_TRACING=false \
  bash deploy/start_backend_stack.sh --vllm
```

## Địa chỉ service

Sau khi khởi động thành công:

| Service | Địa chỉ |
| --- | --- |
| Web UI | `http://localhost:8010/` |
| Backend health | `http://localhost:8010/health` |
| Backend ready | `http://localhost:8010/ready` |
| vLLM models | `http://localhost:8006/v1/models` |
| Elasticsearch | `http://localhost:9201` |
| Qdrant | `http://localhost:6333` |

Kiểm tra nhanh:

```bash
curl http://localhost:9201
curl http://localhost:6333/collections
curl http://localhost:8006/v1/models
curl http://localhost:8010/health
curl http://localhost:8010/ready
```

Kết quả `/ready` hợp lệ có dạng:

```json
{
  "status": "ready",
  "checks": {
    "elasticsearch": {"ok": true},
    "qdrant": {"ok": true},
    "llm_server": {"ok": true},
    "model_paths": {
      "embedding": true,
      "reranker": true,
      "vncorenlp": true
    }
  }
}
```

## Chạy từng service riêng

### Elasticsearch và Qdrant

```bash
bash dbs/start_dbs.sh
```

### vLLM

Chạy foreground để xem log trực tiếp:

```bash
bash deploy/run_llm_service.sh vllm
```

Các biến thường dùng:

```bash
LLM_GPU_ID=1 \
LLM_PORT=8006 \
LLM_MODEL_PATH="$(pwd)/model_cache/Qwen3.5-9B" \
LLM_SERVED_MODEL_NAME=qwen3.5-9b \
VLLM_GPU_MEMORY_UTILIZATION=0.6 \
  bash deploy/run_llm_service.sh vllm
```

Không chạy lệnh này nếu port `8006` đã có vLLM đang hoạt động.

### FastAPI backend

Backend cần Elasticsearch, Qdrant và LLM endpoint hoạt động trước:

```bash
bash deploy/start_backend_api.sh
```

Chạy trực tiếp để debug, không preload retrieval resources:

```bash
AUTO_INIT_RETRIEVAL_RESOURCES=false \
WEB_PRELOAD_RESOURCES=false \
CHECKPOINTER_BACKEND=none \
BACKEND_PORT=8010 \
  python demo.py
```

## Restart và dừng service

Restart riêng backend, không restart vLLM:

```bash
bash deploy/restart_backend_api.sh
```

Dừng riêng backend:

```bash
bash deploy/stop_backend_api.sh
```

Dừng backend và vLLM:

```bash
bash deploy/stop_backend_stack.sh --vllm
```

Lưu ý: `stop_backend_stack.sh` không dừng Elasticsearch và Qdrant.

## Log và PID

Các file runtime được ghi vào:

```text
runtime/logs/legal-vllm-8006.log
runtime/logs/legal-rag-api-8010.log
runtime/pids/
```

Xem log:

```bash
tail -f runtime/logs/legal-vllm-8006.log
tail -f runtime/logs/legal-rag-api-8010.log
```

## Chuyển sang SGLang

Nếu môi trường đã cài SGLang:

```bash
bash deploy/start_backend_stack.sh --sglang
```

Hoặc chạy model server riêng:

```bash
bash deploy/run_llm_service.sh sglang
```

## Lỗi thường gặp

### Port đã được sử dụng

```bash
ss -ltnp | grep -E ':(6333|8006|8010|9201)'
```

Stack script tự bỏ qua service có endpoint đang hoạt động. Nếu process lỗi giữ port,
dừng đúng process trước khi chạy lại.

### Backend chạy nhưng `/ready` chưa thành công

Kiểm tra lần lượt:

```bash
curl http://localhost:9201
curl http://localhost:6333/collections
curl http://localhost:8006/v1/models
tail -n 160 runtime/logs/legal-rag-api-8010.log
```

### Không tìm thấy model

Các thư mục sau phải tồn tại:

```text
model_cache/Qwen3.5-9B/
model_cache/ViRanker/
model_cache/vietlegal-harrier-0.6b/
model_cache/vncorenlp/
```

### Hết VRAM

Giảm `VLLM_GPU_MEMORY_UTILIZATION`, giảm context length hoặc tách LLM và backend
sang hai GPU:

```bash
LLM_GPU_ID=1 BACKEND_GPU_ID=0 \
VLLM_GPU_MEMORY_UTILIZATION=0.5 \
  bash deploy/start_backend_stack.sh --vllm
```

### Qdrant đang mở public

`dbs/start_dbs.sh` mặc định bind Qdrant vào `127.0.0.1`. Nếu kiểm tra thấy Qdrant
listen trên `0.0.0.0:6333`, cần restart với host local hoặc chặn port bằng firewall
trước khi mở máy ra Internet.

## Systemd và Nginx

Các file trong `deploy/*.service` và `deploy/nginx-legal-assistant.conf` được copy
nguyên từ project gốc, nên vẫn chứa đường dẫn `/home/uet/cuongdam/Legal_assistant`.
Với bản nằm trong `src/main/`, nên dùng các shell script ở trên. Chỉ dùng systemd
sau khi đã cập nhật lại đường dẫn trong service file cho đúng máy triển khai.
