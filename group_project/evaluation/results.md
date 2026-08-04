# RAG Evaluation Results

- Thời điểm chạy: 2026-08-04T17:40:49+07:00
- Framework: **LLM-as-a-Judge**
- Pipeline: `/home/uet/cuongdam/K4-Day08-RAG-Pipeline-B1-1/src/main`
- Golden samples: **20**

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ A−B |
|---|---:|---:|---:|
| Faithfulness | 0.695 | 0.620 | +0.075 |
| Answer Relevance | 0.815 | 0.820 | -0.005 |
| Context Recall | 0.790 | 0.698 | +0.093 |
| Context Precision | 0.920 | 0.828 | +0.092 |
| LLM Judge Correctness | 0.662 | 0.575 | +0.088 |
| Average | 0.776 | 0.708 | +0.069 |

## A/B Comparison

- **Config A:** Config A — Elasticsearch + Qdrant (RRF) + ViRanker.
- **Config B:** Config B — Qdrant dense-only, không ViRanker.
- Mean latency A/B: 234.56s / 197.10s.
- **Kết luận:** Config A có điểm trung bình cao hơn (0.776 so với 0.708).

## Worst Performers (Bottom 3)

| # | Config | Question | Average | Failure Stage | Root Cause |
|---:|---|---|---:|---|---|
| 1 | dense_only | Báo cáo tài chính của công ty cần cung cấp những thông tin cơ bản nào về tình hình tài chính và kin… | 0.000 | both | Câu trả lời hoàn toàn sai lệch so với câu hỏi và ngữ cảnh. Câu hỏi yêu cầu liệt kê các thông tin tài chính cụ thể (tài sản, nợ, vốn, doanh thu...) theo Thông tư 133/2016/TT-BTC (c… |
| 2 | hybrid_rerank | Thời hạn để cơ quan quản lý công trình thủy lợi cho ý kiến phê duyệt kết quả thẩm định báo cáo đánh… | 0.200 | generation | Câu trả lời sai hoàn toàn so với Gold Answer và Context. Gold Answer chỉ ra thời hạn là 03 ngày (theo TT 09/2026 sửa đổi TT 02/2022), trong khi hệ thống trả lời là 05 ngày làm việ… |
| 3 | dense_only | Việc từ chối kết quả trúng đấu giá tài sản của Bộ Quốc phòng được quy định như thế nào? | 0.200 | generation | Câu trả lời hoàn toàn sai lệch so với Gold Answer và Context. Hệ thống đã trích dẫn sai điều luật (Điều 51, 52 thay vì Điều 28, 39 của Thông tư 126/2020/TT-BQP như yêu cầu), đưa r… |

## Recommendations

1. Siết grounding/citation ở reasoning prompt và từ chối kết luận khi thiếu căn cứ trực tiếp.
2. Bổ sung kiểm tra con số, thời hạn, điều kiện và ngoại lệ so với điều khoản trước khi trả lời.

## Reproduce

```bash
cd /home/uet/cuongdam/K4-Day08-RAG-Pipeline-B1-1
LEGAL_ASSISTANT_ENV_FILE=/path/to/legal_assistant.env \
  /home/uet/miniconda3/envs/gemma/bin/python group_project/evaluation/eval_pipeline.py
```
