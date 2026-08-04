"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    from markitdown import MarkItDown
    import fitz

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting legal doc: {filepath.name}")
            content_text = ""
            try:
                result = md.convert(str(filepath))
                if result and result.text_content:
                    content_text = result.text_content.strip()
            except Exception as exc:
                print(f"  [WARNING] MarkItDown error for {filepath.name}: {exc}")

            # Thử OCR với PyMuPDF (fitz) nếu MarkItDown không đọc được text (PDF scanned image)
            if len(content_text) < 100 and filepath.suffix.lower() == ".pdf":
                try:
                    doc = fitz.open(filepath)
                    ocr_pages = []
                    print(f"  [INFO] Đang OCR {len(doc)} trang cho {filepath.name}...")
                    for i, page in enumerate(doc):
                        try:
                            tp = page.get_textpage_ocr(language="eng+vie", full=True)
                            p_text = tp.extractTEXT() or ""
                        except Exception:
                            p_text = page.get_text() or ""
                        if p_text.strip():
                            ocr_pages.append(f"## Trang {i+1}\n\n{p_text.strip()}")
                    if ocr_pages:
                        content_text = f"# {filepath.stem}\n\n" + "\n\n".join(ocr_pages)
                except Exception as e:
                    print(f"  [WARNING] PyMuPDF OCR error: {e}")

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content_text, encoding="utf-8")
            print(f"  [OK] Saved: {output_path} ({len(content_text)} chars)")




def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + str(
                data.get("content_markdown") or data.get("content") or ""
            ).strip()
            output_path.write_text(content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[DONE] Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()

