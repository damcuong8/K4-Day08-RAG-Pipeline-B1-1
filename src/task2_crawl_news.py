"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=1873&l=Nghiencuutraodoi",
    "https://thuvienphapluat.vn/banan/tin-tuc/cach-vien-dan-van-ban-quy-pham-phap-luat-moi-nhat-theo-nghi-dinh-782025-14807.html",
    "https://vdb.gov.vn/tin-tuc/17374/luat-%E2%80%9Cban-hanh-van-ban-quy-pham-phap-luat%E2%80%9D.aspx",
    "https://isos.gov.vn/tin-hoat-dong/diem-tin/7-diem-moi-cua-luat-ban-hanh-van-ban-quy-pham-phap-luat-2025-43625.html",
    "https://luatvietnam.vn/tu-phap/thong-tu-26-2025-tt-btp-huong-dan-xay-dung-ban-hanh-van-ban-quy-pham-phap-luat-422980-d1.html",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết/văn bản hướng dẫn pháp luật và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    date_crawled = datetime.now().isoformat()
    title = "Văn bản Hướng dẫn Pháp luật Việt Nam"
    content_markdown = ""

    skip_keywords = [
        "bản quyền thuộc", "giấy phép số", "xem với cỡ chữ", "về đầu trang", "các tin mới", 
        "các tin đã đưa", "tìm theo ngày", "thông tin mới", "tin nổi bật", "liên kết website", 
        "hỏi đáp", "danh mục", "kỷ yếu", "thư điện tử", "bài viết này có hữu ích không", 
        "nội dung nêu trên là phần giải đáp", "điều khoản được áp dụng có thể đã hết hiệu lực"
    ]

    # 1. Thử HTTP request + BeautifulSoup với headers giả lập browser chuẩn
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        }
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")

            # Loại bỏ script, style, nav, header, footer, iframe
            for tag in soup(["script", "style", "header", "footer", "nav", "iframe"]):
                tag.decompose()

            h1 = (
                soup.find("h1", class_=["title", "detail-title", "post-title", "news-title"])
                or soup.find("h1")
                or soup.find("h2")
            )
            if h1:
                title_text = h1.get_text().strip()
                if title_text and len(title_text) > 5:
                    title = title_text

            main_node = (
                soup.find("div", class_="Around_News_Content")
                or soup.find("div", class_="content-detail")
                or soup.find("div", class_="content-news")
                or soup.find("div", class_="the-content")
                or soup.find("div", class_="article-detail")
                or soup.find("main")
                or soup.body
            )

            if main_node:
                raw_text = main_node.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 20]
                clean_lines = []
                cutoff_phrases = [
                    "xem thêm các bài viết liên quan", "chuyên viên pháp lý", "bài viết này có hữu ích không",
                    "nội dung nêu trên là phần giải đáp", "các tin mới", "các tin đã đưa", "tìm theo ngày",
                    "thông tin mới", "tin nổi bật", "liên kết website", "cách nộp gia hạn", "lịch khai giảng",
                    "tải phần mềm htkk", "lưu ý: doanh nghiệp trả lương", "công bố thủ tục xử lý"
                ]

                for line in lines:
                    l_lower = line.lower()
                    if any(c in l_lower for c in cutoff_phrases):
                        break
                    if line not in clean_lines and not any(k in l_lower for k in skip_keywords):
                        clean_lines.append(line)

                if clean_lines:
                    import textwrap
                    wrapped_lines = []
                    for line in clean_lines:
                        if len(line) > 100 and not line.startswith("#"):
                            wrapped_lines.append(textwrap.fill(line, width=100, replace_whitespace=False))
                        else:
                            wrapped_lines.append(line)
                    content_markdown = f"# {title}\n\nURL nguồn: {url}\n\n" + "\n\n".join(wrapped_lines)

    except Exception as err:
        print(f"  [WARNING] HTTP request exception: {type(err).__name__}")

    # 2. Nếu HTTP request không lấy đủ nội dung, dùng Crawl4AI
    if len(content_markdown.strip()) < 300:
        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                if result and result.success and result.markdown:
                    content_markdown = result.markdown
                    if hasattr(result, "metadata") and result.metadata:
                        title = result.metadata.get("title") or title
        except Exception as e:
            print(f"  [WARNING] Crawl4AI exception ({type(e).__name__})")

    return {
        "url": url,
        "title": title,
        "date_crawled": date_crawled,
        "content_markdown": content_markdown,
    }







async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")



if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
