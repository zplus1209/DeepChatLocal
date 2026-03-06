# DeepChat Local

RAG chat UI kết nối với backend FastAPI. Lấy cảm hứng từ [kotaemon](https://github.com/Cinnamon/kotaemon).

---

## Cấu trúc dự án

```
DeepChatLocal/
├── backend/          ← FastAPI + RAG (Python)
└── frontend/         ← React + Vite (Node.js)
```

---

## Yêu cầu

| Công cụ | Phiên bản |
|---------|-----------|
| Python  | >= 3.10   |
| Node.js | >= 18     |
| npm     | >= 9      |
| Ollama  | mới nhất (nếu dùng LLM local) |

---

## Bước 1 — Cài đặt Backend

```bash
cd backend

# Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt

# Cấu hình
cp .env.example .env
# Mở .env và chỉnh theo nhu cầu
```

### Cấu hình `.env` tối thiểu (dùng Ollama + Qdrant)

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_BASE_URL=http://localhost:11434

EMBEDDING_NAME=Alibaba-NLP/gte-multilingual-base
EMBEDDING_BACKEND=hf
EMBEDDING_DIM=768

DB_TYPE=qdrant
RETRIEVAL_MODE=hybrid
COLLECTION_NAME=deepchat
TOP_K=4

QDRANT_PATH=./qdrant_db
```

### Khởi động Ollama (nếu chưa chạy)

```bash
# Cài Ollama từ https://ollama.com
ollama pull llama3.2
ollama serve
```

### Chạy Backend

```bash
# Đảm bảo venv đang active
uvicorn main:app --reload --port 8000
```

Kiểm tra: http://localhost:8000/api/v1/health

---

## Bước 2 — Cài đặt Frontend

```bash
cd frontend

# Cài dependencies
npm install

# Chạy dev server
npm run dev
```

Mở trình duyệt: **http://localhost:3000**

---

## Bước 3 — Sử dụng

1. **Tải tài liệu**: Bấm tab "Files" ở sidebar trái → kéo thả hoặc click để upload PDF/DOCX/TXT
2. **Đặt câu hỏi**: Gõ câu hỏi vào ô nhập liệu bên dưới
3. **Xem nguồn**: Mỗi câu trả lời có nút "nguồn tham khảo" để xem đoạn văn bản được dùng
4. **Cài đặt RAG**: Bấm "Cài đặt" góc trên phải để bật/tắt Hybrid Search, Rerank, Reflection

---

## Build Production

```bash
# Frontend
cd frontend
npm run build
# Output: frontend/dist/

# Serve static từ FastAPI (thêm vào backend/main.py):
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")
```

---

## Docker Compose (tuỳ chọn)

```yaml
# docker-compose.yml
version: "3.9"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/qdrant_db
    env_file: ./backend/.env

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
```

---

## Các tính năng

| Tính năng | Mô tả |
|-----------|-------|
| Multi-conversation | Lưu nhiều cuộc trò chuyện trong sidebar |
| File upload | Upload PDF/DOCX/TXT trực tiếp từ UI |
| Hybrid RAG | Dense + Sparse search với RRF fusion |
| Rerank | CrossEncoder reranking sau retrieval |
| Reflection | Tự viết lại query từ lịch sử chat |
| Source citations | Hiển thị nguồn tham khảo cho mỗi câu trả lời |
| DB status | Hiển thị trạng thái kết nối backend ở sidebar |
| Settings drawer | Bật/tắt tính năng RAG không cần restart |
