# DeepChatLocal

---

## Luồng xử lý tách biệt (chat vs upload)

- `POST /api/v1/chat`: chỉ dùng để hỏi đáp.
  - Nếu `use_rag=true`: truy vấn DB vector rồi trả lời theo ngữ cảnh.
  - Nếu `use_rag=false`: gọi LLM trực tiếp, không truy vấn DB.
- `POST /api/v1/ingest/file` và `POST /api/v1/ingest/files`: chỉ dùng để nạp tài liệu vào DB.

Thiết kế này giúp tách rõ: **hỏi câu hỏi** ≠ **xử lý file**.

---

## Multi-file ingest (tham khảo workflow kiểu kreuzberg)

Đã hỗ trợ endpoint batch:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/files"   -F "files=@/path/doc1.pdf"   -F "files=@/path/doc2.docx"
```

Backend sẽ:
1. Parse từng file.
2. Chuyển về markdown.
3. Chunk văn bản theo cửa sổ chồng lấp.
4. Ingest nhiều chunk vào vector DB kèm metadata (`source`, `chunk`, `chunks`).

---

## Frontend định hướng theo kotaemon

- Sidebar có tab `Chats` và `Files` tách biệt.
- Upload file qua tab `Files` (hoặc nút đính kèm), sau đó mới hỏi đáp trong ô chat.
- Batch upload nhiều file trong `Files` tab sẽ gọi endpoint `/ingest/files`.

---

## Chạy để người khác test (mở cổng)

### 1) Chạy backend cho mạng LAN

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Chạy frontend cho mạng LAN

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

### 3) Người khác truy cập

- Cùng mạng nội bộ: dùng IP máy bạn, ví dụ:
  - `http://192.168.1.10:3000`
  - API: `http://192.168.1.10:8000/api/v1/health`

### 4) Nếu có firewall

Mở cổng `3000` (frontend) và `8000` (backend).

### 5) Test nhanh từ máy khác

```bash
curl http://<YOUR_IP>:8000/api/v1/health
```


---

## Mapping chunk theo file/page/line (chunkings folder)

Khi ingest file, backend tạo thư mục `chunkings/` trong parser output của file, gồm:

- `<chunk_id>.json`: metadata chi tiết từng chunk
- `<file_stem>_chunk_index.json`: index tất cả chunk của file

Metadata chunk gồm:
- `file` (tên file gốc)
- `page`
- `line_start`, `line_end`
- `section_type` (`text` | `image` | `table`)
- `items` (bbox + line range theo item)
- `chunk_id`

Nhờ đó khi LLM trả lời có thể map lại nguồn chính xác theo file/trang/dòng.
