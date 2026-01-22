// =============================
// Deep Doc Module
// =============================
const DeepDoc = {
    parsedData: null,
    currentMethod: 'paddleocrvl',

    // =============================
    // INIT
    // =============================
    init() {
        document.getElementById('settings-toggle')
            .addEventListener('click', () => {
                document.getElementById('settings-panel')
                    .classList.toggle('hidden');
            });

        document.getElementById('use-vl-backend')
            .addEventListener('change', (e) => {
                document.getElementById('vl-settings')
                    .classList.toggle('hidden', !e.target.checked);
            });

        document.getElementById('parse-btn')
            .addEventListener('click', () => this.handleParse());

        document.getElementById('query-btn')
            .addEventListener('click', () => this.handleQuery());

        document.getElementById('query-input')
            .addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleQuery();
            });

        document.querySelectorAll('.type-filter')
            .forEach(cb => {
                cb.addEventListener('change', () => this.handleQuery());
            });
    },

    // =============================
    // PARSE
    // =============================
    async handleParse() {
        if (!window.uploadedFilename) return;

        const btn = document.getElementById('parse-btn');
        btn.innerHTML = '<span class="loading"></span> Parsing...';
        btn.disabled = true;

        try {
            const method = document.getElementById('method-select').value;
            const lang = document.getElementById('lang-select').value;
            const useVLBackend = document.getElementById('use-vl-backend').checked;

            this.currentMethod = method;

            const body = { filename: window.uploadedFilename, method, lang };

            if (useVLBackend) {
                body.vl_rec_backend =
                    document.getElementById('vl-backend-select').value;
                body.vl_rec_server_url =
                    document.getElementById('vl-server-url').value;
            }

            const res = await fetch('/api/parse-document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            const result = await res.json();
            if (!res.ok) throw new Error(result.error || 'Parse failed');

            this.parsedData = result.data;

            document.getElementById('query-panel').classList.remove('hidden');
            document.getElementById('query-btn').classList.remove('hidden');
            document.getElementById('query-btn').disabled = false;

            // Enable chatbot
            document.getElementById('chat-input').disabled = false;
            document.getElementById('chat-send-btn').disabled = false;

            this.renderChunks(this.parsedData.chunks);

            alert(`Parsed successfully: ${result.data.total_chunks} elements`);
        } catch (err) {
            console.error(err);
            alert(err.message);
        } finally {
            btn.innerHTML = '🔄 Parse Document';
            btn.disabled = false;
        }
    },

    // =============================
    // QUERY
    // =============================
    async handleQuery() {
        if (!this.parsedData) return;

        const query =
            document.getElementById('query-input').value.trim();

        const types =
            Array.from(document.querySelectorAll('.type-filter:checked'))
                .map(c => c.value);

        const res = await fetch('/api/deepdoc-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: window.uploadedFilename,
                method: this.currentMethod,
                query,
                chunk_types: types
            })
        });

        const result = await res.json();
        if (!res.ok) {
            alert(result.error || 'Query failed');
            return;
        }

        this.renderChunks(result.data.chunks);
    },

    // =============================
    // 📌 CORE RENDER (FIX CHUẨN)
    // =============================
    renderChunks(chunks) {
        const content = document.getElementById('analysis-content');

        if (!chunks || !chunks.length) {
            content.innerHTML =
                `<div class="info-banner">No results</div>`;
            return;
        }

        let html = `
            <div class="info-banner">
                <strong>${chunks.length}</strong>
                elements found – click to highlight PDF
            </div>
        `;

        chunks.forEach(chunk => {
            const selected =
                window.selectedElement?.id === chunk.id;

            // ===== IMAGE PATH FIX =====
            const imageHtml =
                chunk.grounding?.image_path
                    ? `
                        <img
                            class="element-image"
                            src="/api/images/${this.getRelativePath(chunk.grounding.image_path)}"
                            alt="element image"
                            onerror="this.style.display='none'"
                        />
                      `
                    : '';

            // ===== BODY =====
            const cleaned = this.stripAnchor(chunk.markdown);

            const body =
                chunk.type === 'table'
                    ? cleaned                  // table → raw HTML
                    : this.escapeHtml(cleaned.replace(/<[^>]+>/g, '')); // text → escape

            html += `
                <div class="element-card ${selected ? 'selected' : ''}"
                     data-id="${chunk.id}">
                    
                    <div class="element-header">
                        <span class="element-type type-${chunk.type}">
                            ${chunk.type}
                        </span>
                    </div>

                    ${imageHtml}

                    <div class="element-content">
                        ${body}
                    </div>

                    <div class="element-footer">
                        Page ${chunk.grounding.page + 1}
                    </div>
                </div>
            `;
        });

        content.innerHTML = html;

        // ===== CLICK → PDF HIGHLIGHT =====
        document.querySelectorAll('.element-card')
            .forEach(card => {
                card.addEventListener('click', () => {
                    const el =
                        chunks.find(c => c.id === card.dataset.id);
                    if (!el) return;

                    window.selectedElement = el;

                    document.querySelectorAll('.element-card')
                        .forEach(c =>
                            c.classList.toggle('selected', c === card)
                        );

                    window.PDFViewer.navigateToElement(el);
                });
            });
    },

    // =============================
    // 🧹 REMOVE <a id="..."></a> EVERYWHERE
    // =============================
    stripAnchor(md) {
        if (!md) return '';
        return md
            // 1️⃣ remove full anchor tags (normal HTML)
            .replace(/<a\b[^>]*>[\s\S]*?<\/a>/gi, '')
            // 2️⃣ remove orphan closing </a> or broken </a
            .replace(/<\/a\s*>?/gi, '')
            // 3️⃣ remove orphan opening <a ...>
            .replace(/<a\b[^>]*>?/gi, '')
            // 4️⃣ fix markdown heading glued after anchor
            .replace(/(^|\n)\s*#+\s*/g, '$1## ')
            // 5️⃣ clean excessive whitespace
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    },

    // =============================
    // 📝 SAFE TEXT
    // =============================
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    },

    // =============================
    // 🧩 IMAGE PATH HELPER - FIX ĐƯỜNG DẪN
    // =============================
    getRelativePath(path) {
        if (!path) return '';
        
        // Convert absolute path to relative path for API
        // Format: /path/to/output/method/file_stem/page_X/image.png
        // Extract: method/file_stem/page_X/image.png
        
        const parts = path.split('/output/');
        if (parts.length > 1) {
            return parts[1];
        }
        
        // Fallback: extract from last few segments
        const segments = path.split('/');
        const outputIndex = segments.indexOf('output');
        if (outputIndex !== -1 && outputIndex < segments.length - 1) {
            return segments.slice(outputIndex + 1).join('/');
        }
        
        return path;
    }
};

// expose
window.DeepDoc = DeepDoc;