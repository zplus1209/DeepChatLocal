// ===============================
// Document Viewer Module
// ===============================
const PDFViewer = {
    pdfDoc: null,
    currentPage: 1,
    totalPages: 0,
    zoom: 0.67,
    pdfImages: [],
    fileCategory: null,

    // ===========================
    // INIT
    // ===========================
    init() {
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

        document.getElementById('file-upload')
            .addEventListener('change', (e) => this.handleUpload(e));

        document.getElementById('prev-page')
            .addEventListener('click', () => this.changePage(-1));

        document.getElementById('next-page')
            .addEventListener('click', () => this.changePage(1));

        document.getElementById('zoom-out')
            .addEventListener('click', () => this.changeZoom(-0.1));

        document.getElementById('zoom-in')
            .addEventListener('click', () => this.changeZoom(0.1));

        document.getElementById('zoom-reset')
            .addEventListener('click', () => this.resetZoom());

        this.updateZoomDisplay();
    },

    // ===========================
    // FILE UPLOAD
    // ===========================
    async handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const uploadText = document.getElementById('upload-text');
        const parseBtn = document.getElementById('parse-btn');

        uploadText.innerHTML = '<span class="loading"></span> Uploading...';
        parseBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload-file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Upload failed');
            }

            const data = await response.json();
            window.uploadedFilename = data.filename;
            this.fileCategory = data.category;

            const categoryLabel = {
                pdf: 'PDF',
                image: 'Image',
                office: 'Office Document',
                text: 'Text File'
            }[data.category] || 'Document';

            document.getElementById('file-info').textContent =
                `File: ${data.filename} (${categoryLabel})`;
            document.getElementById('file-info').classList.remove('hidden');
            document.getElementById('clear-btn').classList.remove('hidden');

            parseBtn.disabled = false;

            // ===== PREVIEW ROUTING =====
            switch (data.category) {
                case 'pdf':
                    await this.loadPDF(file);
                    break;
                case 'image':
                    await this.loadImage(file);
                    break;
                case 'text':
                    await this.loadText(file);
                    break;
                case 'office':
                    this.showOfficePlaceholder();
                    break;
                default:
                    this.showPlaceholder('Document');
            }

            uploadText.textContent = '📤 Upload File';

        } catch (err) {
            console.error(err);
            alert(err.message);
            uploadText.textContent = '📤 Upload File';
        }
    },

    // ===========================
    // PDF VIEWER
    // ===========================
    async loadPDF(file) {
        const buffer = await file.arrayBuffer();
        this.pdfDoc = await pdfjsLib.getDocument(buffer).promise;

        this.totalPages = this.pdfDoc.numPages;
        this.currentPage = 1;
        this.pdfImages = [];

        for (let i = 1; i <= this.totalPages; i++) {
            const page = await this.pdfDoc.getPage(i);
            const viewport = page.getViewport({ scale: 2 });

            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            await page.render({
                canvasContext: canvas.getContext('2d'),
                viewport
            }).promise;

            this.pdfImages.push(canvas.toDataURL());
        }

        document.getElementById('pdf-controls').classList.remove('hidden');
        this.renderPage();
    },

    renderPage() {
        if (this.fileCategory === 'image') {
            this.renderImage();
            return;
        }

        if (!this.pdfImages.length) return;

        const viewer = document.getElementById('pdf-viewer');
        viewer.innerHTML = '';

        const container = document.createElement('div');
        container.className = 'canvas-container';

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.onload = () => {
            canvas.width = img.width * this.zoom;
            canvas.height = img.height * this.zoom;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };

        img.src = this.pdfImages[this.currentPage - 1];
        container.appendChild(canvas);
        viewer.appendChild(container);

        document.getElementById('page-info').textContent =
            `Page ${this.currentPage} / ${this.totalPages}`;

        document.getElementById('prev-page').disabled = this.currentPage === 1;
        document.getElementById('next-page').disabled =
            this.currentPage === this.totalPages;
    },

    // ===========================
    // IMAGE VIEWER
    // ===========================
    async loadImage(file) {
        const viewer = document.getElementById('pdf-viewer');
        const url = URL.createObjectURL(file);

        viewer.innerHTML = `
            <div class="image-viewer">
                <img id="image-preview" src="${url}" />
            </div>
        `;

        this.pdfImages = [url];
        this.currentPage = 1;
        this.totalPages = 1;

        document.getElementById('pdf-controls').classList.remove('hidden');
        this.renderImage();
    },

    renderImage() {
        const img = document.getElementById('image-preview');
        if (!img) return;

        img.style.transform = `scale(${this.zoom})`;
        img.style.transformOrigin = 'top left';

        document.getElementById('page-info').textContent = 'Image';
    },

    // ===========================
    // TEXT / MD VIEWER
    // ===========================
    async loadText(file) {
        const text = await file.text();
        const viewer = document.getElementById('pdf-viewer');

        viewer.innerHTML = `
            <pre class="text-viewer">${this.escapeHTML(text)}</pre>
        `;

        document.getElementById('pdf-controls').classList.add('hidden');
    },

    escapeHTML(str) {
        return str.replace(/[&<>"']/g, c => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[c]);
    },

    // ===========================
    // OFFICE PLACEHOLDER
    // ===========================
    showOfficePlaceholder() {
        this.showPlaceholder('Office Document');
    },

    showPlaceholder(type) {
        const viewer = document.getElementById('pdf-viewer');
        viewer.innerHTML = `
            <div class="empty-state">
                <p style="font-weight:600">${type} uploaded</p>
                <p style="font-size:12px;margin-top:8px">
                    Preview not supported.<br>
                    Click <b>Parse Document</b> to analyze.
                </p>
            </div>
        `;
        document.getElementById('pdf-controls').classList.add('hidden');
    },

    // ===========================
    // CONTROLS
    // ===========================
    changePage(delta) {
        const p = this.currentPage + delta;
        if (p < 1 || p > this.totalPages) return;
        this.currentPage = p;
        this.renderPage();
    },

    changeZoom(delta) {
        this.zoom = Math.max(0.3, Math.min(2, this.zoom + delta));
        this.updateZoomDisplay();
        this.renderPage();
    },

    resetZoom() {
        this.zoom = 0.67;
        this.updateZoomDisplay();
        this.renderPage();
    },

    updateZoomDisplay() {
        document.getElementById('zoom-info').textContent =
            Math.round(this.zoom * 100) + '%';
    }
};

// ===============================
// INIT
// ===============================
window.PDFViewer = PDFViewer;
