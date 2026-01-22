// PDF Viewer Module
const PDFViewer = {
    pdfDoc: null,
    currentPage: 1,
    totalPages: 0,
    zoom: 0.67, // Default zoom 67%
    pdfImages: [],
    fileCategory: null,
    
    init() {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        
        // Event listeners
        document.getElementById('file-upload').addEventListener('change', (e) => this.handleUpload(e));
        document.getElementById('prev-page').addEventListener('click', () => this.changePage(-1));
        document.getElementById('next-page').addEventListener('click', () => this.changePage(1));
        document.getElementById('zoom-out').addEventListener('click', () => this.changeZoom(-0.1));
        document.getElementById('zoom-in').addEventListener('click', () => this.changeZoom(0.1));
        document.getElementById('zoom-reset').addEventListener('click', () => this.resetZoom());
        
        // Update zoom display
        this.updateZoomDisplay();
    },
    
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
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const data = await response.json();
            window.uploadedFilename = data.filename;
            this.fileCategory = data.category;

            // Show file info
            const categoryLabel = {
                'pdf': 'PDF',
                'image': 'Image',
                'office': 'Office Document',
                'text': 'Text File'
            }[data.category] || 'Document';
            
            document.getElementById('file-info').textContent = `File: ${data.filename} (${categoryLabel})`;
            document.getElementById('file-info').classList.remove('hidden');
            document.getElementById('clear-btn').classList.remove('hidden');
            parseBtn.disabled = false;

            // Load preview for PDF and images
            if (data.category === 'pdf' || data.category === 'image') {
                await this.loadPDF(file);
            } else {
                // For office docs and text files, show placeholder
                this.showPlaceholder(categoryLabel);
            }
            
            uploadText.textContent = '📤 Upload File';

        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file: ' + error.message);
            uploadText.textContent = '📤 Upload File';
        }
    },
    
    showPlaceholder(type) {
        const viewer = document.getElementById('pdf-viewer');
        viewer.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
                <p style="color: #5D7B6F; font-weight: 500;">${type} Uploaded</p>
                <p style="font-size: 12px; margin-top: 8px;">Click "Parse Document" to analyze</p>
            </div>
        `;
    },
    
    async loadPDF(file) {
        const arrayBuffer = await file.arrayBuffer();
        this.pdfDoc = await pdfjsLib.getDocument(arrayBuffer).promise;
        this.totalPages = this.pdfDoc.numPages;
        this.currentPage = 1;

        // Pre-render all pages
        this.pdfImages = [];
        for (let i = 1; i <= this.totalPages; i++) {
            const page = await this.pdfDoc.getPage(i);
            const viewport = page.getViewport({ scale: 2 });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            await page.render({ canvasContext: context, viewport: viewport }).promise;
            this.pdfImages.push(canvas.toDataURL());
        }

        document.getElementById('pdf-controls').classList.remove('hidden');
        this.renderPage();
    },
    
    renderPage() {
        if (!this.pdfImages.length) return;

        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'canvas-container';

        const canvas = document.createElement('canvas');
        canvas.id = 'pdf-canvas';

        const highlightCanvas = document.createElement('canvas');
        highlightCanvas.id = 'highlight-canvas';

        canvasContainer.appendChild(canvas);
        canvasContainer.appendChild(highlightCanvas);

        const viewer = document.getElementById('pdf-viewer');
        viewer.innerHTML = '';
        viewer.appendChild(canvasContainer);

        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width * this.zoom;
            canvas.height = img.height * this.zoom;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            highlightCanvas.width = canvas.width;
            highlightCanvas.height = canvas.height;

            if (window.selectedElement) {
                this.drawHighlight(highlightCanvas, window.selectedElement);
            }
        };
        img.src = this.pdfImages[this.currentPage - 1];

        document.getElementById('page-info').textContent = `Page ${this.currentPage} / ${this.totalPages}`;
        document.getElementById('prev-page').disabled = this.currentPage === 1;
        document.getElementById('next-page').disabled = this.currentPage === this.totalPages;
    },
    
    changePage(delta) {
        const newPage = this.currentPage + delta;
        if (newPage < 1 || newPage > this.totalPages) return;
        this.currentPage = newPage;
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
        document.getElementById('zoom-info').textContent = Math.round(this.zoom * 100) + '%';
    },
    
    drawHighlight(canvas, element) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const bbox = element.grounding.bbox;
        const img = new Image();
        img.onload = () => {
            const scaleX = canvas.width / img.width;
            const scaleY = canvas.height / img.height;

            const x = bbox.left * scaleX;
            const y = bbox.top * scaleY;
            const w = (bbox.right - bbox.left) * scaleX;
            const h = (bbox.bottom - bbox.top) * scaleY;

            // Highlight with green theme
            ctx.strokeStyle = '#5D7B6F';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(164, 195, 162, 0.2)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);

            // ===== LABEL (TABLE / PARAGRAPH / IMAGE) =====
            const label = (element.type || "ELEMENT").toUpperCase();
            ctx.font = '12px Inter, sans-serif';
            
            const padding = 6;
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = '#5D7B6F';
            ctx.fillRect(
                x,
                y - 18,
                textWidth + padding * 2,
                18
            );

            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + padding, y - 5);

            };
            
            img.src = this.pdfImages[this.currentPage - 1];
    },
    
    navigateToElement(element) {
        if (!this.pdfImages.length) return;
        
        if (element.grounding.page + 1 !== this.currentPage) {
            this.currentPage = element.grounding.page + 1;
            this.renderPage();
        } else {
            const highlightCanvas = document.getElementById('highlight-canvas');
            if (highlightCanvas) {
                this.drawHighlight(highlightCanvas, element);
            }
        }
    }
};

// Initialize on load
window.PDFViewer = PDFViewer;