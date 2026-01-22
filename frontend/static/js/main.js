// =============================
// Main Application Module
// =============================
const App = {
    init() {
        // Initialize modules
        window.PDFViewer.init();
        window.DeepDoc.init();
        window.Chatbot.init();
        
        // Setup clear button
        document.getElementById('clear-btn')
            .addEventListener('click', () => this.handleClear());
        
        // Tab switching
        this.initTabs();
        
        // Check for existing files on load
        this.checkExistingFiles();
    },
    
    initTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });
    },
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-tab') === tabName);
        });
        
        // Update tab content
        if (tabName === 'analysis') {
            document.getElementById('analysis-header').classList.add('active');
            document.getElementById('chatbot-header').classList.remove('active');
            document.getElementById('analysis-content').classList.add('active');
            document.getElementById('chatbot-content').classList.remove('active');
        } else if (tabName === 'chatbot') {
            document.getElementById('analysis-header').classList.remove('active');
            document.getElementById('chatbot-header').classList.add('active');
            document.getElementById('analysis-content').classList.remove('active');
            document.getElementById('chatbot-content').classList.add('active');
        }
    },
    
    async checkExistingFiles() {
        try {
            const response = await fetch('/api/list-files');
            const data = await response.json();
            
            if (data.success && data.files.length > 0) {
                console.log('Found existing files:', data.files);
                // Could show a notification or file list here
            }
        } catch (error) {
            console.error('Error checking files:', error);
        }
    },
    
    async handleClear() {
        if (!window.uploadedFilename) return;
        
        if (!confirm('Are you sure you want to delete this file and its results?')) {
            return;
        }

        try {
            const response = await fetch(`/api/delete-file/${window.uploadedFilename}`, { 
                method: 'DELETE' 
            });
            
            if (response.ok) {
                location.reload();
            } else {
                throw new Error('Failed to delete file');
            }
        } catch (error) {
            console.error('Delete error:', error);
            alert('Failed to delete file: ' + error.message);
        }
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// Global variables
window.uploadedFilename = '';
window.selectedElement = null;