// =============================
// Chatbot Module
// =============================
const Chatbot = {
    messages: [],
    
    init() {
        // Send button
        document.getElementById('chat-send-btn')
            .addEventListener('click', () => this.sendMessage());
        
        // Enter to send
        document.getElementById('chat-input')
            .addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        
        // Clear chat
        document.getElementById('clear-chat-btn')
            .addEventListener('click', () => this.clearChat());
        
        // Auto-resize textarea
        document.getElementById('chat-input')
            .addEventListener('input', (e) => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            });
    },
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message
        this.addMessage('user', message);
        input.value = '';
        input.style.height = 'auto';
        
        // Disable input while processing
        const sendBtn = document.getElementById('chat-send-btn');
        input.disabled = true;
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<span class="loading"></span>';
        
        try {
            // Call chatbot API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: window.uploadedFilename,
                    method: window.DeepDoc.currentMethod,
                    message: message,
                    history: this.messages.slice(-10) // Last 10 messages for context
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Chat failed');
            }
            
            // Add assistant response
            this.addMessage('assistant', data.response);
            
            // If references are provided, show them
            if (data.references && data.references.length > 0) {
                this.highlightReferences(data.references);
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            input.focus();
        }
    },
    
    addMessage(role, content) {
        this.messages.push({ role, content });
        
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';
        
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    },
    
    highlightReferences(references) {
        // Switch to analysis tab and highlight referenced chunks
        document.querySelector('[data-tab="analysis"]').click();
        
        // Find and highlight the first reference
        if (references.length > 0 && window.DeepDoc.parsedData) {
            const chunk = window.DeepDoc.parsedData.chunks.find(
                c => c.id === references[0]
            );
            if (chunk) {
                window.selectedElement = chunk;
                window.PDFViewer.navigateToElement(chunk);
            }
        }
    },
    
    clearChat() {
        if (!confirm('Clear all chat messages?')) return;
        
        this.messages = [];
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="chat-message assistant">
                <div class="chat-avatar">🤖</div>
                <div class="chat-bubble">
                    Hello! I'm your document assistant. Upload and parse a document, then ask me questions about it.
                </div>
            </div>
        `;
    }
};

window.Chatbot = Chatbot;