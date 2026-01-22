import json
import traceback
from pathlib import Path
from asyncio import run as asyncio_run

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from parser.paddleocr_parser import PaddleOCRParser
from src.summarizer import ContentSummarizer
from src.multi_retriever import MultiRetriever
from src.rag.core import MultimodalRAG

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / "data/uploads"
OUTPUT_FOLDER = BASE_DIR / "output"
FRONTEND_DIR = BASE_DIR / "frontend"
PDF_PREVIEW_FOLDER = BASE_DIR / "data/pdfs/libreoffice_output"
PDF_PREVIEW_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    # PDF
    "pdf",
    # Images
    "png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp",
    # Office documents
    "doc", "docx", "ppt", "pptx", "xls", "xlsx",
    # Text
    "txt", "md"
}

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)

CORS(app) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

parser = PaddleOCRParser()
summarizer = ContentSummarizer()
multi_retriever = MultiRetriever()
rag_chain = MultimodalRAG(multi_retriever)

async def indexing(chunks, output_path):
    text_elements = []
    table_elements = []
    image_elements = []

    for c in chunks:
        if c["type"] == "text":
            text_elements.append(c)
        elif c["type"] == "table":
            table_elements.append(c)
        elif c["type"] == "image":
            image_elements.append(c)
            

    text_summaries, final_summary = await summarizer.asummarize_documents(
        text_elements=text_elements,
        output_path=output_path,
        detail=1,
        summarize_recursively=True,
    )
    
    table_summaries = await summarizer.asummarize_table_elements(
        table_elements,
        output_path=output_path,
    )

    image_summaries = await summarizer.asummarize_image_elements(
        image_elements
    )
    
    multi_retriever.add_all_documents(
        text_summaries, text_elements,
        table_summaries, table_elements,
        image_summaries, image_elements
    )
    
    return {
        "text": len(text_summaries),
        "table": len(table_summaries),
        "image": len(image_summaries),
        "final_summary": final_summary
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_category(filename):
    """Categorize file type"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if ext == 'pdf':
        return 'pdf'
    elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'gif', 'webp']:
        return 'image'
    elif ext in ['doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx']:
        return 'office'
    elif ext in ['txt', 'md']:
        return 'text'
    return 'unknown'

@app.route("/")
def index():
    """Serve frontend UI"""
    return render_template("index.html")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload file endpoint - supports PDF, images, Office docs, and text files"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported formats: PDF, Images (PNG, JPG, etc.), Office (DOC, DOCX, PPT, PPTX, XLS, XLSX), Text (TXT, MD)'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        # Get file category
        category = get_file_category(filename)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': str(filepath),
            'category': category
        })
    
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
@app.route('/api/parse-document', methods=['POST'])
def parse_document():
    """Parse document with PaddleOCR - Step 1: Analysis"""
    try:        
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        file_stem = Path(filename).stem
        method = data.get('method', 'paddleocrvl')
        lang = data.get('lang', 'vi')
        
        # Get additional parameters
        vl_rec_backend = data.get('vl_rec_backend')
        vl_rec_server_url = data.get('vl_rec_server_url')
        
        filepath = app.config['UPLOAD_FOLDER'] / filename
        
        if not filepath.exists():
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        output_path = app.config['OUTPUT_FOLDER'] / method / file_stem
        
        # Prepare kwargs
        parse_kwargs = {
            'method': method,
            'lang': lang,
            'output_path': output_path
        }
        
        # Add VL recognition settings if provided
        if vl_rec_backend and vl_rec_server_url:
            parse_kwargs['vl_rec_backend'] = vl_rec_backend
            parse_kwargs['vl_rec_server_url'] = vl_rec_server_url
        
        # Parse document (supports PDF, images, office docs)
        print(f"Parsing {filename} with method {method}...")
        result = parser.parse_document(str(filepath), **parse_kwargs)
        
        print(result.get("chunks", []))
        chunks = result.get("chunks", [])
        type_chunks = chunks[1:4]
        summary_info = asyncio_run(indexing(type_chunks, output_path))
        
        # Return result with metadata
        return jsonify({
            'success': True,
            'data': {
                'markdown': result.get('markdown', ''),
                'chunks': result.get('chunks', []),
                'total_chunks': len(result.get('chunks', [])),
                'method': method,
                'lang': lang,
                'rag_index': summary_info
            },
            'message': f'Document parsed successfully with {method}'
        })
    
    except Exception as e:
        print(f"Error parsing document: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Parsing failed: {str(e)}'}), 500

@app.route('/api/deepdoc-query', methods=['POST'])
def deepdoc_query():
    """Deep Doc Query - Step 2: Extraction based on user request"""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        query = data.get('query', '')
        chunk_types = data.get('chunk_types', [])
        
        # Get cached parsing result - FIX: Correct path structure
        file_stem = Path(filename).stem
        method = data.get('method', 'paddleocrvl')
        
        # Correct path: output/method/file_stem/file_stem_content_list.json
        json_file = app.config['OUTPUT_FOLDER'] / method / file_stem / f"{file_stem}_content_list.json"
        md_file = app.config['OUTPUT_FOLDER'] / method / file_stem / f"{file_stem}.md"
        
        print(f"Looking for JSON file: {json_file}")
        print(f"Looking for MD file: {md_file}")
        
        if not json_file.exists() or not md_file.exists():
            return jsonify({'error': 'Document not parsed yet. Please parse first.'}), 404
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        with open(md_file, 'r', encoding='utf-8') as f:
            markdown = f.read()
        
        # Filter chunks by type if specified
        filtered_chunks = chunks
        if chunk_types:
            filtered_chunks = [c for c in chunks if c.get('type') in chunk_types]
        
        # Simple query matching
        if query:
            query_lower = query.lower()
            filtered_chunks = [
                c for c in filtered_chunks 
                if query_lower in c.get('markdown', '').lower()
            ]
        
        return jsonify({
            'success': True,
            'data': {
                'chunks': filtered_chunks,
                'total_results': len(filtered_chunks),
                'query': query,
                'filters': chunk_types
            }
        })
    
    except Exception as e:
        print(f"Error in deepdoc query: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint - Answer questions about the document"""
    try:        
        data = request.get_json()
        
        if not data or 'filename' not in data or 'message' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        filename = data['filename']
        message = data['message']
        history = data.get('history', [])
        
        print(f"Data: {data}")
        result = rag_chain.invoke(message)
        response = result.get("answer", "")
        references = result.get("context", {})
        
        print(result)
        
        return jsonify({
            'success': True,
            'response': response,
            'references': references
        })
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500
    
@app.route('/api/get-result/<filename>', methods=['GET'])
def get_result(filename):
    """Get cached parsing result"""
    try:
        file_stem = Path(filename).stem
        
        # Try to find result in output folder
        for method in ['ppstructurev3', 'paddleocrvl']:
            json_file = app.config['OUTPUT_FOLDER'] / method / file_stem / f"{file_stem}_content_list.json"
            md_file = app.config['OUTPUT_FOLDER'] / method / file_stem / f"{file_stem}.md"
            
            if json_file.exists() and md_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown = f.read()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'markdown': markdown,
                        'chunks': chunks,
                        'total_chunks': len(chunks)
                    },
                    'method': method
                })
        
        return jsonify({'error': 'No cached result found'}), 404
    
    except Exception as e:
        print(f"Error getting result: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Failed to get result: {str(e)}'}), 500
    
@app.route('/api/images/<path:filepath>', methods=['GET'])
def serve_image(filepath):
    """Serve extracted images - FIX: Use correct path structure"""
    try:
        # filepath should be: method/file_stem/page_X/image.png
        image_path = app.config['OUTPUT_FOLDER'] / filepath
        
        print(f"Serving image from: {image_path}")
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return jsonify({'error': 'Image not found'}), 404
        
        return send_from_directory(
            image_path.parent,
            image_path.name,
            mimetype='image/png'
        )
    
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return jsonify({'error': f'Failed to serve image: {str(e)}'}), 500

@app.route("/api/preview/pdf/<path:filename>", methods=["GET"])
def preview_pdf(filename):
    """Serve converted PDF for preview"""
    try:
        pdf_path = PDF_PREVIEW_FOLDER / filename

        if not pdf_path.exists():
            return jsonify({"error": "PDF preview not found"}), 404

        return send_from_directory(
            pdf_path.parent,
            pdf_path.name,
            mimetype="application/pdf",
            as_attachment=False
        )

    except Exception as e:
        print("Error serving preview PDF:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/list-files', methods=['GET'])
def list_files():
    """List all uploaded and processed files"""
    try:
        uploaded_files = []
        
        if app.config['UPLOAD_FOLDER'].exists():
            for file in app.config['UPLOAD_FOLDER'].iterdir():
                if file.is_file() and allowed_file(file.name):
                    file_stem = file.stem
                    processed = False
                    parsing_info = {}
                    
                    for method in ['ppstructurev3', 'paddleocrvl']:
                        result_dir = app.config['OUTPUT_FOLDER'] / method / file_stem
                        json_file = result_dir / f"{file_stem}_content_list.json"
                        
                        if result_dir.exists() and json_file.exists():
                            processed = True
                            with open(json_file, 'r', encoding='utf-8') as f:
                                chunks = json.load(f)
                                parsing_info[method] = {
                                    'total_chunks': len(chunks),
                                    'chunk_types': list(set(c.get('type') for c in chunks))
                                }
                    
                    uploaded_files.append({
                        'filename': file.name,
                        'size': file.stat().st_size,
                        'category': get_file_category(file.name),
                        'processed': processed,
                        'parsing_info': parsing_info
                    })
        
        return jsonify({
            'success': True,
            'files': uploaded_files
        })
    
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@app.route('/api/delete-file/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete uploaded file and its results"""
    try:
        # Delete uploaded file
        filepath = app.config['UPLOAD_FOLDER'] / filename
        if filepath.exists():
            filepath.unlink()
        
        # Delete output files
        file_stem = Path(filename).stem
        for method in ['ppstructurev3', 'paddleocrvl']:
            result_dir = app.config['OUTPUT_FOLDER'] / method / file_stem
            if result_dir.exists():
                import shutil
                shutil.rmtree(result_dir)
        
        return jsonify({
            'success': True,
            'message': f'File {filename} deleted successfully'
        })
    
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Document Parser Server Starting...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER'].resolve()}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER'].resolve()}")
    print("Supported formats: PDF, Images, Office (DOC/DOCX/PPT/PPTX/XLS/XLSX), Text (TXT/MD)")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )