from flask import Flask, request, render_template, send_file, jsonify, url_for
import os
import fitz  # PyMuPDF
import pandas as pd
import xml.etree.ElementTree as ET
import pyreadstat
import uuid
import tempfile
import shutil
import zipfile
import logging
import time
import traceback
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created directory: {UPLOAD_FOLDER}")

# ---------------- File Processing Functions ----------------

def int_to_rgb(color_val):
    if isinstance(color_val, int):
        b = (color_val & 0xff) / 255.0
        g = ((color_val >> 8) & 0xff) / 255.0
        r = ((color_val >> 16) & 0xff) / 255.0
        return (r, g, b)
    elif isinstance(color_val, (tuple, list)) and len(color_val) == 3:
        return tuple(c / 255.0 if c > 1 else c for c in color_val)
    return (0.0, 0.0, 0.0)

def replace_text_in_pdf(input_pdf_path, old_text, new_text):
    """
    Replaces text using the precise insert_text method and a standard font name.
    """
    try:
        with fitz.open(input_pdf_path) as doc:
            for page in doc:
                text_instances = page.search_for(old_text)
                if not text_instances:
                    continue

                all_text_info = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
                instance_properties = {}

                for rect in text_instances:
                    found_properties = False
                    for block in all_text_info:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                span_rect = fitz.Rect(span["bbox"])
                                if rect.intersects(span_rect) and old_text in span["text"]:
                                    instance_properties[rect.irect] = {
                                        "origin": span["origin"],
                                        "size": span["size"],
                                        "color": span["color"]
                                    }
                                    found_properties = True
                                    break
                            if found_properties: break
                        if found_properties: break
                    
                    if not found_properties:
                        instance_properties[rect.irect] = {"origin": rect.bl, "size": 11, "color": 0}

                    page.add_redact_annot(rect, fill=(1.0, 1.0, 1.0))
                
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                for rect in text_instances:
                    properties = instance_properties.get(rect.irect)
                    if properties:
                        page.insert_text(
                            properties["origin"],
                            new_text,
                            fontsize=properties["size"],
                            fontname="times-roman",
                            color=int_to_rgb(properties["color"])
                        )

            output_path = input_pdf_path.replace('.pdf', '_modified.pdf')
            doc.save(output_path, garbage=4, deflate=True, clean=True)
        
        logger.info(f"PDF processed successfully: {output_path}")
        return output_path
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error in replace_text_in_pdf: {str(e)}\n{tb_str}")
        raise

def replace_text_in_csv(input_csv_path, old_text, new_text):
    try:
        df = pd.read_csv(input_csv_path, dtype=str)
        df = df.applymap(lambda x: x.replace(old_text, new_text) if isinstance(x, str) else x)
        output_path = input_csv_path.replace('.csv', '_modified.csv')
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        logger.error(f"Error processing CSV {input_csv_path}: {str(e)}")
        raise

def replace_text_in_xml(input_xml_path, old_text, new_text):
    try:
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        def replace_in_element(elem):
            if elem.text and old_text in elem.text: elem.text = elem.text.replace(old_text, new_text)
            if elem.tail and old_text in elem.tail: elem.tail = elem.tail.replace(old_text, new_text)
            for k, v in elem.attrib.items():
                if old_text in v: elem.attrib[k] = v.replace(old_text, new_text)
            for child in elem: replace_in_element(child)
        replace_in_element(root)
        output_path = input_xml_path.replace('.xml', '_modified.xml')
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path
    except Exception as e:
        logger.error(f"Error processing XML {input_xml_path}: {str(e)}")
        raise

def replace_text_in_xpt(input_xpt_path, old_text, new_text):
    try:
        df, meta = pyreadstat.read_xport(input_xpt_path)
        df = df.applymap(lambda x: x.replace(old_text, new_text) if isinstance(x, str) else x)
        output_path = input_xpt_path.replace('.xpt', '_modified.xpt')
        pyreadstat.write_xport(df, output_path, file_format_version=8, table_name=meta.table_name)
        return output_path
    except Exception as e:
        logger.error(f"Error processing XPT {input_xpt_path}: {str(e)}")
        raise

def process_single_file(file_path, old_text, new_text):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf': return replace_text_in_pdf(file_path, old_text, new_text)
        elif ext == '.csv': return replace_text_in_csv(file_path, old_text, new_text)
        elif ext == '.xml': return replace_text_in_xml(file_path, old_text, new_text)
        elif ext == '.xpt': return replace_text_in_xpt(file_path, old_text, new_text)
        else: return None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def extract_zip_and_process(zip_path, old_text, new_text):
    extract_folder = os.path.join(UPLOAD_FOLDER, f"extracted_{uuid.uuid4()}")
    os.makedirs(extract_folder)
    processed_files = []
    supported_extensions = ['.pdf', '.csv', '.xml', '.xpt']
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    try:
                        processed = process_single_file(file_path, old_text, new_text)
                        if processed: processed_files.append(processed)
                    except Exception as e:
                        logger.error(f"Error processing {file} in ZIP: {str(e)}")
        if processed_files:
            output_zip = zip_path.replace('.zip', '_modified.zip')
            with zipfile.ZipFile(output_zip, 'w') as zipf:
                for pf in processed_files:
                    zipf.write(pf, os.path.basename(pf))
            return output_zip
        return None
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)

@app.route('/')
def index():
    return render_template('index.html', UPLOAD_URL=url_for('upload_file'))

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Document Text Replacer is running'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        old_text = request.form.get('old_text', '').strip()
        new_text = request.form.get('new_text', '').strip()
        if not old_text: return jsonify({'error': 'Text to find is required'}), 400
        
        uploaded_files = request.files.getlist('pdf_file')
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            return jsonify({'error': 'No files selected'}), 400
        
        processed_files_info = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            supported = ['.pdf', '.csv', '.xml', '.xpt', '.zip']
            if ext not in supported: return jsonify({'error': f'Unsupported file type: {ext}'}), 400
            
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            try:
                if ext == '.zip':
                    output_path = extract_zip_and_process(file_path, old_text, new_text)
                else:
                    output_path = process_single_file(file_path, old_text, new_text)
                
                if output_path:
                    processed_files_info.append({'path': output_path, 'name': f"modified_{filename}"})
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Error processing {filename}: {str(e)}\n{tb_str}")
                return jsonify({'error': f'Error processing {filename}: {str(e)}'}), 500
        
        if not processed_files_info:
            return jsonify({'error': 'No files were processed successfully'}), 400
        
        if len(processed_files_info) == 1:
            info = processed_files_info[0]
            return send_file(info['path'], as_attachment=True, download_name=info['name'])
        else:
            zip_filename = f"modified_files_{uuid.uuid4()}.zip"
            zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for info in processed_files_info:
                    zipf.write(info['path'], info['name'])
            return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Unexpected error in upload_file: {str(e)}\n{tb_str}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup_old_files():
    """Cleans up files in the uploads folder older than 1 hour."""
    now = time.time()
    cutoff = now - 3600
    files_deleted = 0
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                try:
                    os.remove(file_path)
                    files_deleted += 1
                    logger.info(f"Deleted old file: {filename}")
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        msg = f"Cleanup complete. Deleted {files_deleted} old file(s)."
        return jsonify({'status': 'success', 'message': msg}), 200
    except Exception as e:
        logger.error(f"Error during cleanup task: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'An internal server error occurred. Please try again.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting Document Text Replacer on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)