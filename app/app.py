import os
import sys
import uuid
import shutil
import time
import json
from threading import Lock

# Add the src directory to Python path to use local development version
current_file_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_file_dir, '..', 'src'))
sys.path.insert(0, src_path)

from flask import Flask, request, render_template, send_file, after_this_request, Response
from werkzeug.utils import secure_filename
from mzrt2h5 import save_dataset_as_sparse_h5, save_single_mzml_as_sparse_h5

app = Flask(__name__)
# Log the src path after app is created
app.logger.info(f"Added src directory to Python path: {src_path}")
app.logger.info(f"Local mzrt2h5 module exists at: {os.path.exists(os.path.join(src_path, 'mzrt2h5'))}")
# Use absolute path for uploads to avoid working directory issues
current_file_dir = os.path.dirname(__file__)
app.logger.info(f"Current file directory (__file__): {__file__}")
app.logger.info(f"os.path.dirname(__file__): {current_file_dir}")

app.config['UPLOAD_FOLDER'] = os.path.join(current_file_dir, 'uploads')
app.logger.info(f"UPLOAD_FOLDER set to: {app.config['UPLOAD_FOLDER']}")
app.logger.info(f"UPLOAD_FOLDER exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")

# Also log the absolute path for debugging
app.config['UPLOAD_FOLDER'] = os.path.abspath(app.config['UPLOAD_FOLDER'])
app.logger.info(f"UPLOAD_FOLDER (absolute): {app.config['UPLOAD_FOLDER']}")

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16 GB limit

# In-memory storage for progress updates
progress_updates = {}
progress_lock = Lock()

# Add SSE endpoint for progress updates
@app.route('/progress/<task_id>')
def progress_stream(task_id):
    def generate():
        while True:
            with progress_lock:
                progress = progress_updates.get(task_id, {})
            
            if 'completed' in progress and progress['completed']:
                yield f"data: {json.dumps(progress)}\n\n"
                break
            
            if progress:
                yield f"data: {json.dumps(progress)}\n\n"
            time.sleep(0.5)
    
    return Response(generate(), content_type='text/event-stream')

@app.route('/download/<task_id>')
def download_file(task_id):
    with progress_lock:
        if task_id not in progress_updates:
            return f"Task not found: {task_id}", 404
        
        progress = progress_updates[task_id]
        app.logger.info(f"Download requested for task {task_id}")
        app.logger.info(f"Current progress: {progress}")
        
        if progress.get('progress') != 100 or progress.get('status') != 'completed':
            return f"File not ready for download (status: {progress.get('status')}, progress: {progress.get('progress')}%)", 400
        
        output_path = progress['output_path']
        output_filename = progress['output_filename']
    
    app.logger.info(f"Checking if file exists: {output_path}")
    app.logger.info(f"File exists: {os.path.exists(output_path)}")
    
    if os.path.exists(output_path):
        app.logger.info(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        # Check if directory exists
        directory = os.path.dirname(output_path)
        app.logger.info(f"Directory exists: {os.path.exists(directory)}")
        if os.path.exists(directory):
            app.logger.info(f"Directory contents: {os.listdir(directory)}")
    
    if not os.path.exists(output_path):
        return f"File not found at {output_path}", 404
    
    @after_this_request
    def cleanup(response):
        try:
            # Clean up the session directory after download
            session_path = os.path.dirname(output_path)
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
                
            # Remove from progress updates
            with progress_lock:
                if task_id in progress_updates:
                    del progress_updates[task_id]
        except Exception as e:
            app.logger.error(f"Error cleaning up after download: {e}")
        return response
    
    return send_file(output_path, as_attachment=True, download_name=output_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    if request.method == 'POST':
        # Create unique identifiers
        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        mzml_folder = os.path.join(session_path, 'mzml')
        os.makedirs(mzml_folder, exist_ok=True)

        # --- File and Parameter Handling ---
        metadata_file = request.files.get('metadata')
        mzml_files = request.files.getlist('mzml_files')
        
        rt_precision = float(request.form.get('rt_precision', 1))
        mz_precision = float(request.form.get('mz_precision', 0.001))
        sample_id_col = request.form.get('sample_id_col', 'Sample Name')

        if not metadata_file or not mzml_files:
            return "Missing metadata file or mzML files", 400

        metadata_path = os.path.join(session_path, secure_filename(metadata_file.filename))
        metadata_file.save(metadata_path)

        for f in mzml_files:
            f.save(os.path.join(mzml_folder, secure_filename(f.filename)))

        # --- Processing ---
        output_filename = f"{session_id}_output.h5"
        output_path = os.path.join(session_path, output_filename)

        # Create progress callback
        def progress_callback(progress):
            with progress_lock:
                progress_updates[task_id] = {
                    **progress,
                    'session_id': session_id,
                    'output_filename': output_filename,
                    'output_path': output_path
                }

        # Store initial progress
        with progress_lock:
            progress_updates[task_id] = {
                'step': 'initializing',
                'status': 'in_progress',
                'message': 'Initializing batch processing',
                'progress': 0,
                'session_id': session_id,
                'output_filename': output_filename,
                'output_path': output_path
            }

        # Start processing in a separate thread
        import threading
        def process_task():
            try:
                save_dataset_as_sparse_h5(
                    folder=mzml_folder,
                    save_path=output_path,
                    rt_precision=rt_precision,
                    mz_precision=mz_precision,
                    metadata_csv_path=metadata_path,
                    sample_id_col=sample_id_col,
                    progress_callback=progress_callback
                )
            except Exception as e:
                with progress_lock:
                    progress_updates[task_id] = {
                        **progress_updates.get(task_id, {}),
                        'step': 'error',
                        'status': 'error',
                        'message': f'Processing error: {str(e)}',
                        'progress': -1,
                        'error': str(e)
                    }

        threading.Thread(target=process_task).start()

        # Return task ID for progress tracking
        return json.dumps({'task_id': task_id, 'mode': 'batch'})

@app.route('/process_single', methods=['POST'])
def process_single_file():
    if request.method == 'POST':
        # Create unique identifiers
        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_path, exist_ok=True)

        # --- File and Parameter Handling ---
        mzml_file = request.files.get('mzml_file')
        rt_precision = float(request.form.get('rt_precision', 1))
        mz_precision = float(request.form.get('mz_precision', 0.001))

        if not mzml_file:
            return "Missing mzML file", 400

        mzml_filename = secure_filename(mzml_file.filename)
        mzml_path = os.path.join(session_path, mzml_filename)
        mzml_file.save(mzml_path)

        # --- Processing ---
        # Use os.path.splitext for more robust extension handling
        base_name, ext = os.path.splitext(mzml_filename)
        app.logger.info(f"Input filename: {mzml_filename}")
        app.logger.info(f"Extension: {ext}")
        app.logger.info(f"Base name: {base_name}")
        
        output_filename = f"{base_name}_output.h5"
        output_path = os.path.join(session_path, output_filename)
        app.logger.info(f"Processing file: {mzml_path}")
        app.logger.info(f"Output will be saved to: {output_path}")
        app.logger.info(f"Output filename: {output_filename}")

        # Create progress callback
        def progress_callback(progress):
            with progress_lock:
                progress_updates[task_id] = {
                    **progress,
                    'session_id': session_id,
                    'output_filename': output_filename,
                    'output_path': output_path
                }

        # Store initial progress
        with progress_lock:
            progress_updates[task_id] = {
                'step': 'initializing',
                'status': 'in_progress',
                'message': 'Initializing single file processing',
                'progress': 0,
                'session_id': session_id,
                'output_filename': output_filename,
                'output_path': output_path
            }

        # Start processing in a separate thread
        import threading
        def process_task():
            try:
                app.logger.info(f"Starting processing thread for task {task_id}")
                app.logger.info(f"mzml_path: {mzml_path}, exists: {os.path.exists(mzml_path)}")
                app.logger.info(f"output_path: {output_path}, directory exists: {os.path.exists(session_path)}")
                
                # Debug paths before processing
                app.logger.info(f"DEBUG - mzml_path: {mzml_path}")
                app.logger.info(f"DEBUG - output_path: {output_path}")
                app.logger.info(f"DEBUG - app.config['UPLOAD_FOLDER']: {app.config['UPLOAD_FOLDER']}")
                app.logger.info(f"DEBUG - os.getcwd(): {os.getcwd()}")
                app.logger.info(f"DEBUG - __file__: {__file__}")
                app.logger.info(f"DEBUG - os.path.dirname(__file__): {os.path.dirname(__file__)}")
                
                save_single_mzml_as_sparse_h5(
                    mzml_file_path=mzml_path,
                    save_path=output_path,
                    rt_precision=rt_precision,
                    mz_precision=mz_precision,
                    progress_callback=progress_callback
                )
                
                # Verify file was created
                if os.path.exists(output_path):
                    app.logger.info(f"HDF5 file created successfully at {output_path}")
                    app.logger.info(f"File size: {os.path.getsize(output_path)} bytes")
                else:
                    app.logger.error(f"ERROR: HDF5 file was NOT created at {output_path}")
                    with progress_lock:
                        progress_updates[task_id] = {
                            **progress_updates.get(task_id, {}),
                            'step': 'error',
                            'status': 'error',
                            'message': 'Processing completed but HDF5 file was not created',
                            'progress': -1,
                            'error': f'File not found at {output_path}'
                        }
            except Exception as e:
                app.logger.error(f"ERROR in processing thread: {str(e)}")
                import traceback
                app.logger.error(traceback.format_exc())
                
                with progress_lock:
                    progress_updates[task_id] = {
                        **progress_updates.get(task_id, {}),
                        'step': 'error',
                        'status': 'error',
                        'message': f'Processing error: {str(e)}',
                        'progress': -1,
                        'error': str(e)
                    }

        threading.Thread(target=process_task).start()

        # Return task ID for progress tracking
        return json.dumps({'task_id': task_id, 'mode': 'single'})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5002)
