import os
import uuid
import shutil
from flask import Flask, request, render_template, send_from_directory, after_this_request
from werkzeug.utils import secure_filename
from mzrt2h5 import save_dataset_as_sparse_h5

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16 GB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    if request.method == 'POST':
        # Create a unique temporary directory for this request
        session_id = str(uuid.uuid4())
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        mzml_folder = os.path.join(session_path, 'mzml')
        os.makedirs(mzml_folder, exist_ok=True)

        # --- File and Parameter Handling ---
        metadata_file = request.files.get('metadata')
        mzml_files = request.files.getlist('mzml_files')
        
        rt_precision = float(request.form.get('rt_precision', 0.1))
        mz_precision = float(request.form.get('mz_precision', 0.01))
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

        try:
            save_dataset_as_sparse_h5(
                folder=mzml_folder,
                save_path=output_path,
                rt_precision=rt_precision,
                mz_precision=mz_precision,
                metadata_csv_path=metadata_path,
                sample_id_col=sample_id_col
            )
        except Exception as e:
            # Clean up the directory before returning the error
            shutil.rmtree(session_path)
            return f"An error occurred during processing: {e}", 500

        # --- Cleanup and File Sending ---
        @after_this_request
        def cleanup(response):
            try:
                shutil.rmtree(session_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up directory {session_path}: {e}")
            return response

        return send_from_directory(session_path, output_filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
