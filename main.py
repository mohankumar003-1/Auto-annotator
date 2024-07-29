from flask import Flask, request, send_from_directory, render_template_string, redirect, url_for
import cv2
import os
from test import process_image

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management

# Define the folder to store uploaded images and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def predict(input_path, output_path):
    """Process the image and save annotations."""
    image = cv2.imread(input_path)
    process_image(image, input_path, output_path, 75)

# HTML template for the upload form with drag-and-drop and file input
upload_form = '''
<!doctype html>
<title>Upload Images</title>
<h1>Upload Images</h1>
<style>
    #drop-area {
        border: 2px dashed #cccccc;
        border-radius: 5px;
        width: 400px;
        height: 200px;
        line-height: 200px;
        text-align: center;
        margin: 20px auto;
        color: #cccccc;
    }
    #drop-area.hover {
        border-color: #333333;
        color: #333333;
    }
    #fileElem {
        display: none;
    }
    label {
        cursor: pointer;
    }
</style>
<div id="drop-area">
    <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" id="fileElem" name="file" multiple />
        <label for="fileElem">Drag and drop images here or click to select them</label>
        <input type="submit" value="Upload" />
    </form>
</div>
<script>
    let dropArea = document.getElementById('drop-area');
    let fileInput = document.getElementById('fileElem');
    let uploadForm = document.getElementById('uploadForm');

    dropArea.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dropArea.classList.add('hover');
    });

    dropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropArea.classList.remove('hover');
    });

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('hover');
        let files = e.dataTransfer.files;
        fileInput.files = files;
        uploadForm.submit();
    });

    fileInput.addEventListener('change', () => {
        let fileNames = Array.from(fileInput.files).map(file => file.name).join(', ');
        dropArea.querySelector('label').textContent = fileNames || 'No files selected';
    });
</script>
'''

# HTML template for viewing the images
view_images = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>View Images</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        #gallery {
            width: 30%;
            padding-right: 20px;
            overflow-y: auto;
        }
        #gallery img {
            width: 100%;
            height: auto;
            cursor: pointer;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            transition: transform 0.2s;
        }
        #gallery img:hover {
            transform: scale(1.05);
        }
        #large-view {
            width: 70%;
        }
        #large-image {
            width: 100%;
            height: auto;
            max-height: 600px;
            border: 1px solid #ddd;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <h1>Uploaded Images with Annotations</h1>
    <div style="display: flex;">
        <!-- Gallery section -->
        <div id="gallery">
            {% for filename in filenames %}
                <div>
                    <img src="{{ url_for('processed_file', filename=filename) }}" 
                         alt="{{ filename }}" 
                         onclick="showImage('{{ url_for('processed_file', filename=filename) }}')"/>
                    <p>{{ filename }}</p>
                </div>
            {% endfor %}
        </div>
        
        <!-- Larger view section -->
        <div id="large-view">
            <img id="large-image" src="" alt="Click on an image to view it here"/>
        </div>
    </div>

    <script>
    function showImage(src) {
        const largeImage = document.getElementById('large-image');
        largeImage.src = src;
    }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(upload_form)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return "No files part", 400

    files = request.files.getlist('file')
    if not files:
        return "No selected files", 400

    filenames = []
    for file in files:
        if file.filename == '':
            continue
        if file:
            filename = file.filename
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)
            
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'])
            predict(original_path, processed_path)
            filenames.append(filename)

    if not filenames:
        return "No valid files uploaded", 400

    return render_template_string(view_images, filenames=filenames)

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve the processed image from the 'processed' directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
