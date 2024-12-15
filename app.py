from flask import Flask, render_template, request
import os
import subprocess

# Initialize Flask app
app = Flask(__name__)

# Set the upload folder for images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected for upload", 400

        # Save the uploaded file to the server
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Run predict.py using subprocess and pass the image path
            result = subprocess.run(
                ['python', 'classification.py', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                # If predict.py returns an error, display it
                return f"Error: {result.stderr}", 500
            
            # Get the prediction output from predict.py
            prediction = result.stdout.strip()

            # Display the result on the web
            return render_template('result.html', prediction=prediction, image_path=file_path)
        
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    # Render the upload form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
