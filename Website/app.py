# from flask import Flask, render_template, request, redirect, url_for
# from deepface import DeepFace # type: ignore
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Analyze age, gender, and emotion
#             try:
#                 result = DeepFace.analyze(
#                     img_path=filepath,
#                     actions=['age', 'gender', 'emotion'],
#                     enforce_detection=False
#                 )
#                 age = result[0]['age']
#                 gender = result[0]['dominant_gender']
#                 emotion = result[0]['dominant_emotion']
#                 return render_template(
#                     'index.html',
#                     filename=filename,
#                     age=age,
#                     gender=gender,
#                     emotion=emotion
#                 )
#             except Exception as e:
#                 return render_template('index.html', error=str(e))
#     return render_template('index.html')

# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)







# from flask import Flask, render_template, request, redirect, url_for
# import os
# import cv2
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Basic face detection
#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             img = cv2.imread(filepath)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
#             if len(faces) > 0:
#                 # Simple age estimation heuristic
#                 (x,y,w,h) = faces[0]
#                 age = int(20 + (w/10))
#                 return render_template('index.html', filename=filename, age=age)
#             return render_template('index.html', filename=filename, error="No face detected")
#     return render_template('index.html')

# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['FACE_DETECTION_SCALE'] = 1.1
app.config['FACE_DETECTION_NEIGHBORS'] = 5

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """Save uploaded file with timestamp prefix"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename, filepath

def detect_faces(image_path):
    """Detect faces in an image using OpenCV"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=app.config['FACE_DETECTION_SCALE'],
            minNeighbors=app.config['FACE_DETECTION_NEIGHBORS']
        )
        
        return faces, img.shape
        
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        raise

def estimate_age(faces, image_shape):
    """Estimate age based on face position and size"""
    if len(faces) == 0:
        return None
        
    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Simple heuristic: age increases with face size relative to image
    face_size_ratio = (w * h) / (image_shape[0] * image_shape[1])
    
    # Base age + adjustment based on face size
    estimated_age = int(20 + (face_size_ratio * 50))
    
    # Cap the age estimation
    return min(max(estimated_age, 15), 80)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', 
                                error='No file part in the request')
            
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('index.html', 
                                error='No file selected for upload')
            
        # Check file type
        if not allowed_file(file.filename):
            return render_template('index.html', 
                                error='File type not allowed. Please upload a JPG or PNG image.')
        
        try:
            # Save the uploaded file
            filename, filepath = save_uploaded_file(file)
            logger.info(f"File saved to: {filepath}")
            
            # Detect faces
            faces, image_shape = detect_faces(filepath)
            
            if len(faces) == 0:
                return render_template('index.html',
                                    filename=filename,
                                    error="No faces detected in the image")
            
            # Estimate age
            age = estimate_age(faces, image_shape)
            
            # Draw rectangle around faces (for visualization)
            img = cv2.imread(filepath)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the processed image
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, img)
            
            return render_template('index.html',
                                filename=processed_filename,
                                age=age,
                                faces_count=len(faces))
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return render_template('index.html',
                                error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0')