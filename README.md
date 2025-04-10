# Age Detection System 👶👴

A deep learning-based system that predicts age from facial images with real-time web interface.

## Features ✨
- Real-time age prediction from webcam or images
- Pre-trained CNN model (85%+ accuracy)
- Flask-based web interface
- Mobile-responsive design
- REST API endpoint for integration

## Tech Stack 🛠️
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Flask
- HTML5/CSS3/JavaScript

## Installation 🚀

### Prerequisites
- Python 3.8+
- pip

### Steps

# Clone repository
git clone https://github.com/NinjaPercia0114/age-detection-system.git
cd age-detection-system

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (if not included)
wget https://example.com/age_model.h5 -O models/age_model.h5

### Usage 🖥️
Web Interface
  python app.py
Open http://localhost:5000 in your browser

### API Endpoint
import requests

response = requests.post(
    'http://localhost:5000/predict',
    files={'file': open('test.jpg', 'rb')}
)
print(response.json())  # Returns {'age': 27, 'confidence': 0.87}


## Project Structure 📁

age-detection-system/
├── app.py              # Flask application

├── models/             # Pretrained models

├── static/             # CSS/JS assets

├── templates/          # HTML templates

├── utils/              # Helper functions

├── requirements.txt    # Dependencies

└── README.md           # This file

#### Dataset 📊
Model trained on:
  UTKFace dataset (23,708 images)
  Age range: 1-116 years

#### Performance 📈
  Metric	Score
  Accuracy	85.2%
  MAE	4.3
  Inference Time	120ms

#### Contributing 🤝
Fork the project
  Create your branch (git checkout -b feature/AmazingFeature)
  Commit changes (git commit -m 'Add amazing feature')
  Push (git push origin feature/AmazingFeature)
  Open Pull Request

### License 📄
MIT License - see LICENSE file

### Contact 📧
Harsh Shah
harsh03030@gmail.com
