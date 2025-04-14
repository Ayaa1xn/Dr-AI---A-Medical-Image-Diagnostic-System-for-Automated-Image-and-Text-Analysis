# Medical-OCR
The Dr AI OCR project aims to revolutionize the healthcare landscape by combining cutting-
edge technologies in Optical Character Recognition (OCR), Natural Language Processing (NLP),
and Artificial Intelligence (AI) to enhance the extraction, analysis, and interpretation of medical
data. By integrating Tesseract OCR for text extraction, SciSpaCy for biomedical entity
recognition, and ChatGPT for AI-powered analysis, this solution provides a robust platform for
processing medical documents and diagnostic data efficiently and accurately. 

-----------------------------------------------------------------------------------------------------------
Files to Add
- ChatGPTApi.py: Handles API communication with ChatGPT.
- .gitignore`**: To ignore unnecessary files like model weights, venv, etc.
- requirements.txt`**: List all dependencies. 

---
Medical Diagnosis & Report Analyzer - Flask App

Overview
This Flask-based web application predicts medical conditions (such as diabetes, cancer, heart disease, liver disease, kidney disease, and more) using trained machine learning models. Additionally, it provides a report analyzer that summarizes medical reports using ChatGPT.

Features
✅ Predicts multiple medical conditions using machine learning models  
✅ Supports image-based and text-based report analysis  
✅ Uses OpenAI's ChatGPT for advanced report summarization  
✅ User-friendly web interface built with Flask  
✅ Secure file upload system  

Folder Structure

/your_project_repo
│── /models                 # Machine learning models (pkl/h5)
│── /static                 # Static assets (CSS, JS, images)
│── /templates              # HTML templates for Flask
│── app.py                  # Main Flask application
│── ChatGPTApi.py           # ChatGPT API integration
│── requirements.txt        # Python dependencies
│── README.md               # Documentation
│── .gitignore              # Files to exclude from Git
│── config.py               # Configuration settings
│── LICENSE                 # License for open-source projects
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Ayaa1xn/your-project-repo.git
   cd your-project-repo
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```sh
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Predictions
- **Diabetes Prediction** (`/diabetes`)
- **Cancer Prediction** (`/cancer`)
- **Heart Disease Prediction** (`/heart`)
- **Liver Disease Prediction** (`/liver`)
- **Kidney Disease Prediction** (`/kidney`)
- **Report Analyzer** (`/summarizer`)

## API Integration
The app integrates OpenAI's ChatGPT API for medical report summarization. To use the API, add your OpenAI API key to `ChatGPTApi.py`.

## License
This project is licensed under the MIT License.

---
🚀
