# 🍽️ Smart Premium – ML-Based Restaurant Recommendation System  

Smart Premium is a machine learning–powered restaurant recommendation system built using Python and Streamlit.  

The project covers the complete ML lifecycle — from EDA and preprocessing to model training, evaluation, and deployment via an interactive web app.

---

## 🚀 Project Highlights

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing & Feature Engineering
- 🤖 Machine Learning Pipeline
- 📈 Model Evaluation & Metrics Tracking
- 💾 Model Serialization using Joblib
- 🌐 Streamlit Web Application
- 📂 Structured, production-style project architecture

---

## 🛠️ Tech Stack

- Python 3.12  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  
- MLflow (experiment tracking)

---

## 📂 Project Structure

SmartPremium/

├── mlruns/0/  
│   └── meta.yaml  

├── outputs/  
│   ├── eda/  
│   ├── Overall_Project_Outputs_with_...  
│   ├── Streamlit_Output_Prediction.png  
│   ├── metrics.json  
│   ├── smartpremium_pipeline.joblib  
│   └── submission.csv  

├── src/  
│   ├── SmartPremium_EDA.py  
│   ├── preprocess.py  
│   ├── train_and_evaluate.py  
│   ├── streamlit_app.py  
│   └── __init__.py  

├── run.py  
├── requirements.txt  
├── README.md  
└── .gitattributes  

---

## ⚙️ How the Project Works

### 1️⃣ Exploratory Data Analysis (EDA)
- Data inspection
- Missing value analysis
- Feature distribution visualization
- Correlation analysis

Script:
src/SmartPremium_EDA.py

---

### 2️⃣ Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Preparing data for ML pipeline

Script:
src/preprocess.py

---

### 3️⃣ Model Training & Evaluation
- Training ML model
- Cross-validation
- Evaluation metrics calculation
- Saving trained pipeline

Script:
src/train_and_evaluate.py

Saved Model:
outputs/smartpremium_pipeline.joblib

Metrics:
outputs/metrics.json

---

### 4️⃣ Streamlit Web App
- Interactive UI
- User input for predictions
- Real-time model inference

Script:
src/streamlit_app.py

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/yourusername/smart-premium.git  
cd smart-premium  

---

### 2️⃣ Install Dependencies

pip install -r requirements.txt  

---

### 3️⃣ Run Training Pipeline

python run.py  

This will:
- Preprocess data
- Train model
- Save pipeline
- Generate evaluation metrics

---

### 4️⃣ Launch Streamlit App

streamlit run src/streamlit_app.py  

---

## 📊 Outputs

Inside the outputs/ folder:

- smartpremium_pipeline.joblib → Trained ML model
- metrics.json → Model evaluation metrics
- submission.csv → Final prediction output
- Streamlit_Output_Prediction.png → App prediction screenshot
- eda/ → Visual analysis files

---

## 📈 Key Concepts Demonstrated

- Modular ML project structure  
- End-to-end ML workflow  
- Model persistence with Joblib  
- Experiment tracking (MLflow)  
- Deployment-ready architecture  
- Reproducible training pipeline  

---

## 🔮 Future Improvements

- Add hyperparameter tuning
- Deploy to cloud (Streamlit Cloud / Render)
- Add Docker containerization
- Implement advanced recommendation logic
- Add model monitoring

---

## 👨‍💻 Author

Sabarish Balakrishnan  
Data Analyst | Machine Learning Enthusiast  
