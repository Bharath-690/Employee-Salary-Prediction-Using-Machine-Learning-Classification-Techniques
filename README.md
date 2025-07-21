# Employee-Salary-Prediction-Using-Machine-Learning-Classification-Techniques
A machine learning project that predicts employee salaries using classification techniques such as Random Forest, Decision Tree, and Logistic Regression.

---

## 🚀 Features

- User-friendly Streamlit interface
- Predicts salary class based on user inputs
- Uses trained ML model (`best_model.pkl`)
- Preprocessing with encoders and scaler
- Visualizations using Seaborn and Matplotlib
- SMOTE for handling imbalanced datasets

---

## 🧠 Machine Learning Components

- `best_model.pkl`: Trained ML classification model
- `label_encoders.pkl`: Label encoders for categorical features
- `scaler.pkl`: Scaler object for feature normalization
- `feature_list.pkl`: List of selected features used for prediction
- `model_utils.py`: Contains ML pipeline and preprocessing logic

---

## 🗂️ Project Structure
- `employee-salary-prediction
- `app.py # Streamlit app interface
model_utils.py # ML pipeline and utility functions
salary_prediction_pickle_files.zip   # All 4 pickle files Compressed 
1.best_model.pkl # Trained ML model
2.label_encoders.pkl # Encoders for categorical columns
3.scaler.pkl # Feature scaler
4.feature_list.pkl # Features used for training
 requirements.txt # List of required Python packages
 README.md # Project documentation
.gitignore # Files/directories to ignore in Git
After_Run_Auto_ML.png
 After_Run_Auto_ML_1.png
 Auto_ML_Prediction_with_Dataset.png
 Auto_ML_Visual.png
 Auto_ML_Visual_1.png
 Employee_salary_prediction_final_output.png
 Employee_salary_prediction_final_output1.png
*.png # Result images and charts

---

## 💻 How to Run the App

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/employee-salary-prediction.git
   cd employee-salary-prediction

pip install -r requirements.txt

streamlit run app.py

http://localhost:8501

---

## 🖼️ Demo Screenshots

### Final Outputs of APP

![Employee Salary Prediction Final Output](Employee_salary_prediction_final_output.png)  
![Employee Salary Prediction Final Output 1](Employee_salary_prediction_final_output1.png)  

### Salary Predictions by Category

![Salary Prediction Less Than 50K](salary_prediction_less_than_50K.png)  
![Salary Prediction More Than 50K](salary_prediction_more_than_50K.png)  

---

### Prediction with Dataset

![Auto ML Prediction with Dataset](Auto_ML_Prediction_with_Dataset.png)  


### After Running Auto ML

![After Run Auto ML](After_Run_Auto_ML.png)  
![After Run Auto ML 1](After_Run_Auto_ML_1.png)  


### Auto ML Visualizations

![Auto ML Visual](Auto_ML_Visual.png)  
![Auto ML Visual1](Auto_ML_Visual_1.png)  


