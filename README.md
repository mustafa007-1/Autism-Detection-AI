# Autism-Detection
## 📌 Project Overview
This project is a robust machine learning pipeline designed to classify and detect patterns associated with the Autism Spectrum. Built with Python and Scikit-learn, the system processes raw autism-related datasets, trains classification models, and provides visual analytics to evaluate model performance. The final model is serialized for seamless deployment into production environments.

## 🚀 Key Features
* **Automated Preprocessing:** Handles raw CSV data utilizing Pandas, applying transformations such as one-hot encoding for categorical variables.
* **Model Training & Classification:** Leverages Scikit-learn to train high-accuracy predictive models.
* **Visual Analytics:** Includes custom scripts to generate performance metrics and visual evaluations (e.g., confusion matrices, ROC curves).
* **Deployment-Ready:** Serializes trained model weights using Pickle, allowing the model to be easily imported and served in web applications or APIs.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model Serialization:** Pickle
* **Data Visualization:** Matplotlib / Seaborn (via custom analytics scripts)

## 📁 Project Structure
├── data/
│   └── processed_autism_data.csv   # Cleaned and preprocessed dataset
├── models/
│   └── autism_model.pkl            # Serialized model weights saved via Pickle
├── src/
│   ├── preprocess.py               # Scripts for one-hot encoding and data cleaning
│   ├── train.py                    # Model training and hyperparameter tuning
│   └── evaluate.py                 # Visual analytics and performance metrics
├── requirements.txt                # Python dependencies
└── README.md

##⚙️ Installation & Setup

pip install -r requirements.txt
Run the preprocessing pipeline:

python src/preprocess.py
Train the model:

python src/train.py
Generate Analytics & Evaluate:

python src/evaluate.py


##👨‍💻 Author : **Mustafa Waheed**

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
