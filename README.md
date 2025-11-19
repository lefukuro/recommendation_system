# Recommendation System Project

## 📋 Project Overview
A machine learning system for personalized content recommendations based on user engagement data. The project demonstrates full ML pipeline from data preprocessing to model deployment.

## 🎯 Business Problem
Increase user engagement by providing relevant content recommendations using historical interaction data.

## 🛠️ Technical Stack
- **Python** (pandas, numpy, scikit-learn)
- **CatBoost** for gradient boosting
- **TF-IDF** for text processing
- **PCA** for dimensionality reduction

## 📊 Key Features
- Data preprocessing and feature engineering
- Advanced text processing with TF-IDF and PCA
- Hyperparameter tuning with RandomizedSearchCV
- Recommendation quality evaluation using DCG metric
- Feature importance analysis

## 🏗️ Project Structure
recommendation_system/
├── 📓 notebooks/
│   └── recommendation-system-catboost.ipynb          # EDA and Model training
├── 🚀 src/
│   └── model_conn.py                                 # FastAPI server
├── 📁 models/
│   └── catboost_model.cbm                            # Trained CatBoost model (model weight)
├── 📋 requirements.txt
├── 📖 README.md
└── 🔧 .gitignore

## 📈 Results
- **DCG@5 Score**: 1.1687
- **Efficiency**: 95.7% of theoretical maximum
- **Key drivers**: User engagement history, content topics, user data, text features

## 🚀 Quick Start
```bash
git clone https://github.com/lefukuro/recommendation_system.git
cd recommendation_system
pip install -r requirements.txt
```

## 📝 License
MIT License
