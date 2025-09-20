# Steam Data Analysis & Recommendation System

## Project Overview

This project is a comprehensive big data analysis of Steam gaming platform data, focusing on data collection, cleaning, exploratory data analysis, natural language processing, and building a collaborative filtering recommendation system. The project was developed as part of a Big Data and Cloud Computing course at the University of Chicago.

## Project Structure

The project is organized into several Jupyter notebooks, each focusing on different aspects of the analysis:

### 📊 Data Collection & Processing
- **`[Data Constructing]DataCollection.ipynb`** - Steam API data collection from Steam and SteamSpy APIs
- **`[Data Constructing]DataCleaning_Json_Steamdata.ipynb`** - Data cleaning and preprocessing of collected JSON data

### 🔍 Exploratory Data Analysis & Classification
- **`[EDA+Classification & Prediction]Steam_Data_Project_cleaning.ipynb`** - Comprehensive EDA, data cleaning, and classification analysis using PySpark
- **`[Classification & Prediction]SteamData_Model.ipynb`** - Advanced machine learning models including Random Forest classification and Neural Network regression for CCU prediction

### 🧠 Natural Language Processing
- **`[Named Entity Recognition & Word Cloud]Steam_Reviews_NER_WordCloud.ipynb`** - NER analysis and word cloud generation for Steam reviews
- **`[Named Entity Recognition & Word Cloud]SteamData_withCCUcategory.ipynb`** - Categorizing games by concurrent user levels

### 🎯 Recommendation System
- **`[Recommendation System]Label.ipynb`** - Game labeling based on review sentiment
- **`[Recommendation System]Recommendation System.ipynb`** - Collaborative filtering recommendation system implementation

## Key Features

### 1. Data Collection
- **Steam API Integration**: Collected game metadata from Steam's official API
- **SteamSpy API**: Gathered additional game statistics and user data
- **Rate Limiting**: Implemented robust rate limiting to handle API constraints
- **Data Volume**: Processed over 25,000 games and 50+ million reviews

### 2. Data Processing & Cleaning
- **PySpark Implementation**: Leveraged Apache Spark for big data processing
- **Data Quality**: Handled missing values, duplicates, and data type conversions
- **Feature Engineering**: Created derived features like positive ratios and review categories
- **Outlier Detection**: Identified and managed outliers using statistical methods

### 3. Exploratory Data Analysis & Machine Learning
- **Game Classification**: Categorized games by concurrent user levels (low, medium-low, medium-high, high, very high)
- **Sentiment Analysis**: Analyzed review sentiment patterns across different game categories
- **Advanced ML Models**: Implemented Random Forest classification and Neural Network regression
- **CCU Prediction**: Built models to predict concurrent user counts with 78.8% R² score
- **Cross-Validation**: Performed 10-fold cross-validation to ensure model robustness
- **Hyperparameter Tuning**: Used Keras Tuner for optimal neural network architecture
- **Visualization**: Created comprehensive visualizations for data distribution and patterns
- **Statistical Analysis**: Performed detailed statistical analysis on game metrics

### 4. Natural Language Processing
- **Named Entity Recognition**: Used Spark NLP for entity extraction from game reviews
- **Word Cloud Generation**: Created visual representations of frequently mentioned terms
- **Sentiment Classification**: Implemented logistic regression for sentiment scoring
- **Text Preprocessing**: Applied advanced text cleaning and preprocessing techniques

### 5. Recommendation System
- **Collaborative Filtering**: Implemented ALS (Alternating Least Squares) algorithm
- **Sentiment-Based Scoring**: Used sentiment analysis scores as rating inputs
- **User Recommendations**: Generated personalized game recommendations for users
- **Item Recommendations**: Identified similar games based on user preferences
- **Model Evaluation**: Achieved RMSE of 0.13 for recommendation accuracy

## Technical Stack

- **Python 3.9.12**
- **Apache Spark (PySpark)** - Big data processing
- **Google Cloud Platform (GCP)** - Cloud storage and computing
- **Spark NLP** - Natural language processing
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning and neural networks
- **XGBoost** - Gradient boosting framework
- **Keras Tuner** - Hyperparameter optimization
- **Imbalanced-learn** - Handling class imbalance
- **Matplotlib/Seaborn** - Data visualization
- **Pandas** - Data manipulation
- **NLTK** - Natural language toolkit
- **WordCloud** - Text visualization

## Data Sources

1. **Steam API** - Game metadata and details
2. **SteamSpy API** - Additional game statistics and user metrics
3. **Steam Reviews** - User-generated review data from Kaggle
4. **Google Cloud Storage** - Centralized data storage

## Key Findings

### Game Classification & Prediction
- Successfully categorized games into 5 concurrent user level categories
- **Random Forest Classification**: Achieved 81.4% accuracy for paid games and 76.7% for free games
- **Neural Network Regression**: Achieved 78.8% R² score for CCU prediction
- **Cross-Validation**: Validated model robustness with 10-fold cross-validation
- Identified patterns in game popularity based on various features
- Analyzed differences between free and paid games

### Sentiment Analysis
- Achieved 86% accuracy in sentiment classification
- Identified key factors that influence positive vs negative reviews
- Generated sentiment scores for over 39 million reviews

### Recommendation System
- Built an effective collaborative filtering model
- Generated personalized recommendations for users
- Identified similar games based on user behavior patterns

## Usage Instructions

### Prerequisites
- Python 3.9+
- Apache Spark
- Google Cloud Platform account
- Required Python packages (see individual notebooks for specific requirements)

### Running the Project
1. Clone the repository
2. Set up Google Cloud Platform credentials
3. Install required dependencies
4. Run notebooks in sequence:
   - Start with data collection notebooks
   - Proceed to data cleaning and EDA
   - Run NLP analysis
   - Execute recommendation system

### Data Access
- Data is stored in Google Cloud Storage
- Ensure proper GCP authentication
- Update bucket paths in notebooks as needed

## Model Performance

### Classification Models
- **Random Forest (Paid Games)**: 81.4% accuracy, 88% F1-score
- **Random Forest (Free Games)**: 76.7% accuracy, 83% F1-score
- **Cross-Validation**: Consistent performance across 10 folds
- **Class Imbalance Handling**: Implemented weighted sampling

### Regression Models
- **Neural Network (CCU Prediction)**: 78.8% R² score
- **Hyperparameter Tuning**: Optimized using Keras Tuner
- **Architecture**: 3-layer neural network with dropout and batch normalization
- **Cross-Validation**: 10-fold validation with best model selection

### Sentiment Analysis Model
- **Accuracy**: 86.09%
- **F1-Score**: 80.04%
- **Algorithm**: Logistic Regression with TF-IDF features

### Recommendation System
- **RMSE**: 0.130
- **Algorithm**: Alternating Least Squares (ALS)
- **Features**: User-item interactions with sentiment scores

## Future Enhancements

1. **Content-Based Filtering**: Implement content-based recommendations using game features
2. **Hybrid Approach**: Combine collaborative and content-based filtering
3. **Real-time Recommendations**: Develop real-time recommendation API
4. **Advanced NLP**: Implement more sophisticated NLP techniques
5. **Model Optimization**: Fine-tune hyperparameters for better performance
