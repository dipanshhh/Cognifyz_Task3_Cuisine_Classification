COGNIFYZ TECHNOLOGIES
MACHINE LEARNING INTERNSHIP

TASK 3: CUISINE CLASSIFICATION

1. OBJECTIVE

The objective of this task is to develop a machine learning model
that classifies restaurants based on their cuisines.

2. DATASET

The restaurant dataset was used for building the classification model.

The target variable is:

Main Cuisine

The first cuisine listed for each restaurant was used as the main
cuisine for classification.

3. DATA PREPROCESSING

The dataset was preprocessed before training the model.

The following steps were performed:

- Missing cuisine values were removed.
- The main cuisine was extracted from the cuisine information.
- Cuisine classes with fewer than 20 examples were removed.
- Numerical missing values were handled using median imputation.
- Categorical missing values were handled using the most frequent value.
- Categorical variables were converted into numerical form using
  One-Hot Encoding.

4. FEATURES USED

The following restaurant features were used:

- City
- Locality
- Average Cost for two
- Price range
- Aggregate rating
- Votes
- Has Table booking
- Has Online delivery

5. TRAINING AND TESTING

The dataset was divided into training and testing data.

80% of the data was used for training and 20% was used for testing.

Stratified splitting was used to maintain the distribution of
different cuisine classes.

6. MACHINE LEARNING MODEL

Random Forest Classifier was selected as the classification algorithm.

The model was trained using the training dataset and then used to
predict the cuisine of restaurants in the testing dataset.

7. MODEL PERFORMANCE

Accuracy: 0.3374

Precision: 0.1959

Recall: 0.3374

The model achieved an accuracy of approximately 33.74% on the
testing data.

8. CUISINE-WISE ANALYSIS

The model performance was analyzed across different cuisine classes
using precision, recall and F1-score.

The cuisine-wise performance results were saved in:

cuisine_classification_results.csv

9. CHALLENGES AND BIASES

The model performance varies across different cuisine classes.

Some cuisines have fewer examples in the dataset, which can make
them harder to classify accurately.

Different cuisines can also have similar restaurant characteristics,
which may cause confusion during classification.

Therefore, the model may perform better for cuisines with more
representative training data than for less frequent cuisines.

10. CONCLUSION

A machine learning model for cuisine classification was successfully
developed using Python, Pandas and Scikit-learn.

The Random Forest Classifier was trained after preprocessing the
restaurant dataset and encoding categorical variables.

The model was evaluated using accuracy, precision and recall, and
cuisine-wise performance was also analyzed.

The task was successfully completed as part of the Cognifyz
Technologies Machine Learning Internship.