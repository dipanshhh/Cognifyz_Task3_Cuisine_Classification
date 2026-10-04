import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# =========================================
# STEP 1: LOAD DATASET
# =========================================

data = pd.read_csv("Dataset .csv")

print("Dataset loaded successfully!")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])


# =========================================
# STEP 2: PREPARE CUISINE TARGET
# =========================================

# Remove rows where cuisine information is missing
data = data.dropna(subset=["Cuisines"]).copy()

# Use the first cuisine listed as the main cuisine
data["Main Cuisine"] = (
    data["Cuisines"]
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)

# Keep only cuisines having enough examples
cuisine_counts = data["Main Cuisine"].value_counts()

valid_cuisines = cuisine_counts[cuisine_counts >= 20].index

data = data[data["Main Cuisine"].isin(valid_cuisines)].copy()

print("\nNumber of cuisine classes:", data["Main Cuisine"].nunique())


# =========================================
# STEP 3: SELECT FEATURES
# =========================================

features = [
    "City",
    "Locality",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Has Table booking",
    "Has Online delivery"
]

X = data[features]
y = data["Main Cuisine"]


# =========================================
# STEP 4: IDENTIFY COLUMN TYPES
# =========================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# =========================================
# STEP 5: PREPROCESS DATA
# =========================================

numerical_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# =========================================
# STEP 6: SPLIT DATA
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================
# STEP 7: RANDOM FOREST CLASSIFIER
# =========================================

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])


# =========================================
# STEP 8: TRAIN MODEL
# =========================================

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# =========================================
# STEP 9: MAKE PREDICTIONS
# =========================================

predictions = model.predict(X_test)


# =========================================
# STEP 10: EVALUATE MODEL
# =========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


# =========================================
# STEP 11: CUISINE-WISE ANALYSIS
# =========================================

print("\n========== CUISINE-WISE PERFORMANCE ==========")

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

cuisine_results = []

for cuisine in y_test.unique():

    if cuisine in report:

        cuisine_results.append({
            "Cuisine": cuisine,
            "Precision": report[cuisine]["precision"],
            "Recall": report[cuisine]["recall"],
            "F1 Score": report[cuisine]["f1-score"],
            "Support": report[cuisine]["support"]
        })

cuisine_results = pd.DataFrame(cuisine_results)

cuisine_results = cuisine_results.sort_values(
    by="Recall",
    ascending=False
)

print(cuisine_results.head(15).to_string(index=False))


# =========================================
# STEP 12: SAVE RESULTS
# =========================================

cuisine_results.to_csv(
    "cuisine_classification_results.csv",
    index=False
)

print("\nCuisine-wise results saved to:")
print("cuisine_classification_results.csv")

print("\n========== TASK 3 COMPLETED ==========")