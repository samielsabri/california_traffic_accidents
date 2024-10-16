import numpy as np
import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Step 3: Feature Selection

# Step 3.1: Create Outcome Variable
def fetch_tables_and_columns():
    tables_info = {}
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Fetch columns for each table
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        tables_info[table_name] = [{'id': col[0], 'column_name': col[1], 'data_type': col[2]} for col in columns]
    
    return tables_info

conn = sqlite3.connect('switrs_db_preprocessed.sqlite')  
cursor = conn.cursor()

tables_and_columns = fetch_tables_and_columns()
for table, columns in tables_and_columns.items():
    print(f"Table: {table}")
    for column in columns:
        print(f"  - Column ID: {column['id']}, Name: {column['column_name']}, Type: {column['data_type']}")
    print() 

query = """
SELECT *
FROM collisions_preprocessed
"""
data = pd.read_sql_query(query, conn)
print("data read successfully")



# Step 3.2: Additional Data Preparation

# Convert NA in certain columns to 0
data['num_victims'] = data['num_victims'].fillna(0)


# Manual selection of columns excluding those are just proxies for other variables (e.g. jurisdiction is a proxy for road_quality, collision_time is a proxy for lighting) or irrelevant (e.g. officer_id, case_id, etc.)

selected_columns = ['case_id', 'collision_time', 'collision_severity', 'collision_date', 'process_date', 'jurisdiction', 'officer_id', 'reporting_district', 'pcf_violation', 'chp_shift', 'population', 'county_city_location',\
                     'county_location', 'special_condition', 'beat_type', 'chp_beat_type', 'city_division_lapd', 'chp_beat_class', \
                        'beat_number', 'primary_road', 'secondary_road', 'distance', 'direction', 'state_highway_indicator', \
                            'caltrans_county', 'caltrans_district', 'state_route', 'route_suffix', 'postmile_prefix', 'postmile', \
                                'location_type', 'chp_road_type', 'pcf_violation_code', 'pcf_violation_subsection', 'ramp_intersection', 'side_of_highway', 'tow_away', 'severe_injury_count', \
                                    'other_visible_injury_count', 'complaint_of_pain_injury_count', 'pedestrian_killed_count',\
                                          'pedestrian_injured_count', 'bicyclist_killed_count', 'bicyclist_injured_count', \
                                            'motorcyclist_killed_count', 'motorcyclist_injured_count', 'longitude', 'latitude']

data = data.drop(columns=selected_columns)

# Step 3.2.1: Replace NA values with 0 in specified columns and use imputation by propagation

columns_to_replace_na = [
    'contains_passenger', 'contains_rear_occupant', 'victim_airbag_not_deployed',
    'victim_lap_belt_not_used', 'victim_shoulder_harness_not_used', 
    'victim_passive_restraint_not_used', 'victim_lap_shoulder_harness_not_used',
    'victim_passenger_helmet_not_used', 'victim_driver_helmet_not_used',
    'victim_child_safety_not_properly_used', 'any_ejected'
]


data[columns_to_replace_na] = data[columns_to_replace_na].fillna(0)

data = data.ffill().bfill()


# Step 3.2.2.: Separate feature types
categorical_cols = data.select_dtypes(include=['object', 'string']).columns
print("Categorical Columns:", categorical_cols)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
binary_cols = [col for col in numeric_cols if data[col].dropna().unique().size <= 2 and 
               set(data[col].dropna().unique()).issubset({0, 1})]
print("Binary Columns:", binary_cols)
true_numeric_cols = list(set(numeric_cols) - set(binary_cols))
print("True Numeric Columns:", true_numeric_cols)
binary_cols.remove('severe_or_fatal')

for col in binary_cols:
    print(f"{col} unique values: {data[col].unique()}")



# Step 3.2.3: Transform Data

# Create transformers

# Numeric Transformer
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler(feature_range=(0,1)))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, true_numeric_cols),
        ('bin', 'passthrough', binary_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply transformations
y = data['severe_or_fatal']

print("Starting preprocessing...")
X_processed = preprocessor.fit_transform(data)
print("Preprocessing complete. Proceeding with feature selection...")



# Step 3.3: Feature Selection

## Chi-Square test

chi_selector = SelectKBest(chi2, k=20)
X_kbest_chi = chi_selector.fit_transform(X_processed, y)

feature_scores = chi_selector.scores_
features = preprocessor.get_feature_names_out()
mask = chi_selector.get_support() 
selected_features = features[mask]
selected_scores = feature_scores[mask]

scored_chi_features = dict(zip(selected_features, selected_scores))
sorted_chi_features = {k: v for k, v in sorted(scored_chi_features.items(), key=lambda item: item[1], reverse=True)}

chi_features = chi_selector.get_support(indices=True)

print(f"Sorted Chi-Features:{sorted_chi_features}")

## ANOVA F-test

anova_selector = SelectKBest(f_classif, k=20)  
X_kbest_f = anova_selector.fit_transform(X_processed, y)

feature_scores = anova_selector.scores_
features = preprocessor.get_feature_names_out()

mask = anova_selector.get_support() 
selected_features = features[mask]
selected_scores = feature_scores[mask]

scored_features = dict(zip(selected_features, selected_scores))
sorted_features_anova = {k: v for k, v in sorted(scored_features.items(), key=lambda item: item[1], reverse=True)}

anova_features = anova_selector.get_support(indices=True)

print(f"Top features selected by ANOVA F-test:{sorted_features_anova}")

## Logistic Regression

lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lr.fit(X_processed, y)

coefs = lr.coef_[0]

sorted_coefs = np.sort(np.abs(coefs))[::-1]
threshold = sorted_coefs[19] 

selected_mask = np.abs(coefs) >= threshold
selected_features = preprocessor.get_feature_names_out()[selected_mask]

lr_selector = SelectFromModel(lr, max_features=20)
X_kbest_lr = lr_selector.fit_transform(X_processed, y)
lr_features = lr_selector.get_support(indices=True)

# Based on the literature (Masoumi et al., 2016; Wang et al., 2020; Hyodo et al., 2021), we know that the following features show statistically significant correlations
# with the severity of a collision:

# - Age of the vehicle driver 
# - Alcohol Consumption
# - Road Condition
# - Weather Condition
# - Hazardous Materials
# - Hit-and-Run
# - Seatbelt Usage
# - Collision Speed

# If these features are not selected by the feature selection methods, we will manually add them to the selected features list.


# Step 3.4: Feature Selection Results

all_selected_features = set(chi_features).union(set(anova_features)).union(set(lr_features))

literature_review_features = ['num__avg_victim_age', 'num__avg_party_age', 'bin__alcohol_involved', 'bin__any_hazardous_materials',\
                               'bin__party_lap_shoulder_harness_not_used', 'bin__victim_lap_shoulder_harness_not_used']

final_selected_features = all_selected_features.union(set([feature for feature in range(len(features)) if features[feature] in literature_review_features]))

feature_names = np.array(preprocessor.get_feature_names_out())
selected_feature_names = feature_names[list(final_selected_features)]

print("Features selected based on multiple criteria:")
print(final_selected_features)


# Step 4: Training the Data

X_filtered = X_processed[:, list(final_selected_features)]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Validation

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Step 5.1: Interpret Model

X_train_df = pd.DataFrame(X_train.toarray(), columns=selected_feature_names)
feature_importance = pd.DataFrame({
    'feature': selected_feature_names,  # This replaces X_train.columns
    'importance': model.coef_[0]
})

print(feature_importance.sort_values(by='importance', ascending=False))


# Step 5.2: Refine Model

param_grid = {'C': [0.1, 1, 10], 
              'solver': ['liblinear', 'lbfgs']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)