import sqlite3
import pandas as pd
import numpy as np
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


# Step 1: Load and clean the Data
conn = sqlite3.connect('switrs_db.sqlite')  
cursor = conn.cursor()

collision_query = """
SELECT *
FROM collisions
"""

victims_query = """
SELECT *
FROM victims
"""

party_query = """
SELECT *
FROM parties
"""

collisions_data = pd.read_sql_query(collision_query, conn)
victims_data = pd.read_sql_query(victims_query, conn)
parties_data = pd.read_sql_query(party_query, conn)

# Step 1.1.: Convert NA in certain columns
collisions_data['alcohol_involved'] = collisions_data['alcohol_involved'].fillna(0)
collisions_data['not_private_property'] = collisions_data['not_private_property'].fillna(0)
victims_data['victim_safety_equipment_1'] = victims_data['victim_safety_equipment_1'].fillna('Unknown')
victims_data['victim_safety_equipment_2'] = victims_data['victim_safety_equipment_2'].fillna('Unknown')
victims_data['victim_ejected'] = victims_data['victim_ejected'].fillna('Unknown')
victims_data['victim_seating_position'] = victims_data['victim_seating_position'].fillna('Unknown')
victims_data['victim_age'] = victims_data['victim_age'].fillna(victims_data['victim_age'].mean())
parties_data['party_age'] = parties_data['party_age'].fillna(parties_data['party_age'].mean())
parties_data['party_safety_equipment_1'] = parties_data['party_safety_equipment_1'].fillna('Unknown')
parties_data['party_safety_equipment_2'] = parties_data['party_safety_equipment_2'].fillna('Unknown')


# Step 1.2: Fetch tables and columns to understand available features

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

tables_and_columns = fetch_tables_and_columns()
for table, columns in tables_and_columns.items():
    print(f"Table: {table}")
    for column in columns:
        print(f"  - Column ID: {column['id']}, Name: {column['column_name']}, Type: {column['data_type']}")
    print() 


# Step 2: Data Preprocessing

# Step 2.1: Create Binary Flags
def add_binary_flags_victim(victims_data):
    """Creates binary flags for the victim data for further processing"""
    victims_data['contains_passenger'] = victims_data['victim_seating_position'].isin([
        'passenger seat 2', 'passenger seat 3', 'passenger seat 4', 
        'passenger seat 5', 'passenger seat 6']).astype(int)

    victims_data['contains_rear_occupant'] = (victims_data['victim_seating_position'] == 'rear occupant of truck or van').astype(int)

    victims_data['victim_airbag_not_deployed'] = ((victims_data['victim_safety_equipment_1'] == 'airbag not deployed') | 
                                                (victims_data['victim_safety_equipment_2'] == 'airbag not deployed')).astype(int)

    victims_data['victim_lap_belt_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'lap belt not used') | 
                                                (victims_data['victim_safety_equipment_2'] == 'lap belt not used')).astype(int)

    victims_data['victim_shoulder_harness_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'shoulder harness not used') | 
                                                        (victims_data['victim_safety_equipment_2'] == 'shoulder harness not used')).astype(int)

    victims_data['victim_passive_restraint_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'passive restraint not used') | 
                                                        (victims_data['victim_safety_equipment_2'] == 'passive restraint not used')).astype(int)

    victims_data['victim_lap_shoulder_harness_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'lap/shoulder harness not used') | 
                                                            (victims_data['victim_safety_equipment_2'] == 'lap/shoulder harness not used')).astype(int)

    victims_data['victim_passenger_helmet_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'passenger, motorcycle helmet not used') | 
                                                        (victims_data['victim_safety_equipment_2'] == 'passenger, motorcycle helmet not used')).astype(int)

    victims_data['victim_driver_helmet_not_used'] = ((victims_data['victim_safety_equipment_1'] == 'driver, motorcycle helmet not used') | 
                                                    (victims_data['victim_safety_equipment_2'] == 'driver, motorcycle helmet not used')).astype(int)

    victims_data['victim_child_safety_not_properly_used'] = (victims_data['victim_safety_equipment_1'].isin(['child restraint in vehicle not used', 'child restraint in vehicle, improper use']) | 
                                                            victims_data['victim_safety_equipment_2'].isin(['child restraint in vehicle not used', 'child restraint in vehicle, improper use'])).astype(int)

    victims_data['any_ejected'] = victims_data['victim_ejected'].isin(['fully ejected', 'partially ejected']).astype(int)

    return victims_data

def add_binary_flags_parties(parties_data):
    """Creates binary flags for the parties data for further processing"""
    
    parties_data['all_male'] = (parties_data['party_sex'] == 'male').astype(int)
    parties_data['any_under_alcohol_influence'] = (parties_data['party_sobriety'] == 'had been drinking, under influence').astype(int)
    parties_data['any_drug_use'] = (parties_data['party_drug_physical'] == 'under drug influence').astype(int)
    parties_data['any_physical_impairment'] = (parties_data['party_drug_physical'] == 'impairment - physical').astype(int)
    parties_data['any_sleepy'] = (parties_data['party_drug_physical'] == 'sleepy/fatigued').astype(int)
    parties_data['party_airbag_not_deployed'] = ((parties_data['party_safety_equipment_1'] == 'airbag not deployed') |
                                                (parties_data['party_safety_equipment_2'] == 'airbag not deployed')).astype(int)
    parties_data['party_lap_belt_not_used'] = ((parties_data['party_safety_equipment_1'] == 'lap belt not used') |
                                            (parties_data['party_safety_equipment_2'] == 'lap belt not used')).astype(int)
    parties_data['party_shoulder_harness_not_used'] = ((parties_data['party_safety_equipment_1'] == 'shoulder harness not used') |
                                                    (parties_data['party_safety_equipment_2'] == 'shoulder harness not used')).astype(int)
    parties_data['party_passive_restraint_not_used'] = ((parties_data['party_safety_equipment_1'] == 'passive restraint not used') |
                                                        (parties_data['party_safety_equipment_2'] == 'passive restraint not used')).astype(int)
    parties_data['party_lap_shoulder_harness_not_used'] = ((parties_data['party_safety_equipment_1'] == 'lap/shoulder harness not used') |
                                                        (parties_data['party_safety_equipment_2'] == 'lap/shoulder harness not used')).astype(int)
    parties_data['party_passenger_helmet_not_used'] = ((parties_data['party_safety_equipment_1'] == 'passenger, motorcycle helmet not used') |
                                                    (parties_data['party_safety_equipment_2'] == 'passenger, motorcycle helmet not used')).astype(int)
    parties_data['party_driver_helmet_not_used'] = ((parties_data['party_safety_equipment_1'] == 'driver, motorcycle helmet not used') |
                                                    (parties_data['party_safety_equipment_2'] == 'driver, motorcycle helmet not used')).astype(int)
    parties_data['party_child_safety_not_properly_used'] = ((parties_data['party_safety_equipment_1'].isin(['child restraint in vehicle not used', 'child restraint in vehicle, improper use'])) |
                                                            (parties_data['party_safety_equipment_2'].isin(['child restraint in vehicle not used', 'child restraint in vehicle, improper use']))).astype(int)
    parties_data['any_hazardous_materials'] = (parties_data['hazardous_materials'] == 1).astype(int)
    parties_data['any_cellphone_in_use'] = (parties_data['cellphone_in_use'] == 1).astype(int)
    parties_data['any_school_bus_related'] = (parties_data['school_bus_related'] == 1).astype(int)
    parties_data['any_stop_and_go_traffic'] = ((parties_data['other_associate_factor_1'] == 'stop and go traffic') | 
                                            (parties_data['other_associate_factor_2'] == 'stop and go traffic')).astype(int)

    parties_data['any_vision_obscurements'] = ((parties_data['other_associate_factor_1'] == 'vision obscurements') | 
                                            (parties_data['other_associate_factor_2'] == 'vision obscurements')).astype(int)

    parties_data['any_defective_vehicle_equipment'] = ((parties_data['other_associate_factor_1'] == 'defective vehicle equipment') | 
                                                    (parties_data['other_associate_factor_2'] == 'defective vehicle equipment')).astype(int)

    parties_data['any_unfamiliar_with_road'] = ((parties_data['other_associate_factor_1'] == 'unfamiliar with road') | 
                                                (parties_data['other_associate_factor_2'] == 'unfamiliar with road')).astype(int)

    parties_data['any_runaway_vehicle'] = ((parties_data['other_associate_factor_1'] == 'runaway vehicle') | 
                                        (parties_data['other_associate_factor_2'] == 'runaway vehicle')).astype(int)

    parties_data['any_inattention'] = ((parties_data['other_associate_factor_1'] == 'inattention') | 
                                    (parties_data['other_associate_factor_2'] == 'inattention')).astype(int)

    parties_data['any_entering_leaving_ramp'] = ((parties_data['other_associate_factor_1'] == 'entering/leaving ramp') | 
                                                (parties_data['other_associate_factor_2'] == 'entering/leaving ramp')).astype(int)

    parties_data['any_uninvolved_vehicle'] = ((parties_data['other_associate_factor_1'] == 'uninvolved vehicle') | 
                                            (parties_data['other_associate_factor_2'] == 'uninvolved vehicle')).astype(int)

    parties_data['any_previous_collision'] = ((parties_data['other_associate_factor_1'] == 'previous collision') | 
                                            (parties_data['other_associate_factor_2'] == 'previous collision')).astype(int)

    parties_data['any_proceeding_straight'] = (parties_data['movement_preceding_collision'] == 'proceeding straight').astype(int)
    parties_data['any_turning_right'] = (parties_data['movement_preceding_collision'] == 'making right turn').astype(int)
    parties_data['any_turning_left'] = (parties_data['movement_preceding_collision'] == 'making left turn').astype(int)
    parties_data['any_u_turn'] = (parties_data['movement_preceding_collision'] == 'making u-turn').astype(int)
    parties_data['any_parking'] = (parties_data['movement_preceding_collision'] == 'parking maneuver').astype(int)
    parties_data['any_backing'] = (parties_data['movement_preceding_collision'] == 'backing').astype(int)
    parties_data['any_changing_lanes'] = (parties_data['movement_preceding_collision'] == 'changing lanes').astype(int)
    parties_data['any_passing'] = (parties_data['movement_preceding_collision'] == 'passing other vehicle').astype(int)
    parties_data['any_merging'] = (parties_data['movement_preceding_collision'] == 'merging').astype(int)
    parties_data['any_stopped_or_slowing'] = (parties_data['movement_preceding_collision'].isin(['stopped', 'slowing/stopping'])).astype(int)
    parties_data['any_parked'] = (parties_data['movement_preceding_collision'] == 'parked').astype(int)
    parties_data['any_entering_traffic'] = (parties_data['movement_preceding_collision'] == 'entering traffic').astype(int)
    parties_data['any_other_unsafe_turning'] = (parties_data['movement_preceding_collision'] == 'other unsafe turning').astype(int)
    parties_data['any_travelling_wrong_way'] = (parties_data['movement_preceding_collision'] == 'travelling wrong way').astype(int)
    parties_data['any_ran_off_road'] = (parties_data['movement_preceding_collision'] == 'ran off road').astype(int)
    parties_data['any_crossed_into_opposing_lane'] = (parties_data['movement_preceding_collision'] == 'crossed into opposing lane').astype(int)
    parties_data['any_motorcycle'] = (parties_data['statewide_vehicle_type'] == 'motorcycle or scooter').astype(int)
    parties_data['any_bicycle'] = (parties_data['statewide_vehicle_type'] == 'bicycle').astype(int)
    parties_data['any_pedestrian'] = (parties_data['statewide_vehicle_type'] == 'pedestrian').astype(int)
    parties_data['any_emergency_vehicle'] = (parties_data['statewide_vehicle_type'] == 'emergency vehicle').astype(int)
    parties_data['any_bus'] = (parties_data['statewide_vehicle_type'] == 'other bus').astype(int)
    parties_data['any_trailer'] = (parties_data['statewide_vehicle_type'].isin(['pickup or panel truck with trailer', 'truck or truck tractor with trailer', 'passenger car with trailer'])).astype(int)
    parties_data['any_pickup_truck'] = (parties_data['statewide_vehicle_type'] == 'pickup or panel truck').astype(int)
    parties_data['any_highway_construction_equipment'] = (parties_data['statewide_vehicle_type'] == 'highway construction equipment').astype(int)
    parties_data['any_moped'] = (parties_data['statewide_vehicle_type'] == 'moped').astype(int)
    parties_data['any_passenger_car'] = (parties_data['statewide_vehicle_type'] == 'passenger car').astype(int)
    parties_data['any_truck_or_tractor'] = (parties_data['statewide_vehicle_type'] == 'truck or truck tractor').astype(int)
    parties_data['any_asian'] = (parties_data['party_race'] == 'asian').astype(int)
    parties_data['any_black'] = (parties_data['party_race'] == 'black').astype(int)
    parties_data['any_hispanic'] = (parties_data['party_race'] == 'hispanic').astype(int)
    parties_data['any_white'] = (parties_data['party_race'] == 'white').astype(int)

    return parties_data


victims_data = add_binary_flags_victim(victims_data)
parties_data = add_binary_flags_parties(parties_data)

print(victims_data.head())
print(parties_data.head())


# Step 2.2: Aggregate Data

def aggregate_victim_data(victims_data):
    """Aggregates the victim data based on the case_id"""
    victims_agg = victims_data.groupby('case_id').agg({
        'id': 'count',
        'victim_age': lambda x: x[(x >= 0) & (x <= 125)].mean(),
        'contains_passenger': 'max',
        'contains_rear_occupant': 'max',
        'victim_airbag_not_deployed': 'max',
        'victim_lap_belt_not_used': 'max',
        'victim_shoulder_harness_not_used': 'max',
        'victim_passive_restraint_not_used': 'max',
        'victim_lap_shoulder_harness_not_used': 'max',
        'victim_passenger_helmet_not_used': 'max',
        'victim_driver_helmet_not_used': 'max',
        'victim_child_safety_not_properly_used': 'max',
        'any_ejected': 'max'
    }).reset_index().rename(columns={'id': 'num_victims', 'victim_age': 'avg_victim_age'})
    
    return victims_agg

def aggregate_parties_data(parties_data):
    """Aggregates the parties data based on the case_id"""
    parties_agg = parties_data.groupby('case_id').agg({
        'party_age': lambda x: x[(x >= 0) & (x <= 125)].mean(),
        'at_fault': 'sum',
        'all_male': 'max',
        'any_under_alcohol_influence': 'max',
        'any_drug_use': 'max',
        'any_physical_impairment': 'max',
        'any_sleepy': 'max',
        'party_airbag_not_deployed': 'max',
        'party_lap_belt_not_used': 'max',
        'party_shoulder_harness_not_used': 'max',
        'party_passive_restraint_not_used': 'max',
        'party_lap_shoulder_harness_not_used': 'max',
        'party_passenger_helmet_not_used': 'max',
        'party_driver_helmet_not_used': 'max',
        'party_child_safety_not_properly_used': 'max',
        'any_hazardous_materials': 'max',
        'any_cellphone_in_use': 'max',
        'any_school_bus_related': 'max',
        'any_stop_and_go_traffic': 'max',
        'any_vision_obscurements': 'max',
        'any_defective_vehicle_equipment': 'max',
        'any_unfamiliar_with_road': 'max',
        'any_runaway_vehicle': 'max',
        'any_inattention': 'max',
        'any_entering_leaving_ramp': 'max',
        'any_uninvolved_vehicle': 'max',
        'any_previous_collision': 'max',
        'any_proceeding_straight': 'max',
        'any_turning_right': 'max',
        'any_turning_left': 'max',
        'any_u_turn': 'max',
        'any_parking': 'max',
        'any_backing': 'max',
        'any_changing_lanes': 'max',
        'any_passing': 'max',
        'any_merging': 'max',
        'any_stopped_or_slowing': 'max',
        'any_parked': 'max',
        'any_entering_traffic': 'max',
        'any_other_unsafe_turning': 'max',
        'any_travelling_wrong_way': 'max',
        'any_ran_off_road': 'max',
        'any_crossed_into_opposing_lane': 'max',
        'any_motorcycle': 'max',
        'any_bicycle': 'max',
        'any_pedestrian': 'max',
        'any_emergency_vehicle': 'max',
        'any_bus': 'max',
        'any_trailer': 'max',
        'any_pickup_truck': 'max',
        'any_highway_construction_equipment': 'max',
        'any_moped': 'max',
        'any_passenger_car': 'max',
        'any_truck_or_tractor': 'max',
        'any_asian': 'max',
        'any_black': 'max',
        'any_hispanic': 'max',
        'any_white': 'max'
    }).reset_index().rename(columns={'party_age': 'avg_party_age', 'at_fault': 'parties_at_fault'})

    return parties_agg

victims_agg = aggregate_victim_data(victims_data)
parties_agg = aggregate_parties_data(parties_data)

print(victims_agg.head())
print(parties_agg.head())

# 'mean_vehicle_year': lambda x: x[('vehicle_year' >= 1886) & ('vehicle_year' <= 2022)].mean()

# Step 2.3: Merge Aggregated Data with Collisions Data
collisions_data = pd.merge(collisions_data, victims_agg, on='case_id', how='left')
collisions_data = pd.merge(collisions_data, parties_agg, on='case_id', how='left')

print(collisions_data.head())

# Write the preprocessed data to a new SQLite database for further analysis
conn_preprocessed = sqlite3.connect('switrs_db_preprocessed.sqlite')
#collisions_data.to_sql('collisions_preprocessed', conn_preprocessed, index=False)


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

X_processed = preprocessor.fit_transform(data)


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

param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

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

X_processed = preprocessor.fit_transform(data)


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

model = LogisticRegression(class_weight='balanced')
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
              'solver': ['liblinear', 'lbfgs'], 
              'class_weight': ['balanced']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)