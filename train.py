import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 2211
PATH_DATA = "data/processed_data.csv"
PATH_UNIQUE_VALUES = "data/unique_values.json"
PATH_MODEL = "models/GBR.sav"

drop_cols = ["date", "time", "geo_lat", "geo_lon", "region"]
categorical_features = ["building_type", "object_type"]
numeric_features = ["level", "levels", "rooms", "area", "kitchen_area"]

df = pd.read_csv(PATH_DATA)
df = df.drop(columns=drop_cols)

y = df["price"]
X = df.drop(columns="price", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
)

params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE  
}

GBR = make_pipeline(preprocessor, GradientBoostingRegressor(**params))

GBR.fit(X_train, y_train)
y_prediction = GBR.predict(X_test)

print(mean_absolute_error(y_test, y_prediction))

joblib.dump(GBR, PATH_MODEL)

dict_unique = {key: X[key].unique().tolist() for key in X.columns}

with open(PATH_UNIQUE_VALUES, "w") as file:
    json.dump(dict_unique, file)
