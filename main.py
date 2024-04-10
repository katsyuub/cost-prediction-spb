import pandas as pd
import streamlit as st
import seaborn as sns
import json
import joblib

st.header('Предсказание стоимости квартиры в Санкт-Петербурге')

PATH_DATA = "data/processed_data.csv"
PATH_UNIQUE_VALUES = 'data/unique_values.json'
PATH_MODEL = "models/GBR.sav"

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data = data.sample(5000)
    return data

@st.cache_data
def load_model(path):
    model = joblib.load(path)
    return model

@st.cache_data
def transform(data):
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)
    data = data.reset_index(drop=True)
    data["norm_price"] = data["price"] / data["area"]
    data["label_colors"] = pd.qcut(data["norm_price"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)

df = load_data(PATH_DATA)
df = transform(df)

st.map(data=df, latitude="geo_lat", longitude="geo_lon", color='label_colors')

st.markdown(
    """
    ### Описание полей 
        - Тип постройки:
        0: Другое
        1: Панельный
        2: Монолитный
        3: Кирпичный
        4: Блочный
        5: Деревянный
        
        - Тип объекта:
        1: Вторичный рынок недвижимости
        2: Новостройка

"""
)

building_type = st.sidebar.selectbox('Тип постройки', (dict_unique['building_type']))
object_type = st.sidebar.selectbox("Тип объекта", (dict_unique["object_type"]))
level = st.sidebar.slider(
    "Этаж", min_value=min(dict_unique["level"]), max_value=max(dict_unique["level"])
)
levels = st.sidebar.slider(
    "Этажность дома", min_value=min(dict_unique["levels"]), max_value=max(dict_unique["levels"])
)
rooms = st.sidebar.slider(
    "Количество комнат", min_value=min(dict_unique["rooms"]), max_value=max(dict_unique["rooms"])
)    
area = st.sidebar.slider(
    "Площадь", min_value=min(dict_unique["area"]), max_value=max(dict_unique["area"])
)
kitchen_area = st.sidebar.slider(
    "Площадь кухни",
    min_value=min(dict_unique["kitchen_area"]),
    max_value=max(dict_unique["kitchen_area"])
)

dict_data = {
    "building_type": building_type,
    "object_type": object_type,
    "level": level,
    "levels": levels,
    "rooms": rooms,
    "area": area,
    "kitchen_area": kitchen_area,
}

data_predict = pd.DataFrame([dict_data])
model = load_model(PATH_MODEL)

button = st.button("Рассчитать")

if button:
    output = model.predict(data_predict)
    st.success(f"{round(output[0])} rub")
