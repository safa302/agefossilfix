import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load data
data = pd.read_csv('Age_Fossil.csv')

# Handle categorical data with Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['geological_period', 'paleomagnetic_data', 'surrounding_rock_type', 'stratigraphic_position']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Features and target
X = data.drop('age', axis=1)
y = data['age']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# Streamlit App
st.title('Fossil Age Prediction')

# Input form
uranium_lead_ratio = st.number_input('Uranium Lead Ratio', min_value=0.0)
carbon_14_ratio = st.number_input('Carbon 14 Ratio', min_value=0.0)
radioactive_decay_series = st.number_input('Radioactive Decay Series', min_value=0.0)
stratigraphic_layer_depth = st.number_input('Stratigraphic Layer Depth', min_value=0.0)
geological_period = st.selectbox('Geological Period', data['geological_period'].unique())
paleomagnetic_data = st.selectbox('Paleomagnetic Data', data['paleomagnetic_data'].unique())
inclusion_of_other_fossils = st.selectbox('Inclusion of Other Fossils', ['True', 'False'])
isotopic_composition = st.number_input('Isotopic Composition', min_value=0.0)
surrounding_rock_type = st.selectbox('Surrounding Rock Type', data['surrounding_rock_type'].unique())
stratigraphic_position = st.selectbox('Stratigraphic Position', data['stratigraphic_position'].unique())
fossil_size = st.number_input('Fossil Size', min_value=0.0)
fossil_weight = st.number_input('Fossil Weight', min_value=0.0)

# Encode categorical inputs
geological_period_encoded = label_encoder.transform([geological_period])[0]
paleomagnetic_data_encoded = label_encoder.transform([paleomagnetic_data])[0]
surrounding_rock_type_encoded = label_encoder.transform([surrounding_rock_type])[0]
stratigraphic_position_encoded = label_encoder.transform([stratigraphic_position])[0]
inclusion_of_other_fossils_encoded = 1 if inclusion_of_other_fossils == 'True' else 0

# Prepare input data
input_data = np.array([[uranium_lead_ratio, carbon_14_ratio, radioactive_decay_series,
                        stratigraphic_layer_depth, geological_period_encoded, paleomagnetic_data_encoded,
                        inclusion_of_other_fossils_encoded, isotopic_composition,
                        surrounding_rock_type_encoded, stratigraphic_position_encoded,
                        fossil_size, fossil_weight]])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button('Predict Fossil Age'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Predicted Age of Fossil: {prediction[0]:.2f} years')
