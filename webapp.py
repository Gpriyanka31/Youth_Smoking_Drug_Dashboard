import numpy as np
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

df = pd.read_csv("youth_smoking_drug_data_10000_rows_expanded.csv")

with open('models/smoking_model.sav', 'rb') as file:
    smoking_model = pickle.load(file)

with open('models/drug_model.sav', 'rb') as file:
    drug_model = pickle.load(file)

# Streamlit App Interface
st.title("Youth Smoking and Drug Use Prediction")
st.write("Predict smoking prevalence and classify drug experimentation risk.")

# User Inputs
st.header("Enter demographic and health data")

# Age Group mapping
age_group_mapping = {
    "10-14": 1,
    "15-19": 2,
    "20-24": 3,
    "25-29": 4,
    "30-39": 5,
    "40-49": 6,
    "50-59": 7,
    "60-69": 8,
    "70-79": 9,
    "80+": 10
}

# Input Widgets
year = st.selectbox("Year", ["2020", "2021", "2022", "2023", "2024"])
age = st.selectbox("Age_Group", list(age_group_mapping.keys()))
gender = st.selectbox("Gender", ["Male", "Female", "Both"])
socioeconomic_status = st.selectbox("Socioeconomic_Status", ["Low", "Medium", "High"])
peer_influence = st.slider("Peer_Influence", 0, 10, 5)
school_programs = st.selectbox("School_Programs", ["Yes", "No"])
family_background = st.slider("Family_Background", 0, 10, 5)
mental_health = st.slider("Mental_Health", 0, 100, 50)
access_to_counseling = st.selectbox("Access_to_Counseling", ["Yes", "No"])
parental_supervision = st.slider("Parental_Supervision", 0, 10, 5)
substance_education = st.slider("Substance_Education", 0, 10, 5)
community_support = st.slider("Community_Support", 0, 10, 5)
media_influence = st.slider("Media_Influence", 0, 10, 5)

# Data Preparation for Prediction
input_data = pd.DataFrame({
    'Year': [year],
    'Age_Group': [age_group_mapping[age]],  # Map age group to ordinal values
    'Gender': [gender],
    'Socioeconomic_Status': [socioeconomic_status],
    'Peer_Influence': [peer_influence],
    'School_Programs': [1 if school_programs == "Yes" else 0],
    'Family_Background': [family_background],
    'Mental_Health': [mental_health],
    'Access_to_Counseling': [1 if access_to_counseling == "Yes" else 0],
    'Parental_Supervision': [parental_supervision],
    'Substance_Education': [substance_education],
    'Community_Support': [community_support],
    'Media_Influence': [media_influence]
})

# Encoding Gender
le = LabelEncoder()
input_data['Gender'] = le.fit_transform(input_data['Gender'])

# Ordinal Encoding for Socioeconomic Status
encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
input_data[['Socioeconomic_Status']] = encoder.fit_transform(input_data[['Socioeconomic_Status']])

# Apply scaling to numerical features
scaler = StandardScaler()
numerical_features = ['Peer_Influence', 'Parental_Supervision', 'Mental_Health',
                      'Family_Background', 'Community_Support', 'Media_Influence']
input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

# Display the scaled input data for verification
st.write("Scaled Input Data for Prediction:")
st.write(input_data)

# Add a Prediction button
# Add a Prediction button
if st.button("Predict Smoking Prevalence"):
    smoking_prevalence = smoking_model.predict(input_data)[0]
    st.write(f"Predicted Smoking Prevalence: {smoking_prevalence:.2f}%")

if st.button("Classify Drug Experimentation Risk"):
    drug_experimentation_risk = drug_model.predict(input_data)[0]
    risk_level = "High Risk" if drug_experimentation_risk == 1 else "Low Risk"
    st.write(f"Drug Experimentation Risk Level: {risk_level}")  

import plotly.express as px

st.header("Exploratory Data Analysis Insights")

# 1. Distribution of Smoking Prevalence by Age Group
st.subheader("Distribution of Smoking Prevalence by Age Group")
fig_age = px.histogram(df, x="Age_Group", y="Smoking_Prevalence", color="Age_Group", 
                       title="Smoking Prevalence Distribution by Age Group",
                       labels={'Age_Group': 'Age Group', 'Smoking_Prevalence': 'Smoking Prevalence'},
                       template="plotly_white")
st.plotly_chart(fig_age)

# 2. Drug Experimentation Risk by Socioeconomic Status
st.subheader("Drug Experimentation Risk by Socioeconomic Status")
fig_ses = px.box(df, x="Socioeconomic_Status", y="Drug_Experimentation", color="Socioeconomic_Status",
                 title="Drug Experimentation by Socioeconomic Status",
                 labels={'Socioeconomic_Status': 'Socioeconomic Status', 'Drug_Experimentation': 'Drug Experimentation'},
                 template="plotly_white")
st.plotly_chart(fig_ses)

from sklearn.preprocessing import LabelEncoder

# Apply label encoding to the 'Gender' column
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# If there are other categorical columns, you can encode them similarly
# Example: Encode 'Socioeconomic_Status' with OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[["Low", "Middle", "High"]])
df[['Socioeconomic_Status']] = encoder.fit_transform(df[['Socioeconomic_Status']])

# Ensure 'Age_Group' is also numeric (if not already done)
df['Age_Group'] = df['Age_Group'].map(age_group_mapping)

# Now select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numeric columns
correlation_matrix = numeric_df.corr()

# Plot the correlation heatmap
import plotly.express as px
fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                     title="Correlation between Factors",
                     labels={'color': 'Correlation Coefficient'})
st.plotly_chart(fig_corr)

# 4. Smoking Prevalence by Gender and Age Group
st.subheader("Smoking Prevalence by Gender and Age Group")
fig_gender_age = px.bar(df, x="Age_Group", y="Smoking_Prevalence", color="Gender", barmode="group",
                        title="Smoking Prevalence by Gender and Age Group",
                        labels={'Smoking_Prevalence': 'Smoking Prevalence', 'Age_Group': 'Age Group'},
                        template="plotly_white")
st.plotly_chart(fig_gender_age)

# 5. Peer Influence vs. Smoking Prevalence (Scatter Plot)
st.subheader("Peer Influence vs Smoking Prevalence")
fig_peer = px.scatter(df, x="Peer_Influence", y="Smoking_Prevalence", color="Age_Group",
                      title="Peer Influence vs Smoking Prevalence by Age Group",
                      labels={'Peer_Influence': 'Peer Influence', 'Smoking_Prevalence': 'Smoking Prevalence'},
                      template="plotly_white")
st.plotly_chart(fig_peer)

