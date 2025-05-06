import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Diabetes Health Indicators Analysis")

# Read dataset directly from GitHub URL
github_url = "https://raw.githubusercontent.com/Kamel-27/Data-computation/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
try:
    df = pd.read_csv(github_url)
    data = df.copy()

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.write("**Dataset Shape:**", df.shape)

    st.write("**Null values per column:**")
    st.write(df.isnull().sum())

    st.write("**Unique values per column:**")
    st.write(df.nunique())

    st.write("**Data Types Before Conversion:**")
    st.write(df.dtypes)

    # Type conversion
    cols_to_convert = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
                       "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                       "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth",
                       "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

    for col in cols_to_convert:
        if col in data.columns:
            data[col] = data[col].astype(int)

    st.write("**Data Types After Conversion:**")
    st.write(data.dtypes)

    # Visualizations
    st.subheader("Diabetes Outcome Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Diabetes_binary', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Modeling
    st.subheader("LDA + SVM Model")
    Y = data['Diabetes_binary']
    X = data.drop(columns=['Diabetes_binary'])

    top_features = ['HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'GenHlth',
                    'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']
    X_selected = X[top_features]

    X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train, Y_train)
    X_test_lda = lda.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lda)
    X_test_scaled = scaler.transform(X_test_lda)

    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, Y_train)
    Y_pred = svm.predict(X_test_scaled)

    st.write("**SVM Accuracy after LDA:**", accuracy_score(Y_test, Y_pred))
    st.text("Classification Report:\n" + classification_report(Y_test, Y_pred))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(Y_test, Y_pred)))

    # Prediction from user input
    st.subheader("Predict Diabetes for New Input")
    with st.form("prediction_form"):
        HighBP = st.selectbox("HighBP", [0, 1])
        HighChol = st.selectbox("HighChol", [0, 1])
        BMI = st.number_input("BMI", value=25.0)
        HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)
        MentHlth = st.slider("Mental Health (Days)", 0, 30, 5)
        PhysHlth = st.slider("Physical Health (Days)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
        Age = st.slider("Age Category (0=18-24, 13=80+)", 0, 13, 9)
        Income = st.slider("Income Category (1=<$10K, 8=$75K+)", 1, 8, 4)

        submit = st.form_submit_button("Predict")

        if submit:
            new_record = pd.DataFrame([{ 'HighBP': HighBP, 'HighChol': HighChol, 'BMI': BMI,
                                         'HeartDiseaseorAttack': HeartDiseaseorAttack, 'GenHlth': GenHlth,
                                         'MentHlth': MentHlth, 'PhysHlth': PhysHlth, 'DiffWalk': DiffWalk,
                                         'Age': Age, 'Income': Income }])

            new_lda = lda.transform(new_record)
            new_scaled = scaler.transform(new_lda)
            prediction = svm.predict(new_scaled)

            st.success(f"Predicted Diabetes Class: {prediction[0]}")

except Exception as e:
    st.error(f"Failed to load dataset from GitHub. Error: {e}")
