import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from sklearn.metrics import accuracy_score, confusion_matrix


models = {
    "Logistic Regression": jb.load('./models/logistic_regression.pkl'),
    "K-Nearest Neighbors": jb.load('./models/k_nearest_neighbors.pkl'),
    "Support Vector Machine": jb.load('./models/support_vector_machine.pkl'),
    "Decision Tree": jb.load('./models/decision_tree.pkl'),
    "Random Forest": jb.load('./models/random_forest.pkl'),
    "Gradient Boosting": jb.load('./models/gradient_boosting.pkl'),
    "AdaBoost": jb.load('./models/ada_boost.pkl'),
    "Gaussian Naive Bayes": jb.load('./models/gaussian_naive_bayes.pkl'),
    "XGBoost": jb.load('./models/x_g_boost.pkl'),
}

grade_class = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'F',
}

option = st.selectbox('請選擇模型', models.keys())

def use_form():
    with st.form('form'):
        st.header("學習狀況")

        study_time_weekly = st.slider('每周讀書時間（小時）：', max_value=40)
        absences = st.checkbox('是否翹課')
        parental_support = st.checkbox('家長是否會協助')

        submitted = st.form_submit_button("確認", use_container_width=True)
        if submitted:
            model = models[option]
            grade = model.predict(pd.DataFrame({
                'Absences': [absences],
                'StudyTimeWeekly': [study_time_weekly],
                'ParentalSupport': [parental_support]
            }))[0]
            st.write("成績預測：", grade_class[grade])


def use_file():
    with st.form('form'):
        st.header("上傳資料")
        uploaded_file = st.file_uploader("選擇檔案", type=['csv'])
        
        submitted = st.form_submit_button("確認", use_container_width=True)
        if submitted and uploaded_file is not None:
            model = models[option]
            df = pd.read_csv(uploaded_file)
            X_test = df.loc[:,['Absences', 'StudyTimeWeekly', 'ParentalSupport']]
            y_test = df.loc[:, ['GradeClass']]
            y_pred = pd.DataFrame({'預測成績': model.predict(X_test)})
            result = df.loc[:, ['StudentID', 'GradeClass']].join(y_pred)
            score = round(accuracy_score(y_test, y_pred), 3)
            cm = confusion_matrix(y_test, y_pred)
            plt.title('Accuracy Score: {0}'.format(score), size=15)
            sns.heatmap(cm, annot=True, fmt=".0f")
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(result)
            with col2: 
                st.pyplot(plt)


page_names_to_funcs = {
    '表單': use_form,
    '上傳': use_file
}

demo_name = st.sidebar.selectbox('選擇操作類型', page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
