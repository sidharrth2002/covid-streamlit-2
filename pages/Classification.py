from os import name
from seaborn.matrix import heatmap
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from urllib.error import URLError
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from multipage import MultiPage
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from multipage import MultiPage
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import plotly.figure_factory as ff
from tensorflow.keras.models import load_model
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
# import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def app():
    # =============================================================================
    # LOADING DATA AND PREPROCESSING
    # =============================================================================
    cases_malaysia = pd.read_csv('./cases/epidemic/cases_malaysia.csv')
    cases_state = pd.read_csv('./cases/epidemic/cases_state.csv')
    clusters = pd.read_csv('./cases/epidemic/clusters.csv')
    deaths_malaysia = pd.read_csv('./cases/epidemic/deaths_malaysia.csv')
    deaths_state = pd.read_csv('./cases/epidemic/deaths_state.csv')
    hospital = pd.read_csv('./cases/epidemic/hospital.csv')
    icu = pd.read_csv('./cases/epidemic/icu.csv')
    pkrc = pd.read_csv('./cases/epidemic/pkrc.csv')
    tests_malaysia = pd.read_csv('./cases/epidemic/tests_malaysia.csv')
    tests_state = pd.read_csv('./cases/epidemic/tests_state.csv')
    vax_malaysia = pd.read_csv('./vaccination/vaccination/vax_malaysia.csv')
    vax_state = pd.read_csv('./vaccination/vaccination/vax_state.csv')
    vaxreg_malaysia = pd.read_csv('./vaccination/registration/vaxreg_malaysia.csv')
    vaxreg_state = pd.read_csv('./vaccination/registration/vaxreg_state.csv')
    population = pd.read_csv('./vaccination/static/population.csv')
    checkins = pd.read_csv('./cases/mysejahtera/checkin_malaysia.csv')
    income = pd.read_csv('./vaccination/static/income.csv')
    global_datasets =  pd.read_csv('./global_datasets/owid-covid-data.csv')
    aefi = pd.read_csv('./cases/vaccination/aefi.csv')
    trace_malaysia = pd.read_csv('./cases/mysejahtera/trace_malaysia.csv')
    trace_malaysia.fillna(0,inplace=True)
    trace_malaysia.drop_duplicates(inplace=True)

    # cluster columns are irrelevant, remove them
    cases_malaysia.drop(columns=['cluster_import', 'cluster_religious', 'cluster_community', 'cluster_highRisk', 'cluster_education', 'cluster_detentionCentre', 'cluster_workplace'], inplace=True)
    # other dates with a null value, just drop that row
    cases_malaysia.fillna(0, inplace=True)
    # cases_malaysia.head()
    cases_state.drop_duplicates(inplace=True)
    cases_state.fillna(0, inplace=True)
    cases_state_pivoted = cases_state.pivot(index='date', columns='state', values='cases_new')
    clusters.drop_duplicates(inplace=True)
    deaths_malaysia.drop_duplicates(inplace=True)
    deaths_malaysia.drop(columns=['deaths_bid', 'deaths_new_dod', 'deaths_bid_dod', 'deaths_pvax', 'deaths_fvax', 'deaths_tat'], inplace=True)
    deaths_state.drop_duplicates(inplace=True)
    deaths_state_pivoted = deaths_state.pivot(index='date', columns='state', values='deaths_new')
    hospital.drop_duplicates(inplace=True)
    hospital.drop(columns=['beds', 'beds_noncrit', 'admitted_pui', 'admitted_total', 'discharged_pui', 'discharged_total','hosp_pui','hosp_noncovid'], inplace=True)
    icu.drop_duplicates(inplace=True)
    icu.drop(columns=['beds_icu', 'beds_icu_rep', 'beds_icu_total', 'vent', 'vent_port', 'icu_pui','icu_noncovid','vent_pui','vent_noncovid','vent_used','vent_port_used'], inplace=True)
    pkrc.drop_duplicates(inplace=True)
    pkrc.drop(columns=['beds', 'admitted_pui', 'admitted_total', 'discharge_pui', 'discharge_total', 'pkrc_pui','pkrc_noncovid'], inplace=True)
    tests_malaysia.drop_duplicates(inplace=True)
    tests_malaysia['total_testing'] = tests_malaysia['rtk-ag'] + tests_malaysia['pcr']
    tests_malaysia.drop(columns=['rtk-ag', 'pcr'], inplace=True)
    vax_malaysia.drop_duplicates(inplace=True)
    vax_malaysia['cumul_vaccine'] = vax_malaysia['daily_full'].cumsum()
    vax_malaysia_all_attributes = vax_malaysia.copy()
    # total up first and second dose
    vax_malaysia_all_attributes['pfizer'] = vax_malaysia_all_attributes['pfizer1'] + vax_malaysia_all_attributes['pfizer2']
    vax_malaysia_all_attributes['astra'] = vax_malaysia_all_attributes['astra1'] + vax_malaysia_all_attributes['astra2']
    vax_malaysia_all_attributes['sinovac'] = vax_malaysia_all_attributes['sinovac1'] + vax_malaysia_all_attributes['sinovac2']
    vax_malaysia.drop(columns=['daily_partial_child','cumul_partial','cumul_full','cumul','cumul_partial_child','cumul_full_child','pfizer1','pfizer2','sinovac1','sinovac2','astra1','astra2','cansino','pending'], inplace=True)
    vax_state.drop_duplicates(inplace=True)
    vax_state.drop(columns=['daily_partial_child', 'daily_full_child','cumul_partial','cumul_full','cumul','cumul_partial_child','cumul_full_child','pfizer1','pfizer2','sinovac1','sinovac2','astra1','astra2','cansino','pending'], inplace=True)
    vaxreg_malaysia.drop_duplicates(inplace=True)
    vaxreg_malaysia.drop(columns=['phase2', 'mysj','call','web','children','elderly','comorb','oku'], inplace=True)
    vaxreg_state.drop_duplicates(inplace=True)
    vaxreg_state.drop(columns=['phase2', 'mysj','call','web','children','elderly','comorb','oku'], inplace=True)
    population.drop(columns=['pop_18', 'pop_60'], inplace=True)
    income = income[income['Year'] == 2020]
    income.rename(columns={'Country/State': 'state', 'Mean Monthly Household Gross Income': 'income', 'Year': 'year'}, inplace=True)
    global_datasets.fillna(0, inplace=True)
    global_datasets.drop_duplicates(inplace=True)
    global_datasets.drop(columns=['iso_code', 'continent','new_cases_smoothed','new_deaths_smoothed','new_cases_smoothed_per_million',
                                'new_deaths_smoothed_per_million','reproduction_rate','icu_patients','icu_patients_per_million','hosp_patients',
                                'hosp_patients_per_million','weekly_icu_admissions','weekly_icu_admissions_per_million','weekly_hosp_admissions',
                                'weekly_hosp_admissions_per_million','new_tests_smoothed','total_boosters','new_vaccinations_smoothed',
                                'total_boosters_per_hundred','new_vaccinations_smoothed_per_million','stringency_index','median_age',
                                'aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence',
                                'female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand','life_expectancy','human_development_index',
                                'excess_mortality_cumulative_absolute','excess_mortality_cumulative','excess_mortality','excess_mortality_cumulative_per_million',
                                ], inplace=True)

    st.markdown('''
    ## Classification
    ''')
    st.markdown('''
    ### Can we classify individual check-ins in Malaysia into groups (Low, Medium and High)?
    We first do feature selection using Boruta, SMOTE the dataset then evaluate Random Forest Classifier, Logistic Regression and the Naive Bayes classifier.
    ''')
    cases_testing_deaths_vax_checkins = cases_malaysia.merge(tests_malaysia, on='date')
    cases_testing_deaths_vax_checkins = cases_testing_deaths_vax_checkins.merge(deaths_malaysia, on='date')
    cases_testing_deaths_vax_checkins = cases_testing_deaths_vax_checkins.merge(vax_malaysia[['date', 'daily_full']], on='date')
    cases_testing_deaths_vax_checkins = cases_testing_deaths_vax_checkins.merge(checkins[['date', 'unique_ind']], on='date')

    cases_testing_deaths_vax_checkins['ind_checkins_class'] = pd.cut(cases_testing_deaths_vax_checkins['unique_ind'], 3, labels=['Low', 'Medium', 'High'])
    cases_testing_deaths_vax_checkins.drop(['unique_ind'], axis=1, inplace=True)

    features = ["cases_new", "cases_import",	"cases_recovered", "cases_active", "cases_cluster",	"cases_pvax", "cases_fvax",	"cases_child","cases_adolescent", "cases_adult", "cases_elderly", "total_testing", "deaths_new", "daily_full", "ind_checkins_class"]

    filtered = cases_testing_deaths_vax_checkins[features]
    filtered['date'] = cases_testing_deaths_vax_checkins['date']
    filtered['date'] = pd.to_datetime(filtered['date'])
    filtered.set_index('date', inplace=True)
    # SMOTE dataset
    X_scaler = MinMaxScaler()
    X = filtered.drop(columns=['ind_checkins_class'])
    X_scaled = X_scaler.fit_transform(X)
    y = filtered['ind_checkins_class']
    smt = SMOTE(random_state=42, k_neighbors=3)
    X_smt, y_smt = smt.fit_resample(X_scaled, y)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.2, random_state=42)

    classification_model = st.selectbox('Which classification model do you want to test?', ['Random Forest Classifier', 'Logistic Regression', 'Naive Bayes'])

    if classification_model == 'Random Forest Classifier':
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        rf.fit(X_train, y_train)
        # get score
        accuracy = rf.score(X_test, y_test)
        # F1-Score
        f1 = f1_score(y_test, rf.predict(X_test), average='weighted')

        st.write(f"Accuracy of Random Forest: {accuracy}")
        st.write(f"Weighted Averaged F1-Score of Random Forest: {f1}")

        y_pred = rf.predict(X_test)

        # plot confusion matrix with plotly
        cf = ff.create_annotated_heatmap(z=confusion_matrix(y_test, y_pred), x=['High', 'Medium', 'Low'], y=['True High', 'True Medium', 'True Low'], annotation_text=confusion_matrix(y_test, y_pred), colorscale='Viridis', showscale=True)
        st.plotly_chart(cf)

    elif classification_model == 'Logistic Regression':
        log = LogisticRegression()
        log.fit(X_train, y_train)
        accuracy = log.score(X_test, y_test)
        f1 = f1_score(y_test, log.predict(X_test), average='weighted')

        st.write(f"Accuracy of Logistic Regression: {accuracy}")
        st.write(f"Weighted Averaged F1-Score of Logistic Regression: {f1}")

        # classification report
        y_pred = log.predict(X_test)

        # plot confusion matrix with plotly
        cf = ff.create_annotated_heatmap(z=confusion_matrix(y_test, y_pred).T, x=['High', 'Medium', 'Low'], y=['True High', 'True Medium', 'True Low'], annotation_text=confusion_matrix(y_test, y_pred).T, colorscale='Viridis', showscale=True)
        st.plotly_chart(cf)

    else:
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        accuracy = gnb.score(X_test, y_test)
        f1 = f1_score(y_test, gnb.predict(X_test), average='weighted')

        st.write(f"Accuracy of Naive Bayes: {accuracy}")
        st.write(f"Weighted Averaged F1-Score of Naive Bayes: {f1}")

        # classification report
        y_pred = gnb.predict(X_test)

        # plot confusion matrix with plotly
        cf = ff.create_annotated_heatmap(z=confusion_matrix(y_test, y_pred).T, x=['High', 'Medium', 'Low'], y=['True High', 'True Medium', 'True Low'], annotation_text=confusion_matrix(y_test, y_pred).T, colorscale='Viridis', showscale=True)
        st.plotly_chart(cf)

    st.markdown('''
    ### Can we predict the type of vaccine based on the symptoms?
    Some vaccines produce more of a certain symptom than others. Hence, would it be possible to predict whether the vaccine is Pfizer, Sinovac, Astra, etc. based purely on the symptoms reported each day.
    We use self-reported symptoms for each vaccine daily as the training data. Appropriate hyperparameter tuning is done using GridSearchCV for the Random Forest Classifier. Both Logistic Regression and the Support Vector Classifier are evaluated for this question using the metrics accuracy and weighted averaged F1-Score. The training set is SMOTE-d.

    Feature selection (symptoms) is done using Recursive Feature Elimination.''')
    vaccine_prediction = aefi.copy()
    vaccine_prediction['vaxtype_label'] = LabelEncoder().fit_transform(vaccine_prediction['vaxtype'])
    vaccine_prediction.drop(columns=['daily_total'], inplace=True)

    
    X_scaler = MinMaxScaler()
    X = vaccine_prediction.drop(columns=['date', 'vaxtype', 'vaxtype_label'])
    X_scaled = X_scaler.fit_transform(X)
    y = vaccine_prediction['vaxtype_label']

    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X_scaled, y)

    X_transformed = pd.DataFrame(rfe.transform(X_scaled), columns=X.columns[rfe.support_])
    
    y_encoder = LabelEncoder()

    X = vaccine_prediction.drop(columns=['date', 'vaxtype', 'vaxtype_label'])
    y = y_encoder.fit_transform(vaccine_prediction['vaxtype'])

    features = ['daily_nonserious_mysj', 'daily_nonserious_npra', 'daily_serious_npra', 'daily_nonserious_mysj_dose1', 'd1_site_pain', 'd1_site_swelling', 'd1_site_redness', 'd1_headache', 'd1_muscle_pain', 'd1_joint_pain', 'd1_weakness', 'd1_fever', 'd1_chills', 'd1_rash', 'd2_site_pain', 'd2_site_swelling', 'd2_headache', 'd2_joint_pain', 'd2_fever', 'd2_chills']

    X_transformed = X[features]
    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_transformed)
    # X_transformed = pd.DataFrame(rfe.transform(X_scaled), columns=X.columns[rfe.support_])

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    smt = SMOTE(random_state=42, k_neighbors=3)
    X_smt, y_smt = smt.fit_resample(X_train, y_train)

    classification_model2 = st.selectbox('Which classification model do you want to test?', ['Logistic Regression', 'Support Vector Classification'])

    if classification_model2 == 'Logistic Regression':
        logreg = LogisticRegression()
        logreg.fit(X_smt, y_smt)
        accuracy = logreg.score(X_test, y_test)
        f1 = f1_score(y_test, logreg.predict(X_test), average='weighted')
        st.write(f"Accuracy of Logistic Regression: {accuracy}")
        st.write(f"Weighted Averaged F1-Score of Logistic Regression: {f1}")

        y_pred = logreg.predict(X_test)

        # confusion matrix
        cf = ff.create_annotated_heatmap(z=confusion_matrix(y_test, y_pred).T, x=['Pfizer', 'Sinovac', 'Astrazeneca', 'Cansino'], y=['True Pfizer', 'True Sinovac', 'True Astrazeneca', 'True Cansino'], annotation_text=confusion_matrix(y_test, y_pred).T, colorscale='Viridis', showscale=True)
        st.plotly_chart(cf)

    elif classification_model2 == 'Support Vector Classification':
        # defining parameter range
        best_params = {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf']}
 
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
        # fitting the model for grid search
        grid.fit(X_smt, y_smt)
        svc = SVC(**{'C': 1000, 'gamma': 1, 'kernel': 'rbf'})
        svc = SVC(**best_params)
        svc.fit(X_smt, y_smt)

        st.write(f'Best Model {svc}')
        accuracy = svc.score(X_test, y_test)
        f1 = f1_score(y_test, svc.predict(X_test), average='weighted')
        st.write(f"Accuracy of Support Vector Regression: {accuracy}")
        st.write(f"Weighted Averaged F1-Score of Support Vector Regression: {f1}")

        y_pred = svc.predict(X_test)

        # confusion matrix
        cf = ff.create_annotated_heatmap(z=confusion_matrix(y_test, y_pred).T, x=['Pfizer', 'Sinovac', 'Astrazeneca', 'Cansino'], y=['True Pfizer', 'True Sinovac', 'True Astrazeneca', 'True Cansino'], annotation_text=confusion_matrix(y_test, y_pred).T, colorscale='Viridis', showscale=True)
        st.plotly_chart(cf)