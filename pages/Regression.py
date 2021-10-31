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
# import Ridge
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor


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

    before_pp_cases_malaysia = cases_malaysia.copy()
    before_pp_cases_state = cases_state.copy()
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
    before_pp_deaths_malaysia = deaths_malaysia.copy()
    deaths_malaysia.drop(columns=['deaths_bid', 'deaths_new_dod', 'deaths_bid_dod', 'deaths_pvax', 'deaths_fvax', 'deaths_tat'], inplace=True)
    before_pp_deaths_state = deaths_state.copy()
    deaths_state.drop_duplicates(inplace=True)
    deaths_state_pivoted = deaths_state.pivot(index='date', columns='state', values='deaths_new')
    hospital.drop_duplicates(inplace=True)
    hospital.drop(columns=['beds', 'beds_noncrit', 'admitted_pui', 'admitted_total', 'discharged_pui', 'discharged_total','hosp_pui','hosp_noncovid'], inplace=True)
    icu.drop_duplicates(inplace=True)
    icu.drop(columns=['beds_icu', 'beds_icu_rep', 'beds_icu_total', 'vent', 'vent_port', 'icu_pui','icu_noncovid','vent_pui','vent_noncovid','vent_used','vent_port_used'], inplace=True)
    pkrc.drop_duplicates(inplace=True)
    pkrc.drop(columns=['beds', 'admitted_pui', 'admitted_total', 'discharge_pui', 'discharge_total', 'pkrc_pui','pkrc_noncovid'], inplace=True)
    before_pp_tests_malaysia = tests_malaysia.copy()
    tests_malaysia.drop_duplicates(inplace=True)
    tests_malaysia['total_testing'] = tests_malaysia['rtk-ag'] + tests_malaysia['pcr']
    tests_malaysia.drop(columns=['rtk-ag', 'pcr'], inplace=True)
    before_pp_vax_malaysia = vax_malaysia.copy()
    vax_malaysia.drop_duplicates(inplace=True)
    vax_malaysia['cumul_vaccine'] = vax_malaysia['daily_full'].cumsum()
    vax_malaysia_all_attributes = vax_malaysia.copy()
    # total up first and second dose
    vax_malaysia_all_attributes['pfizer'] = vax_malaysia_all_attributes['pfizer1'] + vax_malaysia_all_attributes['pfizer2']
    vax_malaysia_all_attributes['astra'] = vax_malaysia_all_attributes['astra1'] + vax_malaysia_all_attributes['astra2']
    vax_malaysia_all_attributes['sinovac'] = vax_malaysia_all_attributes['sinovac1'] + vax_malaysia_all_attributes['sinovac2']
    vax_malaysia.drop(columns=['daily_partial_child','cumul_partial','cumul_full','cumul','cumul_partial_child','cumul_full_child','pfizer1','pfizer2','sinovac1','sinovac2','astra1','astra2','cansino','pending'], inplace=True)
    before_pp_vax_state = vax_state.copy()
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
    before_pp_tests_state = tests_state.copy()
    st.markdown('''
    ## Regression
    ''')
    st.markdown('''
        ### Can we predict the daily vaccination numbers in a multivariate manner?
        Assuming that vaccination numbers depend on various external factors such as government incentives, spikes in cases, etc., could we predict the daily vaccination numbers using time-series regression models?
        The number of people being vaccinated daily can depend on a lot of factors
        To answer this, we test:\n
        1. Multivariate LSTM time-series analysis\n
        2. Multivariate Support Vector Regression
    ''')

    # vax_population = vax_malaysia.merge(population, on='state')
    malaysia_population = population[population['state'] == 'Malaysia']['pop'].iloc[0]
    vax_malaysia['cumul'] = vax_malaysia['daily_full'].cumsum()
    vax_malaysia['percentage_vaccinated'] = vax_malaysia['cumul'] / malaysia_population
    cases_testing_deaths_vax = cases_malaysia.merge(tests_malaysia, on='date')
    cases_testing_deaths_vax = cases_testing_deaths_vax.merge(deaths_malaysia, on='date')
    cases_testing_deaths_vax = cases_testing_deaths_vax.merge(vax_malaysia[['date', 'daily']], on='date')
    icu_covid = pd.DataFrame(icu.groupby('date')['icu_covid'].sum()).reset_index()
    cases_testing_deaths_vax = cases_testing_deaths_vax.merge(icu_covid, on='date')
    cases_testing_deaths_vax['cumul'] = cases_testing_deaths_vax['daily'].cumsum()

    features = ["cases_recovered", "cases_active", "cases_cluster",	"cases_pvax", "cases_fvax", "total_testing", "deaths_new", "icu_covid", "daily"]

    feat_display = st.multiselect('Optimal Feature Set', features, default=features)

    num_features = len(features) - 1
    filtered = cases_testing_deaths_vax[features]
    filtered['date'] = cases_testing_deaths_vax['date']
    filtered['date'] = pd.to_datetime(filtered['date'])
    filtered.set_index('date', inplace=True)

    time_series_model = st.selectbox('Select a time-series model', options=['LSTM', 'SVR'])

    if time_series_model == 'LSTM':
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        input_data = X_scaler.fit_transform(filtered.iloc[:,:-1])
        input_y = y_scaler.fit_transform(filtered.iloc[:,-1].values.reshape(-1,1))
        input_data = np.concatenate((input_data, input_y), axis=1)

        lookback = 100
        total_size = input_data.shape[0]

        X=[]
        y=[]
        for i in range(0, total_size - lookback):
            t = []
            for j in range(0, lookback):
                current_index = i+j
                t.append(input_data[current_index, :-1])
            X.append(t)
            y.append(input_data[lookback+i, num_features])
        X, y = np.array(X), np.array(y)

        test_size = 50
        X_test = X[-test_size:]
        y_test = y[-test_size:]

        X_rest = X[:-test_size]
        y_rest = y[:-test_size]

        X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size=0.15, random_state=42)

        X_train = X_train.reshape(X_train.shape[0], lookback, num_features)
        X_valid = X_valid.reshape(X_valid.shape[0], lookback, num_features)
        X_test = X_test.reshape(X_test.shape[0], lookback, num_features)

        model = load_model('lstm-time-series.h5')
        predicted_vaccination = model.predict(X_test)

        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predicted_vaccination)}")

        predicted_vaccination = y_scaler.inverse_transform(predicted_vaccination)

        # line plot
        time_series, _ = plt.subplots(1,1)
        ax = time_series.add_subplot(1, 1, 1)
        ax.plot(list(range(len(predicted_vaccination))), predicted_vaccination, label='Predicted', color='blue')
        ax.plot(list(range(len(y_test))), y_scaler.inverse_transform(y_test.reshape(-1,1)), label='Actual', color='red')
        time_series.legend()
        st.pyplot(time_series)

    elif time_series_model == 'SVR':
        X_scaler_svr = MinMaxScaler()
        y_scaler_svr = MinMaxScaler()
        X_svr = X_scaler_svr.fit_transform(filtered.iloc[:,:-1])
        y_svr = y_scaler_svr.fit_transform(filtered.iloc[:,-1].values.reshape(-1,1))

        X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size=0.15, random_state=42, shuffle=True)

        params = {'C': 10, 'degree': 2, 'kernel': 'poly'}
        svr = SVR(**params)
        svr.fit(X_train, y_train)

        predicted_vaccination = svr.predict(X_test)
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predicted_vaccination)}")
        predicted_vaccination = y_scaler_svr.inverse_transform(predicted_vaccination.reshape(-1,1))

        # line plot
        time_series2, _ = plt.subplots(1,1)
        ax = time_series2.add_subplot(1, 1, 1)
        ax.plot(list(range(len(predicted_vaccination))), predicted_vaccination, label='Predicted', color='blue')
        ax.plot(list(range(len(y_test))), y_scaler_svr.inverse_transform(y_test.reshape(-1,1)), label='Actual', color='red')
        time_series2.legend()
        st.pyplot(time_series2)

        # mean squared error
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, svr.predict(X_test))}")

    st.write('''
    ### Does the current vaccination rate allow herd immunity to be achieved by 30 November 2021?
    To answer this question, we use ARIMA forecasting (Auto-Regressive Integrated Moving Average) to predict the future. A problem is that ARIMA is univariate in nature, so we have to acknowledge that the estimates are quite rough.
    ''')
    st.write('ARIMA best parameters obtained using SARIMAX hyperparameter tuning.')
    st.write('Best model: SARIMAX(1, 1, 1)')

    vax_malaysia['cumul_full'] = vax_malaysia['daily_full'].cumsum()
    train = vax_malaysia[:len(vax_malaysia) - 50]
    test = vax_malaysia[len(vax_malaysia) - 50:]

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(train['cumul_full'], order=(1, 1, 1))
    result = model.fit()

    start = len(train)
    end = len(train) + len(test) - 1
    predictions = result.predict(start, end, typ='levels').rename('Predictions')

    start = 228
    end = 268
    predictions = result.predict(start, end, typ='levels').rename('Predictions')

    cumul_vaccine = list(vax_malaysia['cumul_full'])
    cumul_vaccine += predictions.tolist()

    from datetime import date, timedelta
    def add_date(dt, row):
        split = dt.split('-')
        year = int(split[0])
        month = int(split[1])
        day = int(split[2])
        date_orig = date(year, month, day)
        new_date = date_orig + timedelta(days=row)
        return str(new_date)
    dts = [add_date('2021-11-09', i) for i in range(1, 42)]
    dts = list(vax_malaysia['date']) + dts

    malaysia_population = population[population['state']=='Malaysia']['pop'].values[0]

    future = px.line(x=dts, y=cumul_vaccine / malaysia_population, title='Vaccination Rate in Malaysia (extrapolated)')
    st.plotly_chart(future)

    st.markdown('''
    Based on ARIMA auto-regressive prediction, it is possible that herd immunity will be reached before 30 November, if it continues at this rate.
    ''')

    st.markdown('''
    ### Can we predict Covid-19 mortality numbers across the nation?
    Covid-19 deaths are definitely dependent on external factors, so the question asked is whether we can train a regression model to predict Covid-19 deaths.
    To supplement static attributes, we introduce rolling averages of past data as features to our predictive models. An example of rolling averages:
    ''')

    reg_model = st.selectbox('Regression Model', ['Linear Regression', 'Ridge Regression', 'Support Vector Regression', 'Gradient Boosting Regression'])

    deaths_malaysia_rolling = deaths_malaysia.copy()
    deaths_malaysia_rolling['5_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(5).mean()
    deaths_malaysia_rolling['10_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(10).mean()
    deaths_malaysia_rolling['15_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(15).mean()
    deaths_malaysia_rolling['20_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(20).mean()
    deaths_malaysia_rolling['25_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(25).mean()
    deaths_malaysia_rolling['30_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(30).mean()
    deaths_malaysia_rolling['60_day_deaths'] = deaths_malaysia_rolling['deaths_new'].rolling(60).mean()
    deaths_malaysia_rolling.fillna(0, inplace=True)

    cases_malaysia_rolling = cases_malaysia.copy()
    cases_malaysia_rolling['5_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(5).mean()
    cases_malaysia_rolling['10_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(10).mean()
    cases_malaysia_rolling['15_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(15).mean()
    cases_malaysia_rolling['20_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(20).mean()
    cases_malaysia_rolling['25_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(25).mean()
    cases_malaysia_rolling['30_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(30).mean()
    cases_malaysia_rolling['60_day_cases'] = cases_malaysia_rolling['cases_new'].rolling(60).mean()
    cases_malaysia_rolling.fillna(0, inplace=True)

    all_data = pd.merge(cases_malaysia_rolling, deaths_malaysia_rolling, on='date')
    all_data = all_data.merge(before_pp_vax_malaysia, on='date')
    all_data = all_data.merge(before_pp_tests_malaysia, on='date')

    st.write(all_data.head())

    best_features = ['cases_recovered', 'cases_active', 'cases_pvax', 'cases_child', 'cases_adolescent', 'cases_elderly', '5_day_cases', '10_day_cases', '15_day_cases', '20_day_cases', '25_day_cases', '30_day_cases', '5_day_deaths', '10_day_deaths', 'daily_partial']

    feat_display = st.multiselect('Optimal Feature Set', best_features, default=best_features)

    X = all_data[best_features]
    y = all_data['deaths_new']

    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    if reg_model == 'Linear Regression':
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        st.write('Mean Squared Error ' + str(mean_squared_error(y_test, y_pred)))

        time_series2, _ = plt.subplots(1,1)
        ax = time_series2.add_subplot(1, 1, 1)
        ax.plot(y_pred, label='Predicted', color='blue')
        ax.plot(y_test, label='Actual', color='red')
        time_series2.legend()
        st.pyplot(time_series2)

    elif reg_model == 'Ridge Regression':
        best_params = {'alpha': 0.1}
        rr = Ridge(**best_params)

        rr.fit(X_train, y_train)

        st.write('Mean Squared Error ' + str(mean_squared_error(y_test, y_pred)))

        y_pred = rr.predict(X_test)

        time_series2, _ = plt.subplots(1,1)
        ax = time_series2.add_subplot(1, 1, 1)
        ax.plot(y_pred, label='Predicted', color='blue')
        ax.plot(y_test, label='Actual', color='red')
        time_series2.legend()
        st.pyplot(time_series2)

    elif reg_model == 'Support Vector Regression':
        best_params = {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
        svr = SVR(**best_params)
        svr.fit(X_train, y_train)

        y_pred = svr.predict(X_test)

        print(f'Mean Squared Error of SVR {mean_squared_error(y_test, y_pred)}')

        time_series2, _ = plt.subplots(1,1)
        ax = time_series2.add_subplot(1, 1, 1)
        ax.plot(y_pred, label='Predicted', color='blue')
        ax.plot(y_test, label='Actual', color='red')
        time_series2.legend()
        st.pyplot(time_series2)

    elif reg_model == 'Gradient Boosting Regressor':
        best_params = {'max_depth': 10, 'n_estimators': 100, 'subsample': 0.1}
        gbr = GradientBoostingRegressor(**best_params)
        gbr.fit(X_train, y_train)

        y_pred = gbr.predict(X_test)

        print(f'Mean Squared Error of GBR {mean_squared_error(y_test, y_pred)}')

        time_series2, _ = plt.subplots(1,1)
        ax = time_series2.add_subplot(1, 1, 1)
        ax.plot(y_pred, label='Predicted', color='blue')
        ax.plot(y_test, label='Actual', color='red')
        time_series2.legend()
        st.pyplot(time_series2)

    # dataset1 = before_pp_cases_malaysia.copy()
    # dataset2 = before_pp_deaths_malaysia.copy()
    # dataset2 = dataset2[['date','deaths_new']]
    # dataset3 = before_pp_tests_malaysia.copy()
    # dataset4 = before_pp_vax_malaysia.copy()
    # total_dataset = dataset1.merge(dataset2, how='inner', on=['date'] )
    # total_dataset.fillna(0, inplace=True)
    # total_dataset = total_dataset.merge(dataset3, how='inner', on=['date'] )
    # total_dataset.fillna(0, inplace=True)
    # total_dataset = total_dataset.merge(dataset4, how='inner', on=['date'] )
    # total_dataset.fillna(0, inplace=True)

    # features = ['cases_recovered', 'cases_active', 'cases_pvax', 'cases_child', 'cases_adolescent', 'cases_elderly', 'daily_partial', 'cumul_partial', 'cumul_full', 'cumul', 'cansino']

    # train_model_dataset = total_dataset[features]
    # train_model_dataset['deaths_new'] = total_dataset['deaths_new']

    # X = train_model_dataset.drop(['deaths_new'], axis=1)
    # X = MinMaxScaler().fit_transform(X)
    # y = train_model_dataset['deaths_new']
    # y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

    # regressor = DecisionTreeRegressor(max_depth=2, criterion="mae", splitter="best")
    # regressor.fit(X_train, y_train)
    # pred = regressor.predict(X_test)

    # mae = round(mean_absolute_error(y_test, pred),4)
    # mse = round(mean_squared_error(y_test, pred),4)

    # st.write(f"DecisionTreeRegressor MAE: {mae}")
    # st.write(f"DecisionTreeRegressor MSE: {mse}")

    st.markdown('''
    ### Can we predict mortality numbers for Melaka, Negeri Sembilan, Perlis, Selangor and W.P. Putrajaya?
    We looked at this analysis nationally, but what about at a state-level?
    ''')

    def state_mortality_prediction(state) :
        dataset1 = before_pp_cases_state.copy()
        dataset2 = before_pp_deaths_state.copy()
        dataset2 = dataset2[['date','state','deaths_new']]
        dataset3 = before_pp_tests_state.copy()
        dataset4 = before_pp_vax_state.copy()
        total_dataset = dataset1[dataset1['state'] == state].merge(dataset2[dataset2['state'] == state], how='inner', on=['date','state'] )
        total_dataset.fillna(0, inplace=True)
        total_dataset = total_dataset.merge(dataset3[dataset3['state'] == state], how='inner', on=['date','state'] )
        total_dataset.fillna(0, inplace=True)
        total_dataset = total_dataset.merge(dataset4[dataset4['state'] == state], how='inner', on=['date','state'] )
        total_dataset.fillna(0, inplace=True)
        X = total_dataset.drop(['date','deaths_new','state'], axis=1)  
        y = total_dataset['deaths_new']  

        state_features = {
            'Selangor': ['cases_new', 'cases_import', 'cases_cluster', 'cases_child','cases_adolescent', 'cases_adult', 'cases_elderly', 'cumul_partial','cumul', 'cumul_full_child'],
            'W.P. Putrajaya': ['cases_new', 'cases_import', 'cases_cluster', 'cases_pvax','cases_child', 'cases_adult', 'cases_elderly', 'daily_full_child','cumul_full_child', 'pending'],
            'Melaka': ['cases_new', 'cases_recovered', 'cases_fvax', 'cases_child','cases_adolescent', 'cases_adult', 'cases_elderly', 'cumul_full_child','astra2', 'pending'],
            'Negeri Sembilan': ['cases_new', 'cases_import', 'cases_fvax', 'cases_child', 'cases_adult','cases_elderly', 'cumul_partial', 'cumul_full', 'cumul', 'cansino'],
            'Perlis': ['cases_new', 'cases_import', 'cases_cluster', 'cases_child','cases_adolescent', 'cases_elderly', 'daily_partial','daily_full_child', 'pfizer1', 'sinovac1']
        }

        rfe_best = state_features[state]
        return rfe_best, total_dataset

    def get_result(rfe_best , total_dataset) :
        train_model_dataset = total_dataset[rfe_best]
        train_model_dataset['deaths_new'] = total_dataset['deaths_new']
        X = train_model_dataset.drop(['deaths_new'], axis=1)  
        X = MinMaxScaler().fit_transform(X)
        y = train_model_dataset['deaths_new']  
        y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2) 
        regressor = DecisionTreeRegressor(max_depth=2, criterion="mae", splitter="best")  
        regressor.fit(X_train, y_train) 
        pred = regressor.predict(X_test)
        mae = round(mean_absolute_error(y_test, pred),4)
        mse = round(mean_squared_error(y_test, pred),4)

        st.write(f"DecisionTreeRegressor MAE: {mae}")
        st.write(f"DecisionTreeRegressor MSE: {mse}")

    state_choosen = st.selectbox('Which state do you want to check?', ['Melaka', 'Negeri Sembilan', 'Perlis','Selangor','W.P. Putrajaya'])

    st.write(f"{state_choosen}")
    boruta_best , total_dataset = state_mortality_prediction(state_choosen)
    get_result(boruta_best, total_dataset)
