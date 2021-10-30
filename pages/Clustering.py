from seaborn.matrix import heatmap
import streamlit as st
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def app():
    # ====================================================================
    # CLUSTERING
    # ====================================================================

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
    ## Clustering
    In this section, we will employ different clustering algorithms to see whether states suffering from Covid-19 form any visible patterns. 🦠
    ''')

    st.markdown('''
    ### How did the clusters change over time with respect to cases and deaths? Did some states reorganise into new clusters?
    ''')
    states = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']
    dates = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01']
    ranges_1 = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

    date_range_2d = st.select_slider('Slide to see the clusters moving through time using **temporal clustering**.', options=ranges_1, key='2d')

    st.markdown('#### K-Means')

    cases_state_date = cases_state[(cases_state['date'] >= date_range_2d[0]) & (cases_state['date'] < date_range_2d[1])]
    deaths_state_date = deaths_state[(deaths_state['date'] >= date_range_2d[0]) & (deaths_state['date'] < date_range_2d[1])]

    cases = []
    vaccinations = []
    deaths = []

    for state in states:
        cases.append(cases_state_date[cases_state_date['state'] == state]['cases_new'].sum())
        deaths.append(deaths_state_date[deaths_state_date['state'] == state]['deaths_new'].sum())

    cases_deaths_vaccinations = pd.DataFrame({"state": states, "cases": cases, "deaths": deaths})
    for col in cases_deaths_vaccinations:
        if cases_deaths_vaccinations[col].dtype != 'object':
            cases_deaths_vaccinations[col] = StandardScaler().fit_transform(cases_deaths_vaccinations[[col]])

    cases_deaths_vaccinations.drop(columns=['state'], inplace=True)
    X_std = cases_deaths_vaccinations

    km = KMeans(n_clusters=3, max_iter=100)
    y_clusters = km.fit_predict(cases_deaths_vaccinations)
    cases_deaths_vaccinations['cluster'] = y_clusters
    centroids = km.cluster_centers_

    twodimensionalclusters = px.scatter(cases_deaths_vaccinations, x="cases", y="deaths", color="cluster")
    st.plotly_chart(twodimensionalclusters)

    st.markdown('#### DBSCAN')

    cases_state_date = cases_state[(cases_state['date'] >= date_range_2d[0]) & (cases_state['date'] < date_range_2d[1])]
    deaths_state_date = deaths_state[(deaths_state['date'] >= date_range_2d[0]) & (deaths_state['date'] < date_range_2d[1])]

    cases = []
    vaccinations = []
    deaths = []

    for state in states:
        cases.append(cases_state_date[cases_state_date['state'] == state]['cases_new'].sum())
        deaths.append(deaths_state_date[deaths_state_date['state'] == state]['deaths_new'].sum())

    cases_deaths_vaccinations = pd.DataFrame({"state": states, "cases": cases, "deaths": deaths})

    for col in cases_deaths_vaccinations:
        if cases_deaths_vaccinations[col].dtype != 'object':
            cases_deaths_vaccinations[col] = StandardScaler().fit_transform(cases_deaths_vaccinations[[col]])

    cases_deaths_vaccinations.drop(columns=['state'], inplace=True)
    X_std = cases_deaths_vaccinations

    dbscan = DBSCAN(eps=0.5, min_samples=1).fit(cases_deaths_vaccinations)
    y_clusters = dbscan.fit_predict(cases_deaths_vaccinations)

    cases_deaths_vaccinations['cluster'] = y_clusters

    dbscanclusters = px.scatter(cases_deaths_vaccinations, x="cases", y="deaths", color="cluster")
    st.plotly_chart(dbscanclusters)

    st.markdown('''
    #### Analysis up until September 2021-October 2021
    We can observe that in the beginning, the majority of the clusters were positioned towards the bottom left and they maintain a similar pattern until about August 2020. In August, the cases were still high but there were fewer deaths, which may signify that the situation was improving, besides the one state that is in the upper corner of the plot that stands out from the rest. Around December, the bottom-right cluster begins to break up and states start moving diagonally upwards in the graph, meaning higher number of deaths and more cases. By September 2021, the states fall in a sort of straight diagonal line, with the performance of states spread across the spectrum from mild to serious.

    If a state has high cases and low deaths, that shows the effectiveness of the vaccination campaign. This is because vaccines have been known to reduce the seriousness of cases. Evidently, as the vaccination campaign begins around the start of April, cluster points start to move mainly horizontally (smaller increase in deaths).

    **We also set out to find that one state that created a cluster of it's own throughout and it was Selangor, to no one's surprise.**
    ''')


    st.markdown('''
    ### How do clusters change over time with respect to cases, deaths and vaccinations (**third factor**)? Did some states reorganise into different clusters?

    This time, let's bring in a third variable- vaccinations.
    ''')

    nrows = 5
    ncols = 4
    dates = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01']
    ranges_2 = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

    date_range = st.select_slider('Slide to see the clusters moving through time using temporal K-Means clustering.', options=ranges_2, key='3d')

    cases_state_date = cases_state[(cases_state['date'] >= date_range[0]) & (cases_state['date'] < date_range[1])]
    deaths_state_date = deaths_state[(deaths_state['date'] >= date_range[0]) & (deaths_state['date'] < date_range[1])]
    vax_state_date = vax_state[(vax_state['date'] >= date_range[0]) & (vax_state['date'] < date_range[1])]

    cases = []
    vaccinations = []
    deaths = []

    states = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']

    for state in states:
        cases.append(cases_state_date[cases_state_date['state'] == state]['cases_new'].sum())
        deaths.append(deaths_state_date[deaths_state_date['state'] == state]['deaths_new'].sum())
        vaccinations.append(vax_state_date[vax_state_date['state'] == state]['daily_full'].sum() / population.loc[population['state'] == state, 'pop'].values[0])

    cases_deaths_vaccinations = pd.DataFrame({"state": states, "cases": cases, "deaths": deaths, "vaccinations": vaccinations})
    for col in cases_deaths_vaccinations:
        if cases_deaths_vaccinations[col].dtype != 'object':
            cases_deaths_vaccinations[col] = StandardScaler().fit_transform(cases_deaths_vaccinations[[col]])

    cases_deaths_vaccinations.drop(columns=['state'], inplace=True)
    X_std = cases_deaths_vaccinations

    km = KMeans(n_clusters=3, max_iter=100)
    y_clusters = km.fit_predict(cases_deaths_vaccinations)
    cases_deaths_vaccinations['cluster'] = y_clusters
    centroids = km.cluster_centers_

    threedimensionalclusters = px.scatter_3d(cases_deaths_vaccinations, x="cases", y="deaths", z="vaccinations", color="cluster")
    st.plotly_chart(threedimensionalclusters)

    st.markdown('''
    For the last month, there are 3 observable clusters:
    1. Low Cases, Low Deaths and Low Vaccinations
    2. High Cases, High Deaths and Low Vaccinations
    3. Low Cases, Moderately High Deaths and High Vaccinations

    Throughout 2020, vaccinations are 0 and hence, the clusters slowly start to expand in terms of cases and deaths. By 2021, the states have spread reasonably wide throughout the 3 dimensions and cases and deaths are at an all time high.

    Around this time, vaccination begins and clusters start moving higher in the "vaccination" dimension. As most states move vertically upward in the dimension, there are still a cluster of states with low vaccination rates.

    By October 2021, most states are on the upper end of the vaccination spectrum, the lower end of the deaths spectrum, but cases are still spread wide across. This may not be too consequential, because the fact that an increase in cases did not lead to an increase deaths shows the effect of the 💉 campaign.
    ''')

    st.markdown('''
    ### Which states require attention in terms of their vaccination campaign and deaths (relatively)?
    ''')

    dates = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01']
    ranges_3 = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]
    date_range = st.select_slider('Slide to see the clusters moving through time using temporal K-Means clustering.', options=ranges_3, key='vaccination_and_deaths')

    cases_state_date = cases_state[(cases_state['date'] >= date_range[0]) & (cases_state['date'] < date_range[1])]
    deaths_state_date = deaths_state[(deaths_state['date'] >= date_range[0]) & (deaths_state['date'] < date_range[1])]
    vax_state_date = vax_state[(vax_state['date'] >= date_range[0]) & (vax_state['date'] < date_range[1])]

    cases = []
    vaccination_rates = []
    deaths = []

    for state in states:
        cases.append(cases_state_date[cases_state_date['state'] == state]['cases_new'].sum())
        deaths.append(deaths_state_date[deaths_state_date['state'] == state]['deaths_new'].sum() / cases_state_date[cases_state_date['state'] == state]['cases_new'].sum())
        # divide number of vaccinations by state population
        vaccination_rates.append(vax_state_date[vax_state_date['state'] == state]['daily_full'].sum() / population.loc[population['state'] == state, 'pop'].values[0])

    cases_deaths_vaccinations = pd.DataFrame({"state": states, "deaths": deaths, "vaccination_rate": vaccination_rates})

    for col in cases_deaths_vaccinations:
        if cases_deaths_vaccinations[col].dtype != 'object':
            cases_deaths_vaccinations[col] = StandardScaler().fit_transform(cases_deaths_vaccinations[[col]])

    cases_deaths_vaccinations.drop(columns=['state'], inplace=True)
    X_std = cases_deaths_vaccinations

    km = KMeans(n_clusters=4, max_iter=100)
    y_clusters = km.fit_predict(cases_deaths_vaccinations)
    centroids = km.cluster_centers_

    # put back state column
    cases_deaths_vaccinations['state'] = states
    cases_deaths_vaccinations['cluster'] = y_clusters
    vaccination_deaths = px.scatter(cases_deaths_vaccinations, x="vaccination_rate", y="deaths", color="cluster", labels={"vaccination_rate": "Vaccination Rate", "deaths": "Death Rate"})

    st.plotly_chart(vaccination_deaths)

    cases_deaths_vaccinations[cases_deaths_vaccinations['cluster'] == 1]
    st.write(cases_deaths_vaccinations)

    st.markdown('''
    We can see that throughout 2020, there are 0 vaccinations in all state since the vaccination campaign was yet to start. By February 2021, two states have begun their vaccination campaigns. It speeds up more rapidly by March and April, the vertical clusters start to spread out on the x-axis indicating higher vaccination numbers. In June 2021, there was a remarkable shoot where the y-axis scale completely changed. As of September 2021, the states that may require attention are those with low vaccination rates and high deaths, namely cluster 2, which contains the following states:

    * Melaka
    * Negeri Sembilan
    * Perlis
    * Selangor
    * W.P. Putrajaya

    On the other hand, these are the states with relatively high vaccination rates w.r.t deaths:
    * Sarawak
    * W.P. Kuala Lumpur
    * W.P. Labuan
    ''')
