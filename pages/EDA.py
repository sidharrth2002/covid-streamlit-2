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

def app():
    # TITLE
    st.title("A Comprehensive Exploration of Covid-19 in Malaysia 🇲🇾")
    st.subheader("Sidharrth Nagappan, Eugene Kan, Tan Zhi Hang")
    st.image('./covid-malaysia.jpeg')
    st.markdown('*Image from The Star*')

    st.markdown("This exploration will conduct a comprehensive analysis of Covid-19 in Malaysia, while evaluating the performance of the nation in combatting the pandemic and putting this performance up against other countries in South East Asia. Covid-19 has taken the world by storm and we use open data to extract critical insights.")

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
    ## Exploratory Data Analysis
    Exploratory Data Analysis helps us obtain a statistical overview of the data and answer interesting questions to extract patterns that may exist.''')


    # =============================================================================
    # Is there a correlation between the mean income of a state and the number of cases?
    # =============================================================================
    st.markdown('''
    ### Mean Income of State vs Cases (with population)
    This is a very miscellaneous question, is Covid related to the average income of a state? Are wealthier areas less affected by Covid? Each point on the graph is a state. The larger the bubble is, the more populated the state is.
    ''')


    # merge supplementary income dataset with the cases dataset on state
    cases_income = pd.DataFrame(cases_state.groupby('state')['cases_new'].sum()).reset_index()
    cases_income = cases_income.merge(income, on='state')
    cases_income = cases_income.merge(population, on='state')
    cases_income['income'] = cases_income['income'].astype(int)

    plot = px.scatter(cases_income, x='income', y='cases_new', color='cases_new', size='pop', labels={'cases_new':'Total Cases in State', 'income': 'Average Income of State'})

    st.plotly_chart(plot)

    st.markdown('''
    There appears to be a weak correlation, hinting that Covid cases may not be a totally socio-economic one. As the average income of the state increases, the more populated it generally is, which would mean more cases.
    The population seems to be a strong **confounding variable**. Hence, we can account for the influence of population and re-plot.
    ''')
    cases_income['cases_per_10k'] = cases_income['cases_new'] / (cases_income['pop'] / 10000)
    plot = px.scatter(cases_income, x='income', y='cases_per_10k', color='cases_new', size='pop', labels={'cases_per_10k':'Total Cases in State per 10k people', 'income': 'Average Income of State'})
    st.plotly_chart(plot)

    st.markdown('''
    By accounting for the population and considering the income of the state with the cases per 10k of the population, there
    exists no trend whatsoever. This suggests that the third population variable can make or break a trend. In this case,
    Covid-19 does not appear to be a socio-economic issue.
    ''')
    # =============================================================================
    # Is there a correlation between vaccination and daily cases at a national level
    # =============================================================================
    # prepare a generic function that calculates the cumulative sum of cases and percentage vaccinated for the whole nation
    st.markdown('''
    ### Vaccination vs Daily Cases (National)
    ''')

    corr_vaccine = vax_malaysia[['date','cumul_vaccine']]
    malaysia_cases = cases_state[['date','state','cases_new']]
    filtered_my_cases = malaysia_cases.groupby('date').sum().reset_index()
    merged_data_frame = pd.merge(filtered_my_cases, corr_vaccine, on=['date'])
    corr_merged_data_frame = merged_data_frame.corr()

    fig = px.scatter(merged_data_frame, x='cumul_vaccine', y='cases_new', trendline='ols', labels={'cumul_vaccine':'Total Vaccinated', 'cases_new':'Total Cases'})
    st.plotly_chart(fig)

    st.markdown('''
    From the regression plot, we can see that the relationship is not exactly linear. It forms more of a parabolic trend towards the beginning. Perhaps it takes a while for the effects of vaccination to kickin, as after a certain volume of vaccines are administered, the number of daily cases are on a daily trend.
    ''')

    # ====================================
    # Has vaccination helped reduce daily cases in Selangor, Sabah and Sarawak?
    st.markdown('### Has the vaccination helped reduce daily cases in Selangor, Sabah and Sarawak?')

    def vaccination_dailycases(state):
        state_vax = vax_state[vax_state['state'] == state]
        state_vax['cum'] = state_vax['daily_full'].cumsum()

        state_cases = cases_state[cases_state['state'] == state]

        state_merged = state_cases.merge(state_vax, on=['date'])

        lineplot = px.line(state_merged, x='cum', y='cases_new', title=f'{state} Cumulative Vaccination vs Daily Cases', labels={'cum':'Cumulative Vaccination', 'cases_new':'Daily Cases'})
        st.write(lineplot)

    vaccination_dailycases('Selangor')

    vaccination_dailycases('Sabah')

    vaccination_dailycases('Sarawak')

    st.markdown('''
    For Selangor and Sabah, there appears to be a curvilinear relationship between cumulative vaccinations and total cases. For the first period, there is a steady increase in cases. However, upon hitting a vaccination threshold, daily cases start to drop, which may be attributed to the effects of vaccination kicking in. Vaccinations have helped reduce daily cases in these two states.

    For Sarawak however, there is an exponential increase in cases. We cannot conclude that vaccination has not been effective in this state. Instead, there may be confounding factors involved.
    ''')
    # =============================================================================
    # Is there any correlation between vaccination and daily cases for Selangor, Sabah, Sarawak, and many more?
    # =============================================================================
    # prepare a generic function that calculates the cumulative sum of cases and percentage vaccinated for each state
    st.markdown('''
    ### If daily cases increases, does that also increase the number of people getting vaccinated on a daily basis?
    #### Does the government put more effort into the vaccination campaign when cases spike?
    A naive way to answer this question is to find the correlation across the entire pandemic, but this does not take into account the time for the vaccines to start showing their effects. We try calculating the correlations only after a certain percentage of the population has been vaccinated.''')

    # prepare a generic function that calculates the cumulative sum of cases and percentage vaccinated for each state with the option
    # to remove a certain percentage of the data before doing so (something like alpha trimming)
    def cases_vax_corr(state, mode = 1,percentage = 0):
        vax_state_temp = vax_state.copy()
        vax_state_temp = vax_state_temp[vax_state_temp['state'] == state]
        population_state = population[population['state'] == state]['pop'].iloc[0]
        vax_state_temp['cmul'] = vax_state_temp['daily_full'].cumsum()
        vax_state_temp['percentage_vaccinated'] = vax_state_temp['cmul'] / population_state

        state_cases = cases_state[cases_state['state'] == state]
        if mode == 2 :
            date = vax_state_temp[vax_state_temp['percentage_vaccinated'] >= percentage]['date'].iloc[0]
            state_cases = cases_state[cases_state['date'] >= date]
        state_vax = vax_state[vax_state['state'] == state]
        state_merged = state_cases.merge(state_vax, on='date')
        corr = state_merged[['daily_full', 'cases_new']].corr()

        return corr,vax_state_temp,state_merged

    st.write('''For each state, calculate the correlation after 5%, 10% and 15% of the population has been vaccinated. ''')
    corr_selangor1,vax_percentage_selangor1,selangor_state_merged1 = cases_vax_corr('Selangor',1)
    corr_selangor2,vax_percentage_selangor2,selangor_state_merged2 = cases_vax_corr('Selangor',2,0.05)
    corr_selangor3,vax_percentage_selangor3,selangor_state_merged3 = cases_vax_corr('Selangor',2,0.10)
    corr_selangor4,vax_percentage_selangor4,selangor_state_merged4 = cases_vax_corr('Selangor',2,0.15)
    corr_sabah1,vax_percentage_sabah1,sabah_state_merged1 = cases_vax_corr('Sabah',1)
    corr_sabah2,vax_percentage_sabah2,sabah_state_merged2 = cases_vax_corr('Sabah',2,0.05)
    corr_sabah3,vax_percentage_sabah3,sabah_state_merged3 = cases_vax_corr('Sabah',2,0.15)
    corr_sabah4,vax_percentage_sabah4,sabah_state_merged4 = cases_vax_corr('Sabah',2,0.2)
    corr_sarawak1,vax_percentage_sarawak1,sarawak_state_merged1= cases_vax_corr('Sarawak',1)
    corr_sarawak2,vax_percentage_sarawak2,sarawak_state_merged2= cases_vax_corr('Sarawak',2,0.05)
    corr_sarawak3,vax_percentage_sarawak3,sarawak_state_merged3= cases_vax_corr('Sarawak',2,0.15)
    corr_sarawak4,vax_percentage_sarawak4,sarawak_state_merged4= cases_vax_corr('Sarawak',2,0.2)

    table = {'Full Period':[corr_selangor1['daily_full']['cases_new'], corr_sabah1['daily_full']['cases_new'], corr_sarawak1['daily_full']['cases_new']],
            'When Vacinated Rate Over 0.05':[corr_selangor2['daily_full']['cases_new'], corr_sabah2['daily_full']['cases_new'], corr_sarawak2['daily_full']['cases_new']],
            'When Vacinated Rate Over 0.10':[corr_selangor3['daily_full']['cases_new'], corr_sabah3['daily_full']['cases_new'], corr_sarawak3['daily_full']['cases_new']],
            'When Vacinated Rate Over 0.15':[corr_selangor4['daily_full']['cases_new'], corr_sabah4['daily_full']['cases_new'], corr_sarawak4['daily_full']['cases_new']]}

    # Creates pandas DataFrame.
    table = pd.DataFrame(table, index =['Selangor','Sabah','Sarawak'])

    # Creates pandas DataFrame.
    st.dataframe(table)
    st.write('''
    Based on the table, we can see that the correlation between vaccination and daily cases changes drastically in different periods of the vaccination campaign, showing no noticeable pattern. We can visualise the correlation plots at different periods of the vaccination campaign.
    ''')

    fig2 = make_subplots(rows=3, cols=4, subplot_titles=('Full Period Selangor', 'Vax Rate > 0.05', 'Vax Rate > 0.10', 'Vax Rate > 0.15', 'Full Period Sabah', 'Vax Rate > 0.05', 'Vax Rate > 0.10', 'Vax Rate > 0.15', 'Full Period Sarawak', 'Vax Rate > 0.05', 'Vax Rate > 0.10', 'Vax Rate > 0.15'))
    fig2.add_trace(go.Scatter(x=selangor_state_merged1['cases_new'], y=selangor_state_merged1['daily_full'], mode='markers', line=go.scatter.Line(), name='Selangor All Days'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=selangor_state_merged2['daily'], y=selangor_state_merged2['daily_full'], mode='markers', line=go.scatter.Line(), name='Selangor after reaching 5% vaccination'), row=1, col=2)
    fig2.add_trace(go.Scatter(x=selangor_state_merged3['daily'], y=selangor_state_merged3['daily_full'], mode='markers', line=go.scatter.Line(), name='Selangor after reaching 10% vaccination'), row=1, col=3)
    fig2.add_trace(go.Scatter(x=selangor_state_merged4['cases_new'], y=selangor_state_merged4['daily_full'], mode='markers', line=go.scatter.Line(), name='Selangor after reaching 15% vaccination'), row=1, col=4)
    fig2.add_trace(go.Scatter(x=sabah_state_merged1['cases_new'], y=sabah_state_merged1['daily_full'], mode='markers', line=go.scatter.Line(), name='Sabah All Days'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=sabah_state_merged2['cases_new'], y=sabah_state_merged2['daily_full'], mode='markers', line=go.scatter.Line()), row=2, col=2)
    fig2.add_trace(go.Scatter(x=sabah_state_merged3['cases_new'], y=sabah_state_merged3['daily_full'], mode='markers', line=go.scatter.Line()), row=2, col=3)
    fig2.add_trace(go.Scatter(x=sabah_state_merged4['cases_new'], y=sabah_state_merged4['daily_full'], mode='markers', line=go.scatter.Line()), row=2, col=4)
    fig2.add_trace(go.Scatter(x=sarawak_state_merged1['cases_new'], y=sarawak_state_merged1['daily_full'], mode='markers', line=go.scatter.Line()), row=3, col=1)
    fig2.add_trace(go.Scatter(x=sarawak_state_merged2['cases_new'], y=sarawak_state_merged2['daily_full'], mode='markers', line=go.scatter.Line()), row=3, col=2)
    fig2.add_trace(go.Scatter(x=sarawak_state_merged3['cases_new'], y=sarawak_state_merged3['daily_full'], mode='markers', line=go.scatter.Line()), row=3, col=3)
    fig2.add_trace(go.Scatter(x=sarawak_state_merged4['cases_new'], y=sarawak_state_merged4['daily_full'], mode='markers', line=go.scatter.Line()), row=3, col=4)
    fig2.update_layout(height=800, width=800, title_text="Daily Cases vs. Daily Vaccination Numbers", showlegend=False)
    st.plotly_chart(fig2)

    st.markdown('''
    If we look at the vaccination campaign of the state as a whole, there is a correlation between daily cases and vaccination numbers. However, if you zoom into different periods of the campaign (after hitting 5%, 10% and 15% vaccination rates), there is no pattern between daily cases and daily vaccinations. Regardless of whether cases up or down, the government still continues administering vaccines.
    ''')

    # =============================================================================
    # Which states have been most affected by Covid clusters?
    # =============================================================================
    import re

    st.markdown("""
    ### Which states have been most affected by Covid clusters?'
    #### May it be Kluster Mahkamah, Court Cluster, Kluster Mamak, etc.
    """
    )

    def get_iqr_values(df_in, col_name):
        median = df_in[col_name].median()
        q1 = df_in[col_name].quantile(0.25) # 25th percentile / 1st quartile
        q3 = df_in[col_name].quantile(0.75) # 7th percentile / 3rd quartile
        iqr = q3-q1 #Interquartile range
        minimum  = q1-1.5*iqr # The minimum value or the |- marker in the box plot
        maximum = q3+1.5*iqr # The maximum value or the -| marker in the box plot
        return median, q1, q3, iqr, minimum, maximum

    def remove_outliers(df_in, col_name):
        _, _, _, _, minimum, maximum = get_iqr_values(df_in, col_name)
        df_out = df_in.loc[(df_in[col_name] > minimum) & (df_in[col_name] < maximum)]
        return df_out

    clusters['single_state'] = clusters['state'].apply(lambda x: [state.strip().title() for state in re.split(', | & ', x)])
    clusters_singlestate = clusters.explode('single_state')
    clusters_singlestate = clusters_singlestate[clusters_singlestate['single_state'].isin(['Wp Kuala Lumpur', 'Wp Putrajaya', 'Selangor', 'Negeri Sembilan', 'Pahang', 'Johor', 'Sarawak', 'Kedah', 'Perak', 'Kelantan'])]
    st.plotly_chart(px.box(clusters_singlestate, x='single_state', y='cases_total'))
    st.markdown('''
    We can see from the boxplot that the majority of Covid Clusters are moderately sized, often under 100 cases per cluster. However, there does exist unusually large clusters that appear as outliers.

    In this case, to get a more meaningful idea of "ordinary" clusters alone, we remove the outliers and show a new boxplot. Surprisingly, Negeri Sembilan and Perak are most affected by clusters, as opposed to more populated states like Selangor or Kuala Lumpur.
    ''')
    clusters_singlestate = remove_outliers(clusters_singlestate, 'cases_total')
    st.plotly_chart(px.box(clusters_singlestate, x='single_state', y='cases_total', points=False))

    st.markdown('''
    ### What type of Covid-19 clusters are most prevalent?
    Across the country, clusters form in schools, workplaces, places of worship, etc. But the question is which ones are most drastic?
    ''')

    st.plotly_chart(px.box(clusters_singlestate, x='category', y='cases_total'))
    st.markdown('''
    The detention centers generally have the largest Covid clusters, with a strong right skew. The rest of the cluster categories have mostly small clusters, but as we observed in the earlier question, there are quite a few unusually large clusters.

    An interesting example is workplace clusters. For instance, most companies/organisations in Malaysia are small, so there may be a **high frequency** of clusters, but only some are **large enough** to appear as an outlier. Case in point, Top Glove.
    ''')

    # =============================================================================
    # How well is Malaysia's vaccination campaign doing compared to other countries in South-East Asia?
    # =============================================================================

    st.markdown('''
    ### How well is Malaysia's vaccination campaign doing compared to other countries in South-East Asia?
    ''')
    def getCountry(country) :
        filter = global_datasets['location'] == country
        df = global_datasets[filter]
        df = df[['date','population','people_fully_vaccinated']]
        df['cumul'] = df['people_fully_vaccinated'].cumsum()
        df['percentage_vaccinated'] = df['cumul'] / df['population']
        filter2 = df['date'] > '2021-04'
        df = df[filter2]
        return df
    Brunei = getCountry('Brunei')
    Myanmar = getCountry('Myanmar')
    Cambodia = getCountry('Cambodia')
    Indonesia = getCountry('Indonesia')
    Laos = getCountry('Laos')
    Malaysia = getCountry('Malaysia')
    Philippines = getCountry('Philippines')
    Singapore = getCountry('Singapore')
    Thailand = getCountry('Thailand')
    Vietnam = getCountry('Vietnam')

    # stupid plotly cannot do multiple lines in one plot, so use seaborn

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Line(x=Malaysia['date'], y=Malaysia['percentage_vaccinated'], name='Malaysia'), secondary_y=False)
    fig.add_trace(go.Line(x=Brunei['date'], y=Brunei['percentage_vaccinated'], name='Brunei'), secondary_y=False)
    fig.add_trace(go.Line(x=Myanmar['date'], y=Myanmar['percentage_vaccinated'], name='Myanmar'), secondary_y=False)
    fig.add_trace(go.Line(x=Cambodia['date'], y=Cambodia['percentage_vaccinated'], name='Cambodia'), secondary_y=False)
    fig.add_trace(go.Line(x=Indonesia['date'], y=Indonesia['percentage_vaccinated'], name='Indonesia'), secondary_y=False)
    fig.add_trace(go.Line(x=Laos['date'], y=Laos['percentage_vaccinated'], name='Laos'), secondary_y=False)
    fig.add_trace(go.Line(x=Philippines['date'], y=Philippines['percentage_vaccinated'], name='Philippines'), secondary_y=False)
    fig.add_trace(go.Line(x=Singapore['date'], y=Singapore['percentage_vaccinated'], name='Singapore'), secondary_y=False)
    fig.add_trace(go.Line(x=Thailand['date'], y=Thailand['percentage_vaccinated'], name='Thailand'), secondary_y=False)
    fig.add_trace(go.Line(x=Vietnam['date'], y=Vietnam['percentage_vaccinated'], name='Vietnam'), secondary_y=False)
    fig.update_layout(title_text='Vaccination Campaigns Across ASEAN', xaxis_title='Date', yaxis_title='Percentage of Population Vaccinated', yaxis_title_text='Percentage of Population Vaccinated')
    st.plotly_chart(fig)

    st.markdown('''
    The line graph above shows that the vaccination rate for each country in South-East Asia. Based on the result, we can see that Cambodia has the highest vaccination rate compared to other countries. For Malaysia, we ranked top 3 in the graph and the vaccinated rate is near 45%. Hence, we can conclude that Malaysia's vaccination campaign is doing better than the majority of South-East Asia's countries.

    **Note: Percentages are on the low end maybe because this dataset is a few weeks old. When countries are vaccinating thousands a day, the percentage will be higher.**

    Even still, Malaysia is doing statistically better than other ASEAN nations.
    '''
    )

    st.markdown('''
    ### Is there a correlation between individual casual contacts (contact tracing) and daily cases? If the link is strong, how contagious is Covid-19?
    ''')
    temp_cases_malaysia = cases_malaysia.copy()
    temp_trace_malaysia = trace_malaysia.copy()
    merged = temp_cases_malaysia.merge(temp_trace_malaysia, on='date')
    corr = merged[['casual_contacts', 'cases_new']].corr()
    st.plotly_chart(px.scatter(merged, x='casual_contacts', y='cases_new', trendline='ols'))
    st.markdown('''
        Based on both the regression plot, we can see that daily new cases are highly correlated with the number of casual contacts for the day. As in, if more people go out and come in close contact with infected people, the number of cases increases. The statistics fall in line with the science behind Covid-19 🦠.
    ''')
    st.image(Image.open('./social-distancing.gif'), width=300)

    st.markdown('''
    ### How have vaccination numbers changed over time across states?
    ''')
    corr_vaccine = vax_malaysia[['date', 'cumul_vaccine']].copy()
    malaysia_cases = cases_state[['date','state','cases_new']]
    filtered_my_cases = malaysia_cases.groupby('date').sum().reset_index()

    states = ['Melaka','Negeri Sembilan','Perlis','Selangor','W.P. Putrajaya']

    vax_state = vax_state[['date','state','daily_full']]
    merged_vax_state = pd.merge(vax_state, malaysia_cases, on=['date','state'])
    scaler = preprocessing.MinMaxScaler()
    names = ['cases_new','daily_full']
    d = scaler.fit_transform(merged_vax_state[['cases_new','daily_full']])
    scaled_df = pd.DataFrame(d, columns=names)
    scaled_df['date'] = pd.to_datetime(merged_vax_state['date'])
    scaled_df['state'] = merged_vax_state['state']

    fig, axes = plt.subplots(3, 2, figsize=(20,20))

    for index, i in enumerate(states):
        ax = fig.add_subplot(3, 2, index + 1)
        ax.plot(scaled_df[scaled_df['state']==i]['date'],scaled_df[scaled_df['state']==i]['daily_full'],label = "Vaccination Rate (%)")
        ax.plot(scaled_df[scaled_df['state']==i]['date'],scaled_df[scaled_df['state']==i]['cases_new'],label = "Daily New Cases")
        plt.xticks(rotation=90)
        plt.title(i + ' Daily Vaccination and Daily Cases')
        plt.xlabel('Date')
        plt.ylabel('Normalized quantity')
        leg = plt.legend(loc='upper left')

    st.pyplot(fig)

    st.markdown('''
    We specifically look at the top 5 states that require more attention( Melaka, Negeri Sembilan, Perlis, Selangor and W.P. Putrajaya) that we found out that have low vaccination rate in the earlier finding. As we can see from the line graph that we plot out, it is clear that when the government put more effort into getting the people vaccinated in that certain state, the daily cases would start to decline, except for Perlis. Perlis's daily cases spikes up a little in October 2021 but not drastically. We can safely assume that vaccination actually in a way help controlling the cluster cases in Perlis without going higher. Regardless, we can conclude that vaccination might be one of the contributing factor to reduce daily covid cases in Malaysia. However, more research is needed in order to conclude this finding.
    ''')

    st.markdown('''
    ### How has the vaccination rate changed across the nation?
    This is a simple question, but the progression of vaccination is an interesting phenonemon to explore.
    ''')
    vax_pop_percentage = vax_malaysia['cumul_vaccine']
    vax_pop_percentage = pd.DataFrame(vax_pop_percentage)
    vax_pop_percentage['percentage'] = vax_pop_percentage.apply(lambda x: (vax_malaysia['cumul_vaccine']/population[population['state']=='Malaysia']['pop'].item())*100,axis=0)
    vax_pop_percentage['date'] = pd.to_datetime(vax_malaysia['date'])

    st.plotly_chart(px.line(vax_pop_percentage, x='date', y='percentage', labels={'date':'Date', 'percentage':'Vaccination Rate (%)'}))
    st.write(''''
    If we consider the population of Malaysia as a whole, we have just crossed the 60% vaccination threshold. There does seem to be a slow start, but the speed of the campaign has picked up.
    ''')

    st.markdown('''
    ### Vaccine 💉 Distribution Modelling
    ''')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['pfizer'], name='Pfizer', mode='lines'), secondary_y=False)
    fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['sinovac'], name='Sinovac', mode='lines'), secondary_y=False)
    fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['astra'], name='Astrazeneca', mode='lines'), secondary_y=False)
    fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['cansino'], name='Cansino', mode='lines'), secondary_y=False)
    fig.update_layout(title='Vaccine brand usage over time')
    st.plotly_chart(fig)

    st.write('''
    At the start of the campaign, MoH was primarily using Pfizer, but by June, Sinovac would rapidly overtake Pfizer to become the most used vaccine. Adoption of Sinovac however drops, but people still do come back for their second dose.
    ''')

    st.image(Image.open('./sinovac-phaseout.png'), width=400)

    vaccine_totals = pd.DataFrame(vax_malaysia_all_attributes[['pfizer', 'astra', 'sinovac']].sum().reset_index())
    vaccine_totals.columns = ['vaccine', 'total']

    # plotly pie chart
    vaccines_pie = px.pie(vaccine_totals, values='total', names='vaccine', title='Vaccine distribution')
    st.plotly_chart(vaccines_pie)

    st.write('''
    Pfizer is the most used vaccine in Malaysia, followed by Sinovac and then Astrazeneca. If you observe the usage of Astrazeneca, it flails in comparison to the other two because it was opened up for voluntary registrations. Furthermore, unlike Pfizer and Sinovac, Astrazeneca usage does not show an upward trend and only has a few spikes, which may be linked to the times the government opens up registrations.
    ''')