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
from sklearn.cluster import KMeans
from urllib.error import URLError

state_locations = []

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
# TITLE
st.title("A Comprehensive Exploration of Covid-19 in Malaysia 🇲🇾")
st.subheader("Sidharrth Nagappan, Eugene Kan, Tan Zhi Hang")
st.image('./covid-malaysia.jpeg')

# DISPLAY MAP WITH CASES AND DEATHS
st.markdown('''
## Spatially Mapped Covid-19 data
''')
with open("./states_lat_lon.txt", "r") as f:
    text = f.read()
    for line in text.split("\n"):
        data = line.split()
        if len(data) == 3:
            data_dict = {}
            data_dict["state"] = data[0]
            data_dict["lat"] = data[1]
            data_dict["lon"] = data[2]
            state_locations.append(data_dict)
state_locations = pd.DataFrame(state_locations)

cases_state_locations = c
ases_state.copy()
cases_state_locations = cases_state_locations[cases_state_locations['date'] == '2021-10-09']
cases_state_locations = cases_state_locations.groupby('state').sum()
cases_state_locations = cases_state_locations.reset_index()
cases_state_locations = cases_state_locations.merge(state_locations, on='state')

try:
    ALL_LAYERS = {
        "Covid Cases": pdk.Layer(
            "HexagonLayer",
            data=cases_state_locations,
            elevation_scale=4,
            pickable=True,
            extruded=True,
            get_position="[lat, lon]",
            get_text="state",
            get_radius="[cases_new]",
            # elevation_range=[0, 1000],
        )
    }
    st.sidebar.markdown('### Map Layers')
    selected_layers = [
        layer for layer_name, layer in ALL_LAYERS.items()
        if st.sidebar.checkbox(layer_name, True)]
    print(selected_layers)
    if selected_layers:
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={"latitude": 6.155672,
                                "longitude": 100.569649, "zoom": 5, "pitch": 50},
            layers=selected_layers,
        ))
    else:
        st.error("Please choose at least one layer above.")
except URLError as e:
    st.error("""
        **This demo requires internet access.**

        Connection error: %s
    """ % e.reason)



# =============================================================================
# Is there a correlation between the mean income of a state and the number of cases?
# =============================================================================
st.markdown('''
### Mean Income vs States (with population)
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
There appears to be a weak correlation, hinting that Covid cases may not be a totally socio-economic one. As the average income of the state increases, the more populated it generally is, which would mean more cases. The population is a strong **confounding variable**.
''')
# =============================================================================
# Is there any correlation between vaccination and daily cases for Selangor, Sabah, Sarawak, and many more?
# =============================================================================
# prepare a generic function that calculates the cumulative sum of cases and percentage vaccinated for each state
st.markdown(''''
### Vaccination vs Daily Cases
We first did correlation with all the values for each state, but soon realised that the initial 10% would stall the trend. Therefore, we calculate a correlation by first removing the first 10% of the vaccination campaign. Now, we can really start to see the effect of vaccination.
''')
def cases_vax_corr(state, mode = 1):
    vax_state_temp = vax_state.copy()
    vax_state_temp = vax_state_temp[vax_state_temp['state'] == state]
    population_state = population[population['state'] == state]['pop'].iloc[0]
    vax_state_temp['cmul'] = vax_state_temp['daily_full'].cumsum()
    vax_state_temp['percentage_vaccinated'] = vax_state_temp['cmul'] / population_state

    state_cases = cases_state[cases_state['state'] == state]
    if mode == 2 :
        date = vax_state_temp[vax_state_temp['percentage_vaccinated'] >= 0.1]['date'].iloc[0]
        state_cases = cases_state[cases_state['date'] >= date]
    state_vax = vax_state[vax_state['state'] == state]
    state_merged = state_cases.merge(state_vax, on='date')
    corr = state_merged[['daily', 'cases_new']].corr()
    return corr,vax_state_temp

corr_selangor1,vax_percentage_selangor1 = cases_vax_corr('Selangor',1)
corr_selangor2,vax_percentage_selangor2 = cases_vax_corr('Selangor',2)
corr_sabah1,vax_percentage_sabah1 = cases_vax_corr('Sabah',1)
corr_sabah2,vax_percentage_sabah2 = cases_vax_corr('Sabah',2)
corr_sarawak1,vax_percentage_sarawak1= cases_vax_corr('Sarawak',1)
corr_sarawak2,vax_percentage_sarawak2= cases_vax_corr('Sarawak',2)

table = {'Full Period':[corr_selangor1['daily']['cases_new'], corr_sabah1['daily']['cases_new'], corr_sarawak1['daily']['cases_new']],
        'When Vacinated Rate Over 0.1':[corr_selangor2['daily']['cases_new'], corr_sabah2['daily']['cases_new'], corr_sarawak2['daily']['cases_new']]}

# Creates pandas DataFrame.
table = pd.DataFrame(table, index =['Selangor','Sabah','Sarawak'])
st.dataframe(table)
st.write('''
Based on the table, we can see that for the full period columns, Selangor and Sabah have a very high positive correlation between daily fully vaccinated and daily new cases. However, Sarawak correlation value is near zero which is very different from Selangor and Sabah. Hence, we did another correlation comparison is to take the period of the state when their vaccinated rate is over 10% of their population and the result show that all 3 state their correlation values are also dropped and near to zero. Moreover, based on the line graph, we can see that Sarawak vaccinated rate increased faster than another two states so, in the full period columns result, Sarawak correlation value is already very low. In conclusion, we can conclude that when the vaccinated rate reaches a certain point, the correlation value between daily fully vaccinated and daily new cases of the specific states will drop to a certain point and near to zero.
''')

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Line(x=vax_percentage_selangor1['date'], y=vax_percentage_selangor1['percentage_vaccinated'], name='Selangor'), secondary_y=False)
fig.add_trace(go.Line(x=vax_percentage_sabah1['date'], y=vax_percentage_sabah1['percentage_vaccinated'], name='Sabah'), secondary_y=False)
fig.add_trace(go.Line(x=vax_percentage_sarawak1['date'], y=vax_percentage_sarawak1['percentage_vaccinated'], name='Sarawak'), secondary_y=False)
st.plotly_chart(fig)

# =============================================================================
# Which states have been most affected by Covid clusters?
# =============================================================================
import re

st.markdown('### Which states have been most affected by Covid clusters?')

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
We can see from the boxplot that the majority of Covid Clusters are small, while there exists some large scale clusters in each state. In this case, to get a more meaningful idea of the drasticity of the majority of outliers, we will first remove the outliers. If we compare the boxplot with and without outliers, we can see that the majority of clusters are quite small, often under 100 cases. Just because a state is big does not mean that there are many cases that are due to Covid clusters. In this case, Negeri Sembilan seems to be the state most affected by individual clusters. Some Covid clusters also span across multiple states.
''')
clusters_singlestate = remove_outliers(clusters_singlestate, 'cases_total')
st.plotly_chart(px.box(clusters_singlestate, x='single_state', y='cases_total', points=False))

st.markdown(''''
### What type of Covid-19 clusters are most prevalent?
Across the country, clusters form in schools, workplaces, places of worship, etc. But the question is which ones are most drastic?
''')

st.plotly_chart(px.box(clusters_singlestate, x='category', y='cases_total'))
st.markdown('''
The detention centers generally have the largest Covid clusters, with a strong right skew. The rest of the cluster categories have mostly small clusters, but quite a few unusually large clusters. This is especially for the workplace clusters. For instance, most companies/organisations in Malaysia are small, so there may be a lot of clusters, but only some are large enought to appear as an outlier.
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
The line graph above shows that the vaccination rate for each country in South-East Asia. Based on the result, we can see that Cambodia has the highest vaccinated rate compared to other countries and Myanmar have the lowest vaccinated rate which is near zero. For Malaysia, we ranked top 3 in the graph and the vaccinated rate is near 45%. Hence, we can conclude that Malaysia's vaccination campaign is doing better than the majority of South-East Asia's countries.
'''
)

st.markdown('''
### Is there a link between casual contacts and the number of daily cases? Can we classify how contagious Covid is?
''')
temp_cases_malaysia = cases_malaysia.copy()
temp_trace_malaysia = trace_malaysia.copy()
merged = temp_cases_malaysia.merge(temp_trace_malaysia, on='date')
corr = merged[['casual_contacts', 'cases_new']].corr()
st.plotly_chart(px.scatter(merged, x='casual_contacts', y='cases_new', trendline='ols'))
st.markdown('''
    We can see that case_new is highly correlated with casual_contacts so I mean that there is a link between the cases_new and casual_contact. If more people go out and get infected, the number of cases will increase.
''')

st.markdown('''
### How has the vaccination rate changed over time across states?
We first obtain a new dataframe which only contains the date and the cummulative vaccination head count then we examine their correlation and visualize it using heatmap.
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

#scaled_df[scaled_df['state']==i]['date'],
for index, i in enumerate(states):
    ax = fig.add_subplot(3, 2, index + 1)
    ax.plot(scaled_df[scaled_df['state']==i]['date'],scaled_df[scaled_df['state']==i]['daily_full'],label = "Vaccination Rate (%)")
    ax.plot(scaled_df[scaled_df['state']==i]['date'],scaled_df[scaled_df['state']==i]['cases_new'],label = "Daily New Cases")
    plt.xticks(rotation=90)
    plt.title(i + ' Vaccination Rate and Daily Cases')
    plt.xlabel('Date')
    plt.ylabel('Normalized quantity')
    leg = plt.legend(loc='upper left')

st.pyplot(fig)
st.markdown('''
We first normalize both the vaccination rate(%) and the daily new cases of covid cases to see if there's any effect of vaccination. We specifically look at the top 5 states that require more attention( Melaka, Negeri Sembilan, Perlis, Selangor and W.P. Putrajaya) that we found out that have low vaccination rate in the earlier finding. As we can see from the line graph that we plot out, it is clear that when the government put more effort into getting the people vaccinated in that certain state, the daily cases would start to decline, except for Perlis. Perlis's daily cases spikes up a little in October 2021 but not drastically. We can safely assume that vaccination actually in a way help controlling the cluster cases in Perlis without going higher. Regardless, we can conclude that vaccination might be one of the contributing factor to reduce daily covid cases in Malaysia. However, more research is needed in order to conclude this finding.
''')

st.markdown('''
### How has the vaccination rate changed across the nation?
''')
vax_pop_percentage = vax_malaysia['cumul_vaccine']
vax_pop_percentage = pd.DataFrame(vax_pop_percentage)
vax_pop_percentage['percentage'] = vax_pop_percentage.apply(lambda x: (vax_malaysia['cumul_vaccine']/population[population['state']=='Malaysia']['pop'].item())*100,axis=0)
vax_pop_percentage['date'] = pd.to_datetime(vax_malaysia['date'])

st.plotly_chart(px.line(vax_pop_percentage, x='date', y='percentage', labels={'date':'Date', 'percentage':'Vaccination Rate (%)'}))

st.markdown('''
### Vaccine Distribution Modelling
''')

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['pfizer'], name='Pfizer', mode='lines'), secondary_y=False)
fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['sinovac'], name='Sinovac', mode='lines'), secondary_y=False)
fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['astra'], name='Astrazeneca', mode='lines'), secondary_y=False)
fig.add_trace(go.Line(x=vax_malaysia_all_attributes['date'], y=vax_malaysia_all_attributes['cansino'], name='Cansino', mode='lines'), secondary_y=False)
fig.update_layout(title='Vaccine brand usage over time')
st.plotly_chart(fig)

vaccine_totals = pd.DataFrame(vax_malaysia_all_attributes[['pfizer', 'astra', 'sinovac']].sum().reset_index())
vaccine_totals.columns = ['vaccine', 'total']

# plotly pie chart
vaccines_pie = px.pie(vaccine_totals, values='total', names='vaccine', title='Vaccine distribution')
st.plotly_chart(vaccines_pie)

st.write('''
Pfizer is the most used vaccine in Malaysia, followed by Sinovac and then Astrazeneca. If you observe the usage of Astrazeneca, it flails in comparison to the other two because it was opened up for voluntary registrations. Furthermore, unlike Pfizer and Sinovac, Astrazeneca usage does not show an upward trend and only has a few spikes, which may be attributed to the government opening up registrations.
''')

# ====================================================================
# CLUSTERING
# ====================================================================
st.markdown('''
## Clustering
In this section, we will employ different clustering algorithms to see whether
states suffering from Covid-19 form any visible patterns. 🦠
''')

st.markdown('''
### How did the clusters change over time with respect to cases and deaths? Did some states reorganise into new clusters?
''')
states = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']
dates = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01']
ranges_1 = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

date_range_2d = st.select_slider('Slide to see the clusters moving through time using temporal K-Means clustering.', options=ranges_1, key='2d')

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

st.markdown('''
We can observe that in the beginning, the majority of the clusters were positioned towards the bottom left and they maintain a similar pattern until about August 2020. In August, the cases were still high but there were fewer deaths, which may signify that the situation was improving, besides the one state that is in the upper corner of the plot that stands out from the rest. Around December, the bottom-right cluster begins to break up and states start moving diagonally upwards in the graph, meaning higher number of deaths and more cases. By September 2021, the states fall in a sort of straight diagonal line, with the performance of states spread across the spectrum from mild to serious.
**We also set out to find that one state that created a cluster of it's own throughout and it was Selangor, to no one's surprise.**
''')


st.markdown('''
### How do clusters change over time with respect to cases, deaths and vaccinations? Did some states reorganise into different clusters?

This time, let's bring in a third variable- vaccinations.
''')

nrows = 5
ncols = 4
dates = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01']
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
    vaccinations.append(vax_state_date[vax_state_date['state'] == state]['daily_full'].sum())

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
We maintain that states can be grouped into 3 clusters. Throughout 2020, vaccinations are 0 and hence, the clusters slowly start to expand in terms of cases and deaths. By 2021, the states have spread reasonably wide throughout the 3 dimensions. By early 2021, cases and deaths are at an all time high. Around this time, vaccination begins and clusters start moving higher in that dimension. However, even in September, there is one cluster of states with relatively low vaccinations.
''')

st.markdown('''
### Which states require attention in terms of their vaccination campaign and deaths (relatively)?
''')

dates = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01']
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
    deaths.append(deaths_state_date[deaths_state_date['state'] == state]['deaths_new'].sum() / population.loc[population['state'] == state, 'pop'].values[0])
    # divide number of vaccinations by state population
    vaccination_rates.append(vax_state_date[vax_state_date['state'] == state]['daily_full'].sum() / population.loc[population['state'] == state, 'pop'].values[0])

cases_deaths_vaccinations = pd.DataFrame({"state": states, "deaths": deaths, "vaccination_rate": vaccination_rates})
    # for col in cases_deaths_vaccinations:
    #     if cases_deaths_vaccinations[col].dtype != 'object':
    #         cases_deaths_vaccinations[col] = StandardScaler().fit_transform(cases_deaths_vaccinations[[col]])

cases_deaths_vaccinations.drop(columns=['state'], inplace=True)
X_std = cases_deaths_vaccinations

km = KMeans(n_clusters=3, max_iter=100)
y_clusters = km.fit_predict(cases_deaths_vaccinations)
centroids = km.cluster_centers_

# put back state column
cases_deaths_vaccinations['state'] = states
cases_deaths_vaccinations['cluster'] = y_clusters
vaccination_deaths = px.scatter(cases_deaths_vaccinations, x="vaccination_rate", y="deaths", color="cluster", labels={"vaccination_rate": "Vaccination Rate", "deaths": "Death Rate"})

st.plotly_chart(vaccination_deaths)

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
    # row, col, num = grid_combos[i]

    # ax = fig.add_subplot(row, col, num)
    # ax.scatter(cases_deaths_vaccinations['vaccination_rate'], cases_deaths_vaccinations['deaths'], c=y_clusters, cmap='viridis')
    # ax.set_xlabel('Vaccination Rate')
    # ax.set_ylabel('Death Rate')
    # # ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
    # # ax.set_yticks(np.arange(-0.05, 3.5, 0.5))
    # ax.set_title(f"{date_range[0]} - {date_range[1]}")
