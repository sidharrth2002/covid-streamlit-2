from seaborn.matrix import heatmap
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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

cases_state_locations = cases_state.copy()
cases_state_locations = cases_state_locations[cases_state_locations['date'] == '2021-10-09']
cases_state_locations = cases_state_locations.groupby('state').sum()
cases_state_locations = cases_state_locations.reset_index()
cases_state_locations = cases_state_locations.merge(state_locations, on='state')

try:
    ALL_LAYERS = {
        "Covid Cases": pdk.Layer(
            "HexagonLayer",
            data=cases_state_locations.to_json(),
            get_position='[lon, lat]',
            get_text="state",
            get_radius="[cases_new]",
            elevation_scale=10,
            # elevation_range=[0, 1000],
            extruded=True,
        )
    }
    st.sidebar.markdown('### Map Layers')
    selected_layers = [
        layer for layer_name, layer in ALL_LAYERS.items()
        if st.sidebar.checkbox(layer_name, True)]
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
st.markdown('### Correlation between mean income and cases')
# merge supplementary income dataset with the cases dataset on state
cases_income = pd.DataFrame(cases_state.groupby('state')['cases_new'].sum()).reset_index()
st.write(income)
cases_income = cases_income.merge(income, on='state')
cases_income['income'] = cases_income['income'].astype(int)

plot = px.scatter(cases_income, x='cases_new', y='income', color='cases_new', size='cases_new', labels={'cases_new':'Cases', 'income': 'Income'}, title='Correlation plot between mean income and cases')

# add plotly to streamlit
st.plotly_chart(plot)


# =============================================================================
# Is there any correlation between vaccination and daily cases for Selangor, Sabah, Sarawak, and many more?
# =============================================================================
# prepare a generic function that calculates the cumulative sum of cases and percentage vaccinated for each state
st.markdown('### Correlation between vaccination and cases')
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
corr_sabah2,vax_percentage_sabah2 = cases_vax_corr('Sabah',2)
corr_sarawak1,vax_percentage_sarawak1= cases_vax_corr('Sarawak',1)
corr_sarawak2,vax_percentage_sarawak2= cases_vax_corr('Sarawak',2)

st.plotly_chart(px.imshow(corr_selangor1))

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

st.markdown('### What type of Covid-19 clusters are most prevalent?')

st.plotly_chart(px.box(clusters_singlestate, x='category', y='cases_total', points=False))
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
fig, ax = plt.subplots(figsize=(25, 10))
ax.plot(Brunei['date'], Brunei['percentage_vaccinated'], label = "Brunei")
ax.plot(Myanmar['date'], Myanmar['percentage_vaccinated'], label = "Myanmar")
ax.plot(Cambodia['date'], Cambodia['percentage_vaccinated'], label = "Cambodia")
ax.plot(Indonesia['date'], Indonesia['percentage_vaccinated'], label = "Indonesia")
ax.plot(Laos['date'], Laos['percentage_vaccinated'], label = "Laos")
ax.plot(Malaysia['date'], Malaysia['percentage_vaccinated'], label = "Malaysia")
ax.plot(Philippines['date'], Philippines['percentage_vaccinated'], label = "Philippines")
ax.plot(Singapore['date'], Singapore['percentage_vaccinated'], label = "Singapore")
ax.plot(Thailand['date'], Thailand['percentage_vaccinated'], label = "Thailand")
ax.plot(Vietnam['date'], Vietnam['percentage_vaccinated'], label = "Vietnam")
ax.set_xticks(ax.get_xticks()[::10])
ax.tick_params(axis='x', labelrotation=90)
plt.legend()
st.pyplot(fig)

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

fig = plt.figure(figsize=(15, 10))
sns.lineplot(x='date', y='pfizer', data=vax_malaysia_all_attributes)
sns.lineplot(x='date', y='sinovac', data=vax_malaysia_all_attributes)
sns.lineplot(x='date', y='astra', data=vax_malaysia_all_attributes)
sns.lineplot(x='date', y='cansino', data=vax_malaysia_all_attributes)
# only show some xtick labels
plt.xticks([date for i, date in enumerate(vax_malaysia_all_attributes['date']) if i % 10 == 0], rotation=45)
plt.legend(['Pfizer', 'Sinovac', 'Astra', 'Cansino'])
st.pyplot(fig)
vaccine_totals = pd.DataFrame(vax_malaysia_all_attributes[['pfizer', 'astra', 'sinovac']].sum().reset_index())
vaccine_totals.columns = ['vaccine', 'total']

# plotly pie chart
vaccines_pie = px.pie(vaccine_totals, values='total', names='vaccine', title='Vaccine Distribution')
st.plotly_chart(vaccines_pie)