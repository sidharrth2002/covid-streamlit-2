from seaborn.matrix import heatmap
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

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


# TITLE
st.title("A Comprehensive Exploration of Covid-19 in Malaysia 🇲🇾")

# DISPLAY MAP WITH CASES AND DEATHS
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
st.markdown(''''
The detention centers generally have the largest Covid clusters, with a strong right skew. The rest of the cluster categories have mostly small clusters, but quite a few unusually large clusters. This is especially for the workplace clusters. For instance, most companies/organisations in Malaysia are small, so there may be a lot of clusters, but only some are large enought to appear as an outlier.
''')