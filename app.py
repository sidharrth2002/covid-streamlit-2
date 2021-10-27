import streamlit as st
import pandas as pd
import pydeck as pdk

from urllib.error import URLError

state_locations = []

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

cases_state = pd.read_csv('./cases/epidemic/cases_state.csv')
cases_state = cases_state[cases_state['date'] == '2021-10-09']
cases_state = cases_state.groupby('state').sum()
cases_state = cases_state.reset_index()
cases_state_locations = cases_state.merge(state_locations, on='state')

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
        ),
        # "Bart Stop Exits": pdk.Layer(
        #     "ScatterplotLayer",
        #     data=cases_state_locations.to_json(),
        #     get_position=["lon", "lat"],
        #     get_color=[200, 30, 0, 160],
        #     get_radius=["exits"],
        #     radius_scale=0.05,
        # ),
        # "Bart Stop Names": pdk.Layer(
        #     "TextLayer",
        #     data=from_data_file("bart_stop_stats.json"),
        #     get_position=["lon", "lat"],
        #     get_text="name",
        #     get_color=[0, 0, 0, 200],
        #     get_size=15,
        #     get_alignment_baseline="'bottom'",
        # ),
        # "Outbound Flow": pdk.Layer(
        #     "ArcLayer",
        #     data=from_data_file("bart_path_stats.json"),
        #     get_source_position=["lon", "lat"],
        #     get_target_position=["lon2", "lat2"],
        #     get_source_color=[200, 30, 0, 160],
        #     get_target_color=[200, 30, 0, 160],
        #     auto_highlight=True,
        #     width_scale=0.0001,
        #     get_width="outbound",
        #     width_min_pixels=3,
        #     width_max_pixels=30,
        # ),
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

