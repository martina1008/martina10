import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


def insert_new_data():
    if (st.session_state['name'] == '') or (st.session_state['shading'] == '') \
            or (st.session_state['building_shape'] == '') \
            or (st.session_state['gross_volume'] == '') or (st.session_state['avg_number_occupants'] == '') \
            or (st.session_state['sv'] == '') or (st.session_state['eui'] == ''):
        pass
    else:
        new_index = len(st.session_state['data'])
        new_row = pd.Series({'Name': st.session_state.name,
                             'Shading': st.session_state.shading,
                             'Building_shape': st.session_state.building_shape,
                             'Gross_volume': st.session_state.gross_volume,
                             'Avg_number_occupants': st.session_state.avg_number_occupants,
                             'S/V': st.session_state.sv,
                             'EUI': st.session_state.eui})
        st.session_state['data'].loc[new_index] = new_row
        # model re-training
        new_X = st.session_state['data'][['Building_shape', 'Gross_volume', 'S/V',
                                          'Shading', 'Avg_number_occupants']].values
        new_y = st.session_state['data']['EUI'].values
        st.session_state['model'].fit(new_X, new_y)


def prediction():
    if (st.session_state['shading_pred'] is None) or (st.session_state['building_shape_pred'] is None) \
            or (st.session_state['gross_volume_pred'] == 0) or (st.session_state['avg_number_occupants_pred'] == 0) \
            or (st.session_state['sv_pred'] == 0):
        pass
    else:
        st.session_state['prediction'] = st.session_state['model'].predict(
            np.array([st.session_state['building_shape_pred'],
                      st.session_state['gross_volume_pred'],
                      st.session_state['sv_pred'],
                      st.session_state['shading_pred'],
                      st.session_state['avg_number_occupants_pred']
                      ]).reshape(1, -1))


st.set_page_config(layout="wide")
st.write("<h1 style='text-align: center;'>Benchmarking and Predictive Tool</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose a file', label_visibility='hidden')
if uploaded_file is not None and not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
    st.error('The Uploaded File must be a csv or xlsx File.')
if uploaded_file is not None and (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
    if uploaded_file.name.endswith('.xlsx'):
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_excel(uploaded_file,
                                                     usecols=['Name', 'Shading', 'Building_shape', 'Gross_volume',
                                                              'Avg_number_occupants',
                                                              'S/V', 'EUI'])
        st.write("<h2 style='text-align: center;'>Dataset</h2>", unsafe_allow_html=True)
        # AgGrid(st.session_state['data'], height=500, fit_columns_on_grid_load=True)
        st.write(
            f"<div style='display: flex; justify-content: center; overflow-x: auto; height: {300}px;'>"
            "<style>table.dataframe {text-align: center;}</style>"
            + st.session_state['data'].to_html(classes='dataframe') +
            "</div>",
            unsafe_allow_html=True
        )
    if uploaded_file.name.endswith('.csv'):
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_csv(uploaded_file,
                                                   usecols=['Name', 'Shading', 'Building_shape', 'Gross_volume',
                                                            'Avg_number_occupants',
                                                            'S/V', 'EUI'])
        st.write("<h3 style='text-align: center;'>Dataset</h3>", unsafe_allow_html=True)
        # AgGrid(st.session_state['data'], height=600, fit_columns_on_grid_load=True)
        st.write(
            f"<div style='display: flex; justify-content: center; overflow-x: auto; height: {300}px;'>"
            "<style>table.dataframe {text-align: center;}</style>"
            + st.session_state['data'].to_html(classes='dataframe') +
            "</div>",
            unsafe_allow_html=True
        )
    # Add some space between the DataFrame and the button
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    excel_file = convert_df(st.session_state['data'])
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col4:
        st.download_button(
            label="Download the Data",
            data=excel_file,
            file_name='dati.xlsx',
            mime='text/xlsx'
        )

    # model training
    if 'model' not in st.session_state:
        X = st.session_state['data'][
            ['Building_shape', 'Gross_volume', 'S/V', 'Shading', 'Avg_number_occupants']].values
        y = st.session_state['data']['EUI'].values
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        st.session_state['model'] = model

    st.write('---')

    st.write("<h2 style='text-align: center;'>Insert New Data</h2>", unsafe_allow_html=True)
    with st.form("Insert New Data", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input('Name', key='name')
        shading = col2.selectbox('Shading', (0, 1, 2, 3), index=None, key='shading')
        building_shape = col1.selectbox('Building_shape', (1, 2, 3, 4, 5, 6, 7), index=None, key='building_shape')
        gross_volume = col2.number_input('Gross_volume', min_value=0, step=10, key='gross_volume')
        avg_number_occupants = col1.number_input('Avg_number_occupants', min_value=0, step=10,
                                                 key='avg_number_occupants')
        sv = col2.number_input('S/V', min_value=0, step=10, key='sv')
        eui = col1.number_input('EUI', min_value=0.0, step=50.0, key='eui')
        submit_button = col1.form_submit_button('Submit', on_click=insert_new_data)
        if submit_button:
            if (st.session_state['name'] == '') or (st.session_state['shading'] == '') or (
                    st.session_state['building_shape'] == '') \
                    or (st.session_state['gross_volume'] == '') or (st.session_state['avg_number_occupants'] == '') \
                    or (st.session_state['sv'] == '') or (st.session_state['eui'] == ''):
                error = st.error('Fill the Missing Fields')
                time.sleep(5)
                error.empty()
            else:
                success = st.success('Submitted Successfully')
                time.sleep(5)
                success.empty()

    st.write('---')

    st.write("<h2 style='text-align: center;'>Benchmarking</h2>", unsafe_allow_html=True)
    colors = ['lightgrey' for _ in range(st.session_state['data'].shape[0])]
    with (st.expander('Select a Building for the Benchmarking')):
        building = st.selectbox(
            "Select a Building for the Benchmarking",
            [name for name in st.session_state['data']['Name']],
            index=None,
            label_visibility='collapsed'
        )

        sorted_data = st.session_state['data'].sort_values(by='EUI').reset_index(drop=True)
        mean_eui = st.session_state['data']['EUI'].mean()

        if building:
            ind = sorted_data.loc[building == sorted_data['Name']].index[0]
            eui = sorted_data.loc[ind, 'EUI']
            delta_perc = (eui - mean_eui) / mean_eui * 100
            colors[ind] = 'red'
            fig = px.bar(sorted_data, x=sorted_data['EUI'])
            fig.add_vline(x=mean_eui, line=dict(color='red', width=3))
            fig.update_traces(marker=dict(color=colors, line=dict(width=0.5)), width=0.8)
            fig.update_layout(xaxis_title='EUI', yaxis_title='Building', width=1100, height=600)
            st.plotly_chart(fig)

            col1, col2 = st.columns(2)
            col1.dataframe(sorted_data[['Name', 'EUI']], height=245)

            with col2.container(border=True):
                st.write(f"<span style='font-size:{25}px;'>Building: {building}</span>", unsafe_allow_html=True)
                st.write(f"<span style='font-size:{25}px;'>Position: {ind}</span>", unsafe_allow_html=True)
                st.write(f"<span style='font-size:{25}px;'>EUI: {round(eui, 2)}</span>", unsafe_allow_html=True)
                if eui > mean_eui:
                    color = 'red'
                    arrow_symbol = '&uarr;'  # Upward arrow symbol
                else:
                    color = 'green'
                    arrow_symbol = '&darr;'  # Downward arrow symbol
                st.write(
                    f"<span style='font-size:25px;'>% Change from the Mean: <span style='color:{color};'> \
{arrow_symbol} {round(delta_perc, 2)}%</span></span>",
                    unsafe_allow_html=True)

    st.write('---')

    st.write("<h2 style='text-align: center;'>Prediction</h2>", unsafe_allow_html=True)
    with st.form("Insert the Data for the Prediction", clear_on_submit=True):
        col1, col2 = st.columns(2)
        shading_pred = col2.selectbox('Shading', (0, 1, 2, 3), index=None, key='shading_pred')
        building_shape_pred = col1.selectbox('Building_shape', (1, 2, 3, 4, 5, 6, 7), index=None,
                                             key='building_shape_pred')
        gross_volume_pred = col2.number_input('Gross_volume', min_value=0, step=10, key='gross_volume_pred')
        avg_number_occupants_pred = col1.number_input('Avg_number_occupants', min_value=0, step=10,
                                                      key='avg_number_occupants_pred')
        sv_pred = col1.number_input('S/V', min_value=0, step=10, key='sv_pred')
        submit_button_pred = col1.form_submit_button('Submit', on_click=prediction)

    if submit_button_pred:
        if (st.session_state['shading_pred'] is None) or (st.session_state['building_shape_pred'] is None) \
                or (st.session_state['gross_volume_pred'] == 0) or (
                st.session_state['avg_number_occupants_pred'] == 0) \
                or (st.session_state['sv_pred'] == 0):
            error = st.error('Fill the Missing Fields')
            time.sleep(5)
            error.empty()
        else:
            if 'prediction' in st.session_state:
                st.write(f"<span style='font-size:{25}px;'>Predicted EUI: {round(st.session_state['prediction'][0], 2)}\
                </span>", unsafe_allow_html=True)
