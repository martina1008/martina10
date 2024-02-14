import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor



data_description = st.markdown("<span style='color:green'><b>Input Data:</b> upload an .xlsx file containing numerical and categorical variables for each building.</span>", unsafe_allow_html=True)
st.markdown("<span style='color:green'><b>Listed as below:</b></span>", unsafe_allow_html=True)


"""
    

    + **name:** building name

    + **function:** activity done in the building [storage, recreational center, offices, school, others]

    + **building_shape:** the typology of the building [rectangular/squared, C L T, circular or irregular]

    + **surface1floor:** [m²]

    + **gross_volume:** total gross volume [m³]

    + **shading:** presence of a shading next to the building, natural or artificial [no, yes, partial]

    + **surrounding:** ground within 5m from the building [asphalt, green, half green/half asphalt]

    + **avg_occupants:** number of people on average present in the building

    + **usage:** how much the building is used [all year, half year, rare]

    + **generation_power:** total power of generation system, heating [kW]

    + **ceiling:** not heated ceiling [present – not present]

    + **EUI:** Thermal energy/ heated surface [kWh/m²]
    """


def remove_description():
    st.session_state['data_description'] = False


def custom_float_format(value):
    # Format the value to display only 2 decimal places
    return "{:.2f}".format(value)


def to_numeric(data):
    df = data.copy()
    df['function'] = df['function'].map({'storage': 1, 'recreational center': 2, 'offices': 3,
                                         'emergency service': 4, 'school': 5, 'others': 6})
    df['building_shape'] = df['building_shape'].map(
        {'rectangular/squared': 1, 'C L T': 2, 'circular or irregular': 3})
    df['shading'] = df['shading'].map({'no': 0, 'yes': 1, 'partial': 2})
    df['surrounding'] = df['surrounding'].map(
        {'asfalt': 2, 'green': 1, 'other soil': 4, 'half green/half asfalt': 3})
    df['usage'] = df['usage'].map({'rare': 0, 'half year': 2, 'all year': 1})
    df['ceiling'] = df['ceiling'].map({'present': 1, 'not present': 0})
    return df


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


def insert_new_data():
    if (st.session_state['name'] == '') or (st.session_state['function'] is None) or (
            st.session_state['building_shape'] is None) \
            or (st.session_state['surface1floor'] == 0) or (st.session_state['gross_volume'] == 0) \
            or (st.session_state['shading'] is None) or (st.session_state['surrounding'] is None) \
            or (st.session_state['avg_occupants'] == 0) or (st.session_state['usage'] is None) \
            or (st.session_state['generation_power'] == 0) or (st.session_state['ceiling'] is None) \
            or (st.session_state['eui'] == 0):
        pass
    else:
        new_index = len(st.session_state['data'])
        new_row = pd.Series({'name': st.session_state.name,
                             'function': st.session_state.function,
                             'building_shape': st.session_state.building_shape,
                             'surface1floor': st.session_state.surface1floor,
                             'gross_volume': st.session_state.gross_volume,
                             'shading': st.session_state.shading,
                             'surrounding': st.session_state.surrounding,
                             'avg_occupants': st.session_state.avg_occupants,
                             'usage': st.session_state.usage,
                             'generation_power': st.session_state.generation_power,
                             'ceiling': st.session_state.ceiling,
                             'EUI': st.session_state.eui})
        st.session_state['data'].loc[new_index] = new_row
        # model re-training
        new_X = to_numeric(st.session_state['data'])[['function', 'building_shape', 'surface1floor', 'gross_volume',
                                                      'shading', 'surrounding', 'avg_occupants', 'usage',
                                                      'generation_power', 'ceiling']].values
        new_y = st.session_state['data']['EUI'].values
        st.session_state['model'].fit(new_X, new_y)


def prediction():
    if (st.session_state['function_pred'] is None) or (st.session_state['building_shape_pred'] is None) \
            or (st.session_state['surface1floor_pred'] == 0) or (st.session_state['gross_volume_pred'] == 0) \
            or (st.session_state['shading_pred'] is None) or (st.session_state['surrounding_pred'] is None) \
            or (st.session_state['avg_occupants_pred'] == 0) or (st.session_state['usage_pred'] is None) \
            or (st.session_state['generation_power_pred'] == 0) or (st.session_state['ceiling_pred'] is None):
        pass
    else:
        function_inserted = \
            {'storage': 1, 'recreational center': 2, 'offices': 3, 'emergency service': 4, 'school': 5, 'others': 6}[
                st.session_state['function_pred']]
        building_shape_inserted = {'rectangular/squared': 1, 'C L T': 2, 'circular or irregular': 3}[
            st.session_state['building_shape_pred']]
        surface1floor_inserted = st.session_state['surface1floor_pred']
        gross_volume_inserted = st.session_state['gross_volume_pred']
        shading_inserted = {'no': 0, 'yes': 1, 'partial': 2}[st.session_state['shading_pred']]
        surrounding_inserted = {'asfalt': 2, 'green': 1, 'other soil': 4, 'half green/half asfalt': 3}[
            st.session_state['surrounding_pred']]
        avg_occupants_inserted = st.session_state['avg_occupants_pred']
        usage_inserted = {'rare': 0, 'half year': 2, 'all year': 1}[st.session_state['usage_pred']]
        generation_power_inserted = st.session_state['generation_power_pred']
        ceiling_inserted = {'present': 1, 'not present': 0}[st.session_state['ceiling_pred']]

        st.session_state['prediction'] = st.session_state['model'].predict(
            np.array([function_inserted, building_shape_inserted, surface1floor_inserted,
                      gross_volume_inserted, shading_inserted, surrounding_inserted,
                      avg_occupants_inserted, usage_inserted, generation_power_inserted,
                      ceiling_inserted]).reshape(1, -1))


st.set_page_config(layout="wide")
st.write("<h1 style='text-align: center;'>Benchmarking and Predictive Tool</h1>", unsafe_allow_html=True)

if 'data_description' not in st.session_state:
    st.write(data_description)

uploaded_file = st.file_uploader('Choose a file', label_visibility='hidden', on_change=remove_description)
if uploaded_file is not None and not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
    st.error('The Uploaded File must be a csv or xlsx File.')
if uploaded_file is not None and (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
    if uploaded_file.name.endswith('.xlsx'):
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_excel(uploaded_file,
                                                     usecols=['name', 'function', 'building_shape',
                                                              'surface1floor', 'gross_volume', 'shading',
                                                              'surrounding', 'avg_occupants', 'usage',
                                                              'generation_power', 'ceiling', 'EUI'])

            numerical_columns = st.session_state['data'].select_dtypes(include='float64').columns
            st.session_state['data'][numerical_columns] = st.session_state['data'][numerical_columns].round(2)

        st.write("<h2 style='text-align: center;'>Dataset</h2>", unsafe_allow_html=True)

        # AgGrid(st.session_state['data'], height=500, fit_columns_on_grid_load=True)
        st.write(
            f"<div style='display: flex; justify-content: center; overflow-x: auto; height: {300}px;'>"
            "<style>table.dataframe {text-align: center; font-size: 12px;}</style>"
            + st.session_state['data'].to_html(classes='dataframe') +
            "</div>",
            unsafe_allow_html=True
        )

    if uploaded_file.name.endswith('.csv'):
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_csv(uploaded_file,
                                                   usecols=['name', 'function', 'building_shape',
                                                            'surface1floor', 'gross_volume', 'shading',
                                                            'surrounding', 'avg_occupants', 'usage',
                                                            'generation_power', 'ceiling', 'EUI'])

            numerical_columns = st.session_state['data'].select_dtypes(include='float64').columns
            st.session_state['data'][numerical_columns] = st.session_state['data'][numerical_columns].round(2)

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
        X = to_numeric(st.session_state['data'])[['function', 'building_shape', 'surface1floor', 'gross_volume',
                                                  'shading', 'surrounding', 'avg_occupants', 'usage',
                                                  'generation_power', 'ceiling']].values
        y = st.session_state['data']['EUI'].values
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        st.session_state['model'] = model

    st.write('---')

    st.write("<h2 style='text-align: center;'>Insert New Data</h2>", unsafe_allow_html=True)
    with st.form("Insert New Data", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input('name', key='name')
        function = col2.selectbox('function', ('storage', 'recreational center', 'offices', 'emergency service',
                                               'school', 'others'), index=None, key='function')
        building_shape = col1.selectbox('building_shape', ('rectangular/squared', 'C L T', 'circular or irregular'),
                                        index=None, key='building_shape')
        surface1floor = col2.number_input('surface1floor', min_value=0, step=200,
                                          key='surface1floor')
        gross_volume = col1.number_input('gross_volume', min_value=0, step=1000,
                                         key='gross_volume')
        shading = col2.selectbox('shading', ('yes', 'no', 'partial'), index=None, key='shading')
        surrounding = col1.selectbox('surrounding', ('asfalt', 'green', 'other soil', 'half green/half asfalt'),
                                     index=None, key='surrounding')
        avg_occupants = col2.number_input('avg_occupants', min_value=0, step=10,
                                          key='avg_occupants')
        usage = col1.selectbox('usage', ('rare', 'all year', 'half year'), index=None, key='usage')
        generation_power = col2.number_input('generation_power', min_value=0, step=50, key='generation_power')
        ceiling = col1.selectbox('ceiling', ('present', 'not present'), index=None, key='ceiling')
        eui = col2.number_input('EUI', min_value=0.0, step=50.0, key='eui')
        submit_button = col1.form_submit_button('Submit', on_click=insert_new_data)
        if submit_button:
            if (st.session_state['name'] == '') or (st.session_state['function'] is None) or (
                    st.session_state['building_shape'] is None) \
                    or (st.session_state['surface1floor'] == 0) or (st.session_state['gross_volume'] == 0) \
                    or (st.session_state['shading'] is None) or (st.session_state['surrounding'] is None) \
                    or (st.session_state['avg_occupants'] == 0) or (st.session_state['usage'] is None) \
                    or (st.session_state['generation_power'] == 0) or (st.session_state['ceiling'] is None) \
                    or (st.session_state['eui'] == 0):
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

    building = st.selectbox(
        "Select a Building for the Benchmarking",
        [name for name in st.session_state['data']['name']],
        index=None,
        label_visibility='collapsed'
    )

    sorted_data = st.session_state['data'].sort_values(by='EUI').reset_index(drop=True)
    mean_eui = st.session_state['data']['EUI'].mean()

    if building:
        ind = sorted_data.loc[building == sorted_data['name']].index[0]
        eui = sorted_data.loc[ind, 'EUI']
        delta_perc = (eui - mean_eui) / mean_eui * 100
        colors[ind] = 'red'
        fig = px.bar(sorted_data, x=sorted_data['EUI'])
        fig.add_vline(x=mean_eui, line=dict(color='red', width=3))
        fig.update_traces(marker=dict(color=colors, line=dict(width=0.5)), width=0.8)
        fig.update_layout(xaxis_title='EUI', yaxis_title='Building', width=1100, height=600)
        st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        col1.dataframe(sorted_data[['name', 'EUI']], height=245)

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
        function = col1.selectbox('function', ('storage', 'recreational center', 'offices', 'emergency service',
                                               'school', 'others'), index=None, key='function_pred')
        building_shape_pred = col2.selectbox('building_shape',
                                             ('rectangular/squared', 'C L T', 'circular or irregular'),
                                             index=None, key='building_shape_pred')
        surface1floor_pred = col1.number_input('surface1floor', min_value=0, step=200,
                                               key='surface1floor_pred')
        gross_volume_pred = col2.number_input('gross_volume', min_value=0, step=1000,
                                              key='gross_volume_pred')
        shading_pred = col1.selectbox('shading', ('yes', 'no', 'partial'), index=None, key='shading_pred')
        surrounding_pred = col2.selectbox('surrounding', ('asfalt', 'green', 'other soil', 'half green/half asfalt'),
                                          index=None, key='surrounding_pred')
        avg_occupants_pred = col1.number_input('avg_occupants', min_value=0, step=10,
                                               key='avg_occupants_pred')
        usage_pred = col2.selectbox('usage', ('rare', 'all year', 'half year'), index=None, key='usage_pred')
        generation_power_pred = col1.number_input('generation_power', min_value=0, step=50, key='generation_power_pred')
        ceiling_pred = col2.selectbox('ceiling', ('present', 'not present'), index=None, key='ceiling_pred')
        submit_button_pred = col1.form_submit_button('Submit', on_click=prediction)

    if submit_button_pred:
        if (st.session_state['function_pred'] is None) or (st.session_state['building_shape_pred'] is None) \
                or (st.session_state['surface1floor_pred'] == 0) or (st.session_state['gross_volume_pred'] == 0) \
                or (st.session_state['shading_pred'] is None) or (st.session_state['surrounding_pred'] is None) \
                or (st.session_state['avg_occupants_pred'] == 0) or (st.session_state['usage_pred'] is None) \
                or (st.session_state['generation_power_pred'] == 0) or (st.session_state['ceiling_pred'] is None):
            error = st.error('Fill the Missing Fields')
            time.sleep(5)
            error.empty()
        else:
            if 'prediction' in st.session_state:
                st.write(f"<span style='font-size:{25}px;'>Predicted EUI: {round(st.session_state['prediction'][0], 2)}\
                </span>", unsafe_allow_html=True)

