import os
import random

import pandas as pd
import requests
import streamlit as st

# Configuration
root_data_dir = '/srv/data'
data_path = os.path.join(root_data_dir, 'client_segmentation.csv')
api_url = 'http://demo_api_service:8000/classifier/'

# Load data
@st.cache_data
def load_data():
    """Load customer segmentation data."""
    df = pd.read_csv(data_path)
    return df

# Initialize app
st.title('Customer Segmentation Classifier')
st.write('Select a customer ID to predict their class using the classifier API')

# Load data
try:
    df_source = load_data()
    num_rows = df_source.shape[0]

    st.write(f'Dataset loaded: {num_rows} customers')

    # Generate 5 random IDs
    if 'random_ids' not in st.session_state:
        st.session_state.random_ids = random.sample(range(num_rows), min(5, num_rows))

    # Create dropdown with random IDs
    selected_id = st.selectbox(
        'Select Customer ID',
        options=st.session_state.random_ids,
        format_func=lambda x: f"Customer #{x}"
    )

    # Button to generate new random IDs
    if st.button('Generate New Random IDs'):
        st.session_state.random_ids = random.sample(range(num_rows), min(5, num_rows))
        st.rerun()

    # Display selected customer features
    st.subheader('Customer Features')

    customer_row = df_source.iloc[selected_id]

    # Extract features (matching train.py)
    x1 = float(customer_row['call_diff'])
    x2 = float(customer_row['sms_diff'])
    x3 = float(customer_row['traffic_diff'])
    actual_class = customer_row['customes_class']

    # Display features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('x1 (call_diff)', f'{x1:.2f}')
    with col2:
        st.metric('x2 (sms_diff)', f'{x2:.2f}')
    with col3:
        st.metric('x3 (traffic_diff)', f'{x3:.2f}')

    st.metric('Actual Class', int(actual_class))

    # Predict button
    if st.button('Predict Class', type='primary'):
        with st.spinner('Calling classifier API...'):
            try:
                # Make API request
                params = {
                    'x1': x1,
                    'x2': x2,
                    'x3': x3
                }
                response = requests.get(api_url, params=params)
                response.raise_for_status()

                result = response.json()

                # Display results
                st.success('Prediction completed!')

                predicted_class = result['prediction']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Predicted Class', predicted_class)
                with col2:
                    match = predicted_class == int(actual_class)
                    st.metric('Match', '✅' if match else '❌')

                # Display probabilities
                if 'probabilities' in result:
                    st.subheader('Class Probabilities')
                    probs = result['probabilities']

                    # Create DataFrame for better display
                    prob_df = pd.DataFrame([
                        {'Class': k.replace('class_', ''), 'Probability': f'{v:.4f}'}
                        for k, v in probs.items()
                    ])
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                # Show raw API response
                with st.expander('Raw API Response'):
                    st.json(result)

            except requests.exceptions.RequestException as e:
                st.error(f'API request failed: {e}')
            except Exception as e:
                st.error(f'Error: {e}')

    # Display sample of dataset
    with st.expander('View Dataset Sample'):
        st.dataframe(df_source.head(10))

except FileNotFoundError:
    st.error(f'Data file not found at {data_path}')
    st.info('Please ensure the client_segmentation.csv file exists in the data directory.')
except Exception as e:
    st.error(f'Error loading data: {e}')