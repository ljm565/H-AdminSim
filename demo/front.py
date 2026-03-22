import os
import sys
import json
import requests
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
import argparse
from importlib import resources

import streamlit as st
from streamlit_chat import message



# Initialize
parser = argparse.ArgumentParser()
parser.add_argument('--backend_host', default='127.0.0.1', type=str, required=False, help='Backend FastAPI host address')
parser.add_argument('--backend_port', default=8778, type=int, required=False, help='Backend FastAPI port')
args = parser.parse_args()
BASE_URL = f'http://{args.backend_host}:{args.backend_port}'


# Initialize session states
def initialize_session():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'content': 'Hello, how can I help you?', 'type': 'bot'}
        ]
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'current_task' not in st.session_state:
        st.session_state.current_task = 'intake'


# Reset session state
def reset_session():
    if st.session_state.session_id is not None:
        try:
            requests.delete(f'{BASE_URL}/session/{st.session_state.session_id}')
        except Exception as e:
            st.error(f'Error deleting session: {e}')

    st.session_state.chat_history = [
        {'content': 'How can I help you?', 'type': 'bot'}
    ]
    st.session_state.session_id = None
    st.session_state.current_task = 'intake'
    st.success('Session has been reset.')


# Save dialogue history
def save_history():
    if st.session_state.session_id is None:
        st.warning('No active session to save.')
        return
    try:
        response = requests.post(f'{BASE_URL}/save_history/{st.session_state.session_id}')
        response = response.json()
        st.success(f"History saved: {response['file_path']}")
    except Exception as e:
        st.error(f'Error saving history: {e}')


# Prepare hospital information
def prepare_hospital_info():
    file_path = resources.files("h_adminsim.assets.demo").joinpath('hospital.json')
    with open(file_path, 'r') as f:
        hospital_info = json.load(f)
    return hospital_info


# Send user prompt via FastAPI
def send_message_to_chatbot(api_request_data, task):
    try:
        api_url = f'{BASE_URL}/llm_{task}_response'
        response = requests.post(api_url, json=api_request_data)
        response = response.json()

        if 'session_id' in response:
            st.session_state.session_id = response['session_id']

        return response

    except Exception as e:
        st.error(f'Error: {e}')
        return 'Error occurred while communicating with the server.'



# ------------------- Page design -------------------
st.set_page_config(layout='wide')

# Session inicialization
initialize_session()

# Divide the page into two columns
col1, col2 = st.columns([0.8, 2])

# Hospital information
with col1:
    st.header('🏥 Hospital')
    hospital_info = prepare_hospital_info()
    metadata = hospital_info['metadata']

    # Metadata
    st.markdown(f"**🏨 {metadata['hospital_name']}**")
    st.markdown(f"📅 {metadata['start_date']} ~ {metadata['end_date']}")
    st.markdown(f"🕐 {metadata['time']['start_hour']:.0f}:00 ~ {metadata['time']['end_hour']:.0f}:00")

    # Buttons
    if st.button('🔄 Reset Session', use_container_width=True):
        reset_session()
    if st.button('💾 Save History', use_container_width=True):
        save_history()

    st.divider()
    
    # Departments
    for dept_name, dept_data in hospital_info['department'].items():
        with st.expander(f"**{dept_name.capitalize()}** (Code: `{dept_data['code']}`)", expanded=False):
            for doctor in dept_data['doctor']:
                st.markdown(f"👨‍⚕️ {doctor}")

# Chat interface
with col2:
    st.header('💬 Staff Agent')
    
    # Init necessaries
    chat_history_expander = st.expander('Chat History', expanded=True)

    # Chat input and button
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('Message: ', '', key='input')
        submit_button = st.form_submit_button('Send')

        with chat_history_expander:
            # User input
            if submit_button and user_input.strip() != '':
                st.session_state.chat_history.append({'content': user_input, 'type': 'user'})
                response = send_message_to_chatbot(
                    {
                        'user_prompt': user_input,
                        'session_id': st.session_state.session_id,
                    },
                    st.session_state.current_task
                )
                if response.get('task', 'intake') == 'schedule':                    
                    response = response.get('response', "No response field in the server's reply.")
                    st.session_state.chat_history.append({'content': str(response), 'type': 'bot'})
                    st.session_state.current_task = 'schedule'
                else:
                    response = response.get('response', "No response field in the server's reply.")
                    st.session_state.chat_history.append({'content': response, 'type': 'bot'})

            # Logging all previous conversations
            for i, past_chat in enumerate(st.session_state.chat_history):
                if past_chat['type'] == 'bot':
                    message(past_chat['content'], key=f'bot_{i}')
                else:
                    message(past_chat['content'], is_user=True, key=f'user_{i}')
