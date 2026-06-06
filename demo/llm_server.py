import os, sys, re
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
import json
import logging
import argparse
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from importlib import resources
from langchain_core.messages import HumanMessage, AIMessage

import uvicorn
from fastapi import FastAPI, HTTPException

import patientsim.utils as pu

from h_adminsim.registry import PromptRequest
from h_adminsim.registry.errors import ToolCallingError
from h_adminsim.environment import OPFVSchedulingSimulation
from h_adminsim.task.opfv_task import OutpatientFirstIntake
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling
from h_adminsim.agent import IntakeAdminStaffAgent, SchedulingAdminStaffAgent
from h_adminsim.utils import log, colorstr, LOGGING_NAME
from h_adminsim.utils.filesys_utils import json_load, json_save_fast


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='Staff agent model')
parser.add_argument('--backend_host', default='127.0.0.1', type=str, required=False, help='Backend FastAPI host address')
parser.add_argument('--backend_port', default=8778, type=int, required=False, help='Backend FastAPI port')
parser.add_argument('--max_rounds', default=5, type=int, required=False, help='LLM maximum interaction rounds')
parser.add_argument('--is_develop', action='store_true', required=False)
args = parser.parse_args()


# Global state
app_state = {}
sessions: dict[str, dict] = {}


# Functions
def bridge_patientsim():
    main_logger = logging.getLogger(LOGGING_NAME)
    pu.LOGGER = main_logger
    pu.LOGGING_NAME = LOGGING_NAME


def init_session(session_id: str):
    sessions[session_id] = {
        'agent': create_intake_client(session_id),
        'tool_client': None,
        'index': 0,
        'known_condition': None,
        'intake_dialog_history': [{"role": "Staff", "content": "How can I help you?"}],
        'schedule_dialog_history': [{"role": "Staff", "content": "How would you like to schedule the appointment?"}],
        'task': 'first_visit_intake'
    }


def update_session(session_id: str, **kwargs):
    for key, value in kwargs.items():
        sessions[session_id][key] = value

    if kwargs.get('task', None) == 'first_visit_scheduling':
        filtered_doctor_information = app_state['environment'].get_doctor_schedule(
            doctor_information=app_state['doctor'],
            department=sessions[session_id]['known_condition']['department'][0],
        )
        client = sessions[session_id]['agent'].build_agent(
            rule=app_state['rule'], 
            doctor_info=filtered_doctor_information
        )
        sessions[session_id]['tool_client'] = client


def create_intake_client(session_id: str) -> IntakeAdminStaffAgent:
    staff_agent = IntakeAdminStaffAgent(
        model=app_state['model'],
        department_list=app_state['departments'],
        max_inferences=app_state['max_rounds'],
        temperature=0 if 'gpt-5' not in app_state['model'].lower() else 1,
    )
    log(f'Session {session_id}: Intake staff agent initialized successfully!', color=True)
    return staff_agent


def create_scheduling_client(session_id: str) -> SchedulingAdminStaffAgent:
    staff_agent = SchedulingAdminStaffAgent(
        target_task='first_visit_scheduling',
        model=app_state['model'],
        temperature=0 if 'gpt-5' not in app_state['model'].lower() else 1,
    )
    log(f'Session {session_id}: Scheduling staff agent initialized successfully!', color=True)
    return staff_agent


def dialog_postprocessing(dialog: str, department: str) -> str:
    try:
        split_pattern = re.compile(r'\bAnswer:')    
        before_answer = re.split(split_pattern, dialog)[0].strip()
        before_answer += f' I will introduce you to a physician who work in the {department}.\n\nThen, how would you like to schedule the appointment?'
    except:
        log('Error occurred during the post-process the answer', level='error')
        before_answer = f'I will introduce you to a physician who work in the {department}.\n\nThen, how would you like to schedule the appointment?'
    return before_answer


def intake_dept_postprocessing(response: str):
    return OutpatientFirstIntake.postprocessing_department(response)


def scheduling_post_processing(response: dict, filtered_doctor_information: dict):
    ## Scheduling result
    if response['type'] == 'tool':
        schedule = OPFVSchedulingSimulation.postprocessing(
            strategy='tool_calling',
            data=response['result'],
            filtered_doctor_information=filtered_doctor_information,
        )
        response['result'] = schedule
        return response
    
    ## Dialogue
    elif response['type'] == 'text':
        if 'no tool' in response['result'].lower():
            raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
    
    ## Error
    else:
        raise TypeError(colorstr("red", "Error: Unexpected return type from scheduling method."))

    return response


def _to_lc_history(session_id: str) -> list:
    msgs = []
    for m in sessions[session_id]['schedule_dialog_history']:
        if m["role"] == "Patient":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "Staff":
            msgs.append(AIMessage(content=m["content"]))
    return msgs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log('Backend starts up!', color=True)
    bridge_patientsim()
    hospital_file_path = resources.files("h_adminsim.assets.demo").joinpath('hospital.json')
    intake_last_task_prompt_file_path = resources.files("h_adminsim.assets.prompts").joinpath('opfv_intake_staff_task_user.txt')
    hospital_info = json_load(hospital_file_path)
    intake_last_task_user_prompt = intake_last_task_prompt_file_path.read_text()
    environment = HospitalEnvironment(hospital_info)
    rule = SchedulingRule(hospital_info['metadata'], hospital_info['department'], environment)
    
    # Save to the app state
    app_state['model'] = args.model
    app_state['departments'] = list(hospital_info['department'])
    app_state['max_rounds'] = args.max_rounds
    app_state['intake_last_task_user_prompt'] = intake_last_task_user_prompt
    app_state['reasoning_kwargs'] = {'reasoning_effort': 'low'} if 'gpt-5' in args.model.lower() else {}
    app_state['environment'] = environment
    app_state['doctor'] = hospital_info['doctor']
    app_state['metadata'] = hospital_info['metadata']
    app_state['rule'] = rule

    yield

    # Shutdown
    log('Backend shuts down!', color=True)
    app_state.clear()
    sessions.clear()


app = FastAPI(lifespan=lifespan)


# APIs
@app.post('/llm_first_visit_intake_response')
async def generate_intake_response(request: PromptRequest):
    user_prompt = request.user_prompt.strip()

    if not user_prompt:
        raise HTTPException(status_code=400, detail='Prompt is empty')

    # Make a new session
    session_id = request.session_id or str(uuid4())
    if session_id not in sessions:
        init_session(session_id)

    # Intake
    sessions[session_id]['intake_dialog_history'].append({"role": "Patient", "content": user_prompt})
    intake_staff = sessions[session_id]['agent']
    intake_idx = sessions[session_id]['index']
    response = intake_staff(
        user_prompt=user_prompt + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis." \
            if intake_idx >= app_state['max_rounds'] - 1 else user_prompt,
        using_multi_turn=True,
        verbose=True,
        **app_state['reasoning_kwargs']
    )
    sessions[session_id]['index'] += 1
    sessions[session_id]['intake_dialog_history'].append({"role": "Staff", "content": response})
    _dept = intake_dept_postprocessing(response)
    
    # Flag which indicates whether the last of the intake or not
    if _dept != 'wrong':
        _patient_info = intake_staff(
            app_state['intake_last_task_user_prompt'],
            using_multi_turn=True,
            verbose=False,
            **app_state['reasoning_kwargs']
        )
        patient_info = OutpatientFirstIntake.postprocessing_information(_patient_info)
        department = patient_info.pop('department')
        
        # Ensure the department name
        department = 'rheumatology' if department.lower() == 'rheumatory' else department
        department = 'pulmonary' if department.lower() == 'pulmonology' else department
        for _dept in app_state['departments']:
            if _dept in department:
                department = _dept
                break
        
        # Final output
        structured_data = {'patient': patient_info, 'department': [department]}

        # Update intake results
        update_session(
            session_id,
            **{
                'agent': create_scheduling_client(session_id),
                'index': 0, 
                'known_condition': structured_data, 
                'task': 'first_visit_scheduling'
            }
        )
        response = dialog_postprocessing(response, department)
        sessions[session_id]['intake_dialog_history'][-1] = {"role": "Staff", "content": response}
    
    return {'response': response, 'session_id': session_id, 'task': sessions[session_id]['task']}


@app.post('/llm_first_visit_scheduling_response')
async def generate_schedule_response(request: PromptRequest):
    user_prompt = request.user_prompt.strip()
    session_id = request.session_id
    sessions[session_id]['schedule_dialog_history'].append({"role": "Patient", "content": user_prompt})

    if not user_prompt:
        raise HTTPException(status_code=400, detail='Prompt is empty')

    # Scheduling
    agent = sessions[session_id]['agent']
    client = sessions[session_id]['tool_client']
    environment = app_state['environment']
    department = sessions[session_id]['known_condition']['department'][0]
    filtered_doctor_information = environment.get_doctor_schedule(
        doctor_information=app_state['doctor'],
        department=sessions[session_id]['known_condition']['department'][0],
    )
    
    # First, try to use the tool calling
    try:
        # Invoke
        response = scheduling_tool_calling(
            client=client, 
            user_prompt=user_prompt,
            history=_to_lc_history(session_id),
        )
        response = scheduling_post_processing(
            response,
            filtered_doctor_information
        )
    
    # If tool calling fails, fallback to LLM-based scheduling
    except:
        log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')
        filtered_doctor_information = environment.get_doctor_schedule(
            doctor_information=app_state['doctor'],
            department=department,
            express_detail=True
        )
        user_prompt = agent.scheduling_user_prompt_template.format(
            START_HOUR=app_state['metadata']['time']['start_hour'],
            END_HOUR=app_state['metadata']['time']['end_hour'],
            TIME_UNIT=app_state['metadata']['time']['interval_hour'],
            CURRENT_TIME=environment.current_time,
            DEPARTMENT=department,
            PREFERENCE=user_prompt, 
            RESCHEDULING_FLAG='Not requested.',
            DAY=app_state['metadata']['days'],
            DOCTOR=json.dumps(filtered_doctor_information, indent=2),
        )
        schedule = agent(
            user_prompt,
            using_multi_turn=False,
            verbose=False,
            **app_state['reasoning_kwargs'],
        )
        schedule = OPFVSchedulingSimulation.postprocessing(
            strategy='reasoning',
            data=schedule,
        )
        response = {
            'type': 'tool',
            'result': schedule,
            'raw': None,
            'token': {},
        }

    if response['type'] == 'text':
        response = response['result']
        sessions[session_id]['schedule_dialog_history'].append({"role": "Staff", "content": response})
    elif response['type'] == 'tool':
        pred_schedule = response['result']
        response = agent.schedule_suggestion.format(schedule=pred_schedule)
        sessions[session_id]['schedule_dialog_history'].append({"role": "Staff", "content": response})
    
    return {'response': response, 'session_id': session_id, 'task': sessions[session_id]['task']}


@app.delete('/session/{session_id}')
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail='Session not found')
    del sessions[session_id]
    log(f'Delete {session_id} session successfully!', color=True)
    return {'message': f'Session {session_id} deleted'}


@app.get('/sessions')
async def list_sessions():
    return {'active_sessions': list(sessions.keys())}


@app.post('/save_history/{session_id}')
async def save_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail='Session not found')
    
    # Save to file
    save_dir = Path('./demo/human_dialogs')
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = save_dir / f'{session_id}_{timestamp}.json'
    histories = sessions[session_id]['intake_dialog_history'] + sessions[session_id]['schedule_dialog_history'][1:]
    json_save_fast(file_path, histories)
    
    return {'message': f'History saved', 'file_path': str(file_path)}


if __name__ == '__main__':
    uvicorn.run('llm_server:app', host=args.backend_host, port=args.backend_port, reload=args.is_develop)
