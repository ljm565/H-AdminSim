import os
from tqdm import tqdm
from typing import Optional
from decimal import Decimal, getcontext

from utils import Information
from utils.fhir_utils import *
from utils.random_utils import generate_random_symptom
from utils.filesys_utils import json_load, json_save_fast, get_files
from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    convert_time_to_segment,
    convert_segment_to_time,
)



class AgentTestBuilder:
    def __init__(self, config):
        # Initialize configuration
        data_dir = os.path.join(config.project, config.data_name, 'data')
        self.data_files = get_files(data_dir, ext='json')
        getcontext().prec = 10


    @staticmethod
    def build(data: dict, 
              save_path: Optional[str] = None, 
              symptom_file_path: str = './asset/departments/symptom.json') -> dict:
        """
        Build agent test data from a single hospital data entry.

        Args:
            data (dict): Dictionary containing metadata, departments, doctors, and patients.
            save_path (Optional[str], optional): If provided, the generated agent data will be saved to this path.
            symptom_file_path (str, optional): Path to the JSON file containing symptoms per department.
                                               Defaults to './asset/departments/symptom.json'.

        Returns:
            dict: A dictionary with the following keys:
                - 'metadata': Original metadata from input.
                - 'department': Department-level information.
                - 'doctor': Doctor-level information.
                - 'agent_data': List of tuples. Each tuple consists of:
                    - Ground-truth scheduling information.
                    - Agent input (symptom and constraints).
        """
        hospital_name = data.get('metadata')['hospital_name']
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        date = data.get('metadata').get('date', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        agent_data = {'metadata': data['metadata'], 'department': data['department'], 'doctor': data['doctor'], 'agent_data': []}
        
        for patient, patient_values in data['patient'].items():
            doctor = patient_values['attending_physician']
            department = patient_values['department']
            schedule_time_range = patient_values['schedule']
            schedule_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour, schedule_time_range)
            appointment_id = get_appointment_id(
                get_individual_id(hospital_name, department, doctor),
                schedule_time_segments[0],
                schedule_time_segments[-1]
            )
            gt = {
                'patient': patient,
                'gender': patient_values['gender'],
                'telecom': patient_values['telecom'],
                'birthDate': patient_values['birthDate'],
                'department': department,
                'examination': data['doctor'][doctor]['specialty'],
                'attending_physician': doctor,
                'schedule': {
                    'time': schedule_time_range, 
                    'segment': schedule_time_segments,
                    'iso': {'start': get_iso_time(schedule_time_range[0], date, utc_offset), 'end': get_iso_time(schedule_time_range[1], date, utc_offset)}
                },
                'priority': patient_values['priority'],
                'flexibility': patient_values['flexibility'],
                'fhir_resource': f'{appointment_id}.fhir.json'
            }
            agent = {
                'patient': patient,
                'gender': patient_values['gender'],
                'telecom': patient_values['telecom'],
                'birthDate': patient_values['birthDate'],
                'symptom': generate_random_symptom(department, symptom_file_path),
                'constraint': {
                    'duration': float(Decimal(str(schedule_time_range[-1])) - Decimal(str(schedule_time_range[0]))),
                    'priority': patient_values['priority'],
                    'flexibility': patient_values['flexibility'],
                    # 'final_time_to_leave': {  # TODO: Finalize this field
                    #     'time': schedule_time_range[-1],
                    #     'segment': [schedule_time_segments[-1]],
                    #     'iso': get_iso_time(schedule_time_range[-1], date, utc_offset)
                    # }
                }
            }
            agent_data['agent_data'].append((gt, agent))
        
        if save_path:
            json_save_fast(
                save_path,
                agent_data
            )
        
        return agent_data
            

    def __call__(self,
                 output_dir: Optional[str] = None, 
                 symptom_file_path: str = './asset/departments/symptom.json') -> list[dict]:
        """
        Generate agent test datasets for all input data files.

        Args:
            output_dir (Optional[str], optional): Directory to save the generated agent data files.
                                                  If not provided, files are not saved.
            symptom_file_path (str, optional): Path to the symptom file used during agent construction.
                                               Defaults to './asset/departments/symptom.json'.

        Returns:
            list[dict]: A list of agent test data dictionaries, one for each processed input file.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_agent_data = list()
        
        for data_file in tqdm(self.data_files, desc='Generating..'):
            data = json_load(data_file)
            basename, ext = os.path.splitext(os.path.basename(data_file))
            agent_data = AgentTestBuilder.build(data, os.path.join(output_dir, f"{basename}_agent{ext}"), symptom_file_path)
            all_agent_data.append(agent_data)
        
        return all_agent_data
        