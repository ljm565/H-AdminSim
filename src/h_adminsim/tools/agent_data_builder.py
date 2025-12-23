import os
from tqdm import tqdm
from typing import Optional
from decimal import getcontext
from importlib import resources

from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.random_utils import generate_random_symptom
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, get_files



class AgentDataBuilder:
    def __init__(self, config):
        # Initialize configuration
        data_dir = os.path.join(config.project, config.data_name, 'data')
        self.data_files = get_files(data_dir, ext='json')
        getcontext().prec = 10


    @staticmethod
    def build(data: dict, 
              save_path: Optional[str] = None, 
              symptom_file_path: Optional[str] = None) -> dict:
        """
        Build agent test data from a single hospital data entry.

        Args:
            data (dict): Dictionary containing metadata, departments, doctors, and patients.
            save_path (Optional[str], optional): If provided, the generated agent data will be saved to this path.
            symptom_file_path (str, optional): Path to the JSON file containing symptoms per department. Defaults to None.

        Returns:
            dict: A dictionary with the following keys:
                - 'metadata': Original metadata from input.
                - 'department': Department-level information.
                - 'doctor': Doctor-level information.
                - 'agent_data': List of tuples. Each tuple consists of:
                    - Ground-truth scheduling information.
                    - Agent input (symptom and constraints).
        """
        if symptom_file_path == None:
            symptom_file_path = str(resources.files("h_adminsim.assets.departments").joinpath("symptom.json"))

        agent_data = {'metadata': data['metadata'], 'department': data['department'], 'doctor': data['doctor'], 'agent_data': []}
        
        for patient, patient_values in data['patient'].items():
            doctor, department, date = patient_values['attending_physician'], patient_values['department'], patient_values['date']
            gender, telecom, birth_date, identifier, address = \
                patient_values['gender'], patient_values['telecom'], patient_values['birthDate'], patient_values['identifier'], patient_values['address']
            preference, symptom_level = patient_values['preference'], patient_values['symptom_level']
            disease = generate_random_symptom(
                department=department,
                symptom_file_path=symptom_file_path, 
                ensure_unique_department='doctor' in patient_values['preference']
            )
            gt_department = disease['department'] if isinstance(disease, dict) else [department]
            gt = {
                'patient': patient,
                'gender': gender,
                'telecom': telecom,
                'birthDate': birth_date,
                'identifier': identifier,
                'address': address,
                'department': gt_department,
                'attending_physician': doctor,
                'valid_from': date if 'date' in preference else 'N/A',
                'preference': preference,
                'symptom_level': symptom_level,
            }
            agent = {
                'patient': patient,
                'gender': gender,
                'telecom': telecom,
                'birthDate': birth_date,
                'identifier': identifier,
                'address': address,
                'constraint': {
                    'preference': preference,
                    'attending_physician': doctor,
                    'valid_from': date if 'date' in preference else 'N/A',
                    'symptom_level': symptom_level,
                    'symptom': disease,
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
                 symptom_file_path: Optional[str] = None) -> list[dict]:
        """
        Generate agent test datasets for all input data files.

        Args:
            output_dir (Optional[str], optional): Directory to save the generated agent data files.
                                                  If not provided, files are not saved.
            symptom_file_path (Optional[str], optional): Path to the symptom file used during agent construction. Defaults to None.

        Returns:
            list[dict]: A list of agent test data dictionaries, one for each processed input file.
        """
        if symptom_file_path == None:
            symptom_file_path = str(resources.files("h_adminsim.assets.departments").joinpath("symptom.json"))

        os.makedirs(output_dir, exist_ok=True)
        all_agent_data = list()
        
        for data_file in tqdm(self.data_files, desc='Generating data for agent simulation..'):
            data = json_load(data_file)
            basename, ext = os.path.splitext(os.path.basename(data_file))
            agent_data = AgentDataBuilder.build(data, os.path.join(output_dir, f"{basename}_agent{ext}"), symptom_file_path)
            all_agent_data.append(agent_data)
        
        return all_agent_data
        