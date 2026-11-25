import os
from tqdm import tqdm
from copy import deepcopy
from decimal import getcontext
from importlib import resources
from typing import Optional, TYPE_CHECKING

from h_adminsim.utils.random_utils import generate_random_symptom
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, get_files

if TYPE_CHECKING:
    from sconf import Config



class AgentDataBuilder:
    def __init__(self, config: 'Config'):
        """
        Collect the synthesized hospital data files under ``<project>/<data_name>/data``.

        Args:
            config (Config): Data synthesis config (uses ``project`` and ``data_name``).
        """
        # Initialize configuration
        data_dir = os.path.join(config.project, config.data_name, 'data')
        self.data_files = get_files(data_dir, ext='json')
        getcontext().prec = 10


    @staticmethod
    def build(data: dict, 
              save_path: Optional[str] = None, 
              symptom_file_path: Optional[str] = None) -> dict:
        """
        Build agent simulation data from one synthesized hospital data entry.

        Args:
            data (dict): Hospital data with ``metadata``, ``department``, ``doctor``, ``patient`` (and optional ``test``).
            save_path (Optional[str], optional): Where to save the result as JSON; not saved if falsy. Defaults to None.
            symptom_file_path (Optional[str], optional): Per-department symptom JSON; the packaged ``symptom.json`` if None. Defaults to None.

        Returns:
            dict: ``metadata``, ``department``, ``doctor``, ``test`` and ``agent_data`` (list of ``(gt, agent input)`` tuples).
        """
        if symptom_file_path == None:
            symptom_file_path = str(resources.files("h_adminsim.assets.departments").joinpath("symptom.json"))

        agent_data = {
            'metadata': data['metadata'], 
            'department': data['department'], 
            'doctor': data['doctor'], 
            'test': data.get('test'),
            'agent_data': []
        }
        
        for patient, patient_data in data['patient'].items():
            for appn in patient_data:
                # Basic information
                visit_type = appn['visit_type']
                doctor, department, date = appn['attending_physician'], appn['department'], appn['date']
                gender, telecom, birth_date, identifier, address, occupation, distance = \
                    appn['gender'], appn['telecom'], appn['birthDate'], appn['identifier'], appn['address'], appn['occupation'], appn['distance']
                preference = appn['preference']
                
                # Make disease-symptom pair
                disease = generate_random_symptom(
                    department=department,
                    symptom_file_path=symptom_file_path, 
                    ensure_unique_department='doctor' in appn['preference']
                )
                
                # Branch depends on the visit type
                if isinstance(disease, dict) and visit_type == 'first_visit':
                    gt_department = disease['department'] if isinstance(disease, dict) else [department]
                    symptom_level = appn['symptom_level']
                    unavailable, required_tests = None, None
                else:
                    gt_department = [department]    # For follow up visit, the patient must have only one department
                    unavailable, required_tests = appn['unavailable'], deepcopy(appn['required_tests'])
                    symptom_level = None
                    for _test in required_tests:
                        del _test['schedule']
                        del _test['date']
                        del _test['device_name']
                
                gt = {
                    'visit_type': visit_type,
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
                    'unavailable': unavailable,
                    'symptom_level': symptom_level,
                    'required_tests': required_tests,
                    'occupation': occupation,
                    'distance': distance,
                }
                agent = {
                    'visit_type': visit_type,
                    'patient': patient,
                    'gender': gender,
                    'telecom': telecom,
                    'birthDate': birth_date,
                    'identifier': identifier,
                    'address': address,
                    'occupation': occupation,
                    'distance': distance,
                    'constraint': {
                        'preference': preference,
                        'unavailable': unavailable,
                        'attending_physician': doctor,
                        'valid_from': date if 'date' in preference else 'N/A',
                        'symptom_level': symptom_level,
                        'symptom': disease,
                        'required_tests': required_tests,
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
        Build and save agent simulation data for every input data file.

        Args:
            output_dir (Optional[str], optional): Directory for the ``<name>_agent.json`` files; must be set (None fails in ``os.makedirs``). Defaults to None.
            symptom_file_path (Optional[str], optional): Per-department symptom JSON; the packaged ``symptom.json`` if None. Defaults to None.

        Returns:
            list[dict]: The agent data of each input file, in order.
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
        