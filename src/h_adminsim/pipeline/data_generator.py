import os
import random
import numpy as np
from sconf import Config
from pathlib import Path
from importlib import resources
from typing import Optional, Union

from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.tools import DataConverter, AgentDataBuilder
from h_adminsim.tools.synthdat import FirstVisitDataSynthesizer, FollowUpDataSynthesizer
from h_adminsim.utils import Information, colorstr, log
from h_adminsim.utils.random_utils import random_uuid
from h_adminsim.utils.filesys_utils import get_files, json_load



class DataGenerator:
    def __init__(self,
                 task: Optional[list[str]] = None,
                 care_level: str = 'primary',
                 config: Optional[Union[str, Config]] = None):

        # Initialize
        self.config = self.load_config(care_level, config)
        self.task = self.__init_task(task or ['first_visit'])
        self.__env_setup(self.config)
        self.fhir_url = self.config.get('fhir_url', None)
        if 'first_visit' in self.task:
            self.fv_synthesizer = FirstVisitDataSynthesizer(self.config)
            self.save_dir = self.fv_synthesizer.save_dir
            log(f'Data saving directory: {colorstr(self.save_dir)}')

        
    def load_config(self, care_level: str, config: Optional[Union[str, Config]]) -> Config:
        """
        Load a configuration object.

        If `config` is None, a default configuration is loaded based on the given
        `care_level`. If `config` is a string, it is treated as a file path and
        loaded as a Config object. If a Config instance is provided, it is returned
        as-is.

        Args:
            care_level (str): Care level used to select the default config.
            config (Optional[Union[str, Config]]): A file path or Config instance.

        Raises:
            TypeError: If `config` is not None, str, or Config.

        Returns:
            Config: A fully initialized Config object.
        """
        # Case 1: config is None -> load built-in config based on care_level
        if config is None:
            log(f"No config provided; using default {care_level} config.", "warning")
            assert care_level in ['primary', 'secondary', 'tertiary'], \
                log(f"Invalid care_level: '{care_level}'. Expected one of: primary, secondary, tertiary.", "error")
            default_path = str(resources.files("h_adminsim.assets.configs").joinpath(f"data4{care_level}.yaml"))
            return Config(default_path)

        # Case 2: config is a string path
        if isinstance(config, str):
            config_inst = Config(config)
            return config_inst

        # Case 3: config is already a Config object
        if isinstance(config, Config):
            return config

        # Otherwise error
        raise TypeError(
            log(f"Invalid config: expected None, str, or Config, got {type(config).__name__}", "error")
        )
    

    def __init_task(self, task: list[str]) -> set:
        """
        Initialize task set.

        Args:
            task (list[str]): Task list for synthesizing data.

        Returns:
            set: Initialized task set.
        """
        _task = set()
        if 'first_visit' in task:
            _task.add('first_visit')
        if 'follow_up_visit' in task:
            _task.add('follow_up_visit')
        
        if 'follow_up_visit' in _task and 'first_visit' not in _task:
            log("'follow_up_visit' task requires 'first_visit' task. Adding 'first_visit' to task set.", level="warning")
            _task.add('first_visit')
        
        return _task


    def __env_setup(self, config: Config):
        """
        Initialize environment-level random seeds using the given configuration.

        Args:
            config (Config): Configuration containing the seed value.
        """
        random.seed(config.seed)
        np.random.seed(config.seed)
    

    def build(self,
              sanity_check: bool = True,
              convert_to_fhir: bool = False,
              build_agent_data: bool = True,
              source_data_dir: Optional[str] = None,
              department_info_path: Optional[str] = None,
              symptom_file_path: Optional[str] = None) -> Information:
        """
        Build the complete data pipeline based on the configured task set.

        Dispatches to first-visit and/or follow-up data synthesis depending on
        which tasks are in `self.task`. The execution order is always:
        first_visit -> follow_up_visit (follow-up requires existing first-visit data).

        Args:
            sanity_check (bool, optional): Whether to perform validation checks. Defaults to True.
            convert_to_fhir (bool, optional): If True, converts data into FHIR resources. Defaults to False.
            build_agent_data (bool, optional): If True, generates agent simulation data. Defaults to True.
            source_data_dir (Optional[str], optional): Path to existing hospital data for follow-up synthesis.
                If None, uses `self.save_dir / 'data'`. Defaults to None.
            department_info_path (Optional[str], optional): Path to a file containing department information. If provided, it will be used to load names. 
                                                            Defaults to None.
            symptom_file_path (Optional[str], optional): Path to the symptom file used during agent construction. Defaults to None.

        Returns:
            Information:
                A structured container holding:
                    - `data`: first-visit synthesized dataset (or None)
                    - `fhir_data`: list of FHIR resources (or None)
                    - `agent_data`: processed agent input data (or None)
                    - `followup_data`: follow-up merged data (or None)
        """
        data, all_resource_list, agent_data_list = None, None, None

        # First-visit data synthesis
        if 'first_visit' in self.task:
            try:
                data = self.fv_synthesizer.synthesize(department_info_path)
                log(f"Data synthesis completed successfully", color=True)
            except Exception:
                log("Data synthesis failed.", level="error")
                raise

        # Follow-up visit data synthesis
        if 'follow_up_visit' in self.task:
            assert hasattr(self.config.hospital_data, 'follow_up_visit'), \
                log("Config must contain a 'hospital_data.follow_up_visit' section for follow-up synthesis.", "error")

            try:
                # First-visit data already generated -> merge follow-up into existing files
                if source_data_dir is None:
                    source_data_dir = str(self.save_dir / 'data')
                followup_synthesizer = FollowUpDataSynthesizer(self.config, source_data_dir)
                data = followup_synthesizer.synthesize(
                    sanity_check=sanity_check,
                    department_info_path=department_info_path,
                )
                log(f"Follow-up data synthesis completed successfully", color=True)
            except Exception:
                log("Follow-up data synthesis failed.", level="error")
                raise

        # FHIR conversion
        if convert_to_fhir:
            converter = DataConverter(self.config)
            try:
                all_resource_list = converter(
                    self.save_dir / 'fhir_data', 
                    sanity_check
                )
                log(f"Data FHIR conversion completed successfully", color=True)
            except Exception:
                log("Data FHIR conversion failed.", level='error')
                raise

        # Build data for agent simulation
        if build_agent_data:
            builder = AgentDataBuilder(self.config)
            try:
                agent_data_list = builder(
                    self.save_dir / 'agent_data',
                    symptom_file_path,
                )
                log(f"Agent data generation completed successfully", color=True)
            except Exception:
                log("Agent data generation failed.", level='error')
                raise

        output = Information(
            data=data,
            fhir_data=all_resource_list,
            agent_data=agent_data_list,
        )

        return output


    def upload_to_fhir(self,
                       fhir_data_dir: str,
                       fhir_url: Optional[str] = None):
        """
        Upload synthesized FHIR resources to the specified FHIR server.

        Args:
            fhir_data_dir (str): Directory containing FHIR resource JSON files (e.g., practitioner, practitionerrole, schedule, slot).
            fhir_url (Optional[str], optional): Base URL of the FHIR server. If not provided, the instance's default FHIR URL is used.
        """
        # Initialize FHIR URL and manager
        if not fhir_url:
            fhir_url = self.fhir_url
        assert fhir_url != None, log('')
        
        if not fhir_url.endswith('fhir'):
            fhir_url = os.path.join(fhir_url, 'fhir')

        fhir_manager = FHIRManager(fhir_url)

        # FHIR resources
        fhir_data_dir = Path(fhir_data_dir)
        fhir_resources_dirs = [fhir_data_dir / resource for resource in ['practitioner', 'practitionerrole', 'schedule', 'slot']]

        # Upload resources to FHIR
        for path in fhir_resources_dirs:
            files = get_files(path, ext='json')
            error_files = list()

            for file in files:
                resource_data = json_load(file)
                resource_type = resource_data.get('resourceType')
                if 'id' not in resource_data:
                    resource_data['id'] = random_uuid(False)
                
                response = fhir_manager.create(resource_type, resource_data)
                if 200 <= response.status_code < 300:
                    log(f"Created {resource_type} with ID {response.json().get('id')}")
                else:
                    error_files.append(file)
            
            if len(error_files):
                log(f'Error files during creating data: {error_files}', 'warning')
