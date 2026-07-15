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
                 task: Union[str, list[str]] = ['first_visit'],
                 care_level: str = 'primary',
                 config: Optional[Union[str, Config]] = None):
        """
        Initialize the data generator.
        If `config` is provided, `care_level` is ignored and the config is used as-is.
        Otherwise, a built-in config is loaded based on `care_level`.

        Args:
            task (Union[str, list[str]], optional): Task(s) to synthesize. Defaults to ['first_visit'].
            care_level (str, optional): Care level preset used when `config` is None. One of 'primary', 'secondary', 'tertiary'. Defaults to 'primary'.
            config (Optional[Union[str, Config]], optional): A file path or Config instance. Defaults to None.

        Raises:
            ValueError: If `care_level` is invalid.
            TypeError: If `task` is not a str, list, or tuple.
        """

        # Initialize variable conditions
        ## Configuration type
        if config is not None:
            if care_level != 'primary':     # Non-default care_level is silently overridden by config
                log(f'care_level={care_level} is ignored because config is provided.', level='warning')
            self._care_level = None
        else:
            valid_levels = ('primary', 'secondary', 'tertiary')
            if care_level not in valid_levels:
                raise ValueError(colorstr("red", f"Invalid care_level: '{care_level}'. Expected one of: {valid_levels}."))
            self._care_level = care_level

        ## Task type
        if isinstance(task, str):
            task = [task]
        elif not isinstance(task, list):
            raise TypeError(colorstr("red", f'Invalid task type: {type(task).__name__}'))    

        # Initialize necessary information
        self.config = self.load_config(config)
        self.task = self.__task_builder(task)
        self.__env_setup(self.config)
        self.fhir_url = self.config.get('fhir_url', None)
        log(f'Data saving directory: {colorstr(self.task.save_dir)}')

        
    def load_config(self, config: Optional[Union[str, Config]] = None) -> Config:
        """
        Load a configuration object.
        If `config` is None, a default configuration is loaded based on the given `self._care_level`. 
        If `config` is a string, it is treated as a file path and loaded as a Config object. 
        If a Config instance is provided, it is returned as-is.

        Args:
            config (Optional[Union[str, Config]]): A file path or Config instance. Defaults to None.

        Raises:
            TypeError: If `config` is not None, str, or Config.

        Returns:
            Config: A fully initialized Config object.
        """
        # Case 1: config is None -> load built-in config based on care_level
        if config is None:
            log(f"No config provided; using default {self._care_level} config.", "warning")
            default_path = str(resources.files("h_adminsim.assets.configs").joinpath(f"data4{self._care_level}.yaml"))
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
            colorstr("red", f"Invalid config: expected None, str, or Config, got {type(config).__name__}")
        )
    

    def __task_builder(self, task: list[str]) -> Information:
        """
        Build the synthesizers for the requested tasks.
        `follow_up_visit` cannot run standalone, so `first_visit` is added automatically when missing.

        Args:
            task (list[str]): Task list for synthesizing data. Valid values: 'first_visit', 'follow_up_visit'.

        Raises:
            ValueError: If `task` is empty or contains unknown values.

        Returns:
            Information: Container holding the requested `tasks` set, `save_dir`, and the
                eagerly-built `fv_synthesizer`. The follow-up synthesizer is built later in `build()`.
        """
        task, valid_tasks = list(task), {'first_visit', 'follow_up_visit'}
        unknown = set(task) - valid_tasks

        # Check task validity
        if not task or unknown:
            raise ValueError(colorstr("red", f"Invalid task(s): {unknown}. Expected {valid_tasks}."))
        
        # `follow_up_visit` cannot run standalone; it requires first_visit data
        if 'follow_up_visit' in task and 'first_visit' not in task:
            log("'follow_up_visit' task requires 'first_visit' task. Adding 'first_visit' to task set.", level="warning")
            task.append('first_visit')

        # Initialize task
        _task = Information()
        _task.update(tasks=set(task))
        if 'first_visit' in task:
            fv_synthesizer = FirstVisitDataSynthesizer(self.config)
            _task.update(
                save_dir=fv_synthesizer.save_dir,
                fv_synthesizer=fv_synthesizer,
            )
        # `fu_synthesizer` reads first-visit output at construction time, so it is
        # built lazily in `build()` once first-visit data has been written to disk.

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
        if 'first_visit' in self.task.tasks:
            try:
                data = self.task.fv_synthesizer.synthesize(department_info_path)
                log(f"Data synthesis completed successfully", color=True)
            except Exception:
                log("Data synthesis failed.", level="error")
                raise

        # Follow-up visit data synthesis
        if 'follow_up_visit' in self.task.tasks:
            assert hasattr(self.config.hospital_data, 'follow_up_visit'), \
                colorstr("red", "Config must contain a 'hospital_data.follow_up_visit' section for follow-up synthesis.")

            try:
                # First-visit data already written to disk -> merge follow-up into existing files
                fu_synthesizer = FollowUpDataSynthesizer(self.config, str(self.task.save_dir / 'data'))
                data = fu_synthesizer.synthesize(
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
                    self.task.save_dir / 'fhir_data', 
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
                    self.task.save_dir / 'agent_data',
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
        assert fhir_url != None, colorstr("red", 'Please double check the FHIR URL')
        
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
