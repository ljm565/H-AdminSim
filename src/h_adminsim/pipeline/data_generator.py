import os
import random
import numpy as np
from sconf import Config
from pathlib import Path
from importlib import resources
from typing import Optional, Union

from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.tools import DataSynthesizer, DataConverter, AgentDataBuilder
from h_adminsim.utils import Information, colorstr, log
from h_adminsim.utils.random_utils import random_uuid
from h_adminsim.utils.filesys_utils import get_files, json_load



class DataGenerator:
    def __init__(self,
                 care_level: str = 'primary',
                 config: Optional[Union[str, Config]] = None):
        
        # Initialize
        self.config = self.load_config(care_level, config)
        self.__env_setup(self.config)
        self.fhir_url = self.config.get('fhir_url', None)
        self.data_synthesizer = DataSynthesizer(self.config)
        self.save_dir = self.data_synthesizer._save_dir
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

        Returns:
            Config: A fully initialized Config object.

        Raises:
            TypeError: If `config` is not None, str, or Config.
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


    def __env_setup(self, config: Config) -> None:
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
              build_agent_data: bool = True) -> Information:
        """
        Build the complete information bundle for the administrative simulation pipeline.

        Args:
            sanity_check (bool, optional): Whether to perform validation checks during synthetic data generation. Defaults to True.
            convert_to_fhir (bool, optional): If True, converts synthesized data into FHIR-compliant resources and stores them 
                                              in the configured output directory. Defaults to False.
            build_agent_data (bool, optional): If True, generates additional derived data required for agent-based
                                               simulations (e.g., patient profiles, department assignments, task inputs). Defaults to True.

        Raises:
            Exception: Propagates any exception encountered during:
                - synthetic data synthesis
                - FHIR conversion
                - agent data generation

        Returns:
            Information:
                A structured container holding:
                    - `data`: the synthesized dataset
                    - `fhir_data`: list of FHIR resources (or None if disabled)
                    - `agent_data`: processed agent input data (or None if disabled)
        """
        # Data generator
        try:
            data, hospital_obj = self.data_synthesizer.synthesize(sanity_check=sanity_check)
            log(f"Data synthesis completed successfully", color=True)
        except Exception as e:
            log("Data synthesis failed.", level="error")
            raise e
        
        # FHIR conversion
        all_resource_list = None
        if convert_to_fhir:
            converter = DataConverter(self.config)
            try:
                all_resource_list = converter(self.save_dir / 'fhir_data', sanity_check)
                log(f"Data FHIR conversion completed successfully", color=True)
            except Exception as e:
                log("Data FHIR conversion failed.", level='error')
                raise e
            
        # Build data for agent simulation
        agent_data_list = None
        if build_agent_data:
            builder = AgentDataBuilder(self.config)
            try:
                agent_data_list = builder(self.save_dir / 'agent_data')
                log(f"Agent data generation completed successfully", color=True)
            except Exception as e:
                log("Agent data generation failed.", level='error')
                raise e
        
        output = Information(
            data=data,
            fhir_data=all_resource_list,
            agent_data=agent_data_list
        )
        
        return output
    

    def upload_to_fhir(self,
                       fhir_data_dir: str,
                       fhir_url: Optional[str] = None) -> None:
        """
        Upload synthesized FHIR resources to the specified FHIR server.

        Args:
            fhir_data_dir (str):
                Directory containing FHIR resource JSON files (e.g., practitioner,
                practitionerrole, schedule, slot).
            fhir_url (Optional[str], optional):
                Base URL of the FHIR server. If not provided, the instance's default
                FHIR URL is used.

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
