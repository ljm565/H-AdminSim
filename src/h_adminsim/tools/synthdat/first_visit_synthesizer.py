from tqdm import tqdm
from typing import Tuple

from .data_synthesizer import DataSynthesizer
from h_adminsim.utils import Information, log
from h_adminsim.utils.common_utils import *
from h_adminsim.utils.filesys_utils import *



class FirstVisitDataSynthesizer(DataSynthesizer):
    def __init__(self, config):
        super().__init__(config)
        
    
    def synthesize(self,
                   return_obj: bool = False,
                   sanity_check: bool = False) -> Tuple[list[Information], list[Hospital]]:
        """
        Synthesize hospital data based on the configuration settings.

        Args:
            return_obj (bool, optional): Whether to return the hospital data object.
            sanity_check (bool, optional): If you want to check whether the generated data are compatible with the `Hospital` object,
                                 you can use this option.

        Raises:
            e: Exception if data synthesis fails.

        Returns:
            Tuple[list[Information], list[Hospital]]: A tuple containing the synthesized hospital data as an Information object and a Hospital object.
        """
        if sanity_check:
            return_obj = True

        try:
            all_data, all_hospitals = list(), list()
            hospitals = DataSynthesizer.hospital_list_generator(self.config.hospital_data.hospital_n)
            for i, hospital in tqdm(enumerate(hospitals), desc='Synthesizing data..', total=len(hospitals)):
                data = DataSynthesizer.define_hospital_info(self.config, hospital)
                hospital_obj = convert_info_to_obj(data) if return_obj else None
                if sanity_check:
                    new_data = convert_obj_to_info(hospital_obj)
                    assert to_dict(data) == to_dict(new_data)
                json_save_fast(self.data_save_dir / f'hospital_{padded_int(i, len(str(self._n)))}.json', to_dict(data))
                all_data.append(data)
                all_hospitals.append(hospital_obj)
            log(f"Total {len(hospitals)} data synthesizing completed. Path: `{self.data_save_dir}`", color=True)
            return all_data, all_hospitals
        
        except Exception as e:
            log(f"Data synthesizing failed: {e}", level='error')
            raise
