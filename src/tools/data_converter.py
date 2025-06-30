import os
from tqdm import tqdm
from typing import Optional

from utils import Information
from utils.fhir_utils import sanitize_id
from utils.filesys_utils import json_load, json_save_fast
from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    convert_time_to_segment,
    convert_segment_to_time,
)



class DataConverter:
    def __init__(self, config):
        # Initialize configuration
        self.fhir_url = config.fhir_url
        data_dir = os.path.join(config.project, config.data_name, 'data')
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('json')]
    

    @staticmethod
    def data_to_practitioner(data: dict, output_dir: Optional[str] = None) -> list[dict]:
        """
        Convert synthetic hospital data into `Practitioner` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Practitioner resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.

        Returns:
            list[dict]: A list of converted FHIR Practitioner resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'practitioner'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'practitioner')
        
        hospital_name = data.get('metadata')['hospital_name']
        practitioners = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = sanitize_id(f"{hospital_name}-{doctor_values['department']}-{doctor_name}")
            names = doctor_name.split()
            practitioner_obj = {
                'resourceType': 'Practitioner',
                'id': practitioner_id,
                'active': True,
                'name': [
                    {
                        'family': names[-1],
                        'given': [' '.join(names[1:-1])],
                        'prefix': [names[0]]
                    }
                ]
            }
            practitioners.append(practitioner_obj)

            if save_dir:
                json_save_fast(
                    os.path.join(save_dir, f'{practitioner_id}.fhir.json'),
                    practitioner_obj
                )
        
        return practitioners


    @staticmethod
    def data_to_patient(data: dict, output_dir: Optional[str] = None) -> list[dict]:
        """
        Convert synthetic hospital data into `Patient` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Patient resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.

        Returns:
            list[dict]: A list of converted FHIR Patient resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'patient'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'patient')
        
        hospital_name = data.get('metadata')['hospital_name']
        patients = list()

        for patient_name, patient_values in data['patient'].items():
            patient_id = sanitize_id(f"{hospital_name}-{patient_values['department']}-{patient_name}")
            names = patient_name.split()
            patient_obj = {
                'resourceType': 'Patient',
                'id': patient_id,
                'active': True,
                'name': [
                    {
                        'family': names[-1],
                        'given': [' '.join(names[:-1])],
                    }
                ]
            }
            patients.append(patient_obj)

            if save_dir:
                json_save_fast(
                    os.path.join(save_dir, f'{patient_id}.fhir.json'),
                    patient_obj
                )
        
        return patients


    @staticmethod
    def data_to_schedule(data: dict, output_dir: Optional[str] = None) -> list[dict]:
        """
        Convert synthetic hospital data into `Schedule` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Schedule resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.

        Returns:
            list[dict]: A list of converted FHIR Schedule resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'schedule'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'schedule')

        hospital_name = data.get('metadata')['hospital_name']
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        date = data.get('metadata').get('date', None)
        start = get_iso_time(data.get('metadata')['time']['start_hour'], date, get_utc_offset(country_code, time_zone))
        end = get_iso_time(data.get('metadata')['time']['end_hour'], date, get_utc_offset(country_code, time_zone))
        schedules = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = sanitize_id(f"{hospital_name}-{doctor_values['department']}-{doctor_name}")
            schedule_id = f'{practitioner_id}-schedule'
            schedule_obj = {
                'resourceType': 'Schedule',
                'id': schedule_id,
                'active': True,
                # 'serviceCategory': [None],    TODO: Add default value
                # 'serviceType': [None],    TODO: Add default value
                # 'specialty': [None],    TODO: Add default value
                'actor': [{'reference': f'Practitioner/{practitioner_id}'}],
                'planningHorizon': {'start': start, 'end': end}
            }
            schedules.append(schedule_obj)

            if save_dir:
                json_save_fast(
                    os.path.join(save_dir, f'{schedule_id}.fhir.json'),
                    schedule_obj
                )

        return schedules


    @staticmethod
    def data_to_slot(data: dict, output_dir: Optional[str] = None) -> list[dict]:
        """
        Convert synthetic hospital data into `Slot` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Slot resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.

        Returns:
            list[dict]: A list of converted FHIR Slot resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'slot'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'slot')

        hospital_name = data.get('metadata')['hospital_name']
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        date = data.get('metadata').get('date', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        entire_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)
        slots = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = sanitize_id(f"{hospital_name}-{doctor_values['department']}-{doctor_name}")

            # Filtering fixed schedule
            fixed_schedule = []
            for schedule in doctor_values['schedule']:
                fixed_schedule += convert_time_to_segment(start_hour, end_hour, interval_hour, schedule)

            # Appointment available time segments
            free_schedule = sorted(list(set(entire_segments) - set(fixed_schedule)))

            # Add slot as a `busy` status
            for seg in fixed_schedule:
                st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                slot_id = f'{practitioner_id}-slot{seg}'
                slot_obj = {
                    'resourceType': 'Slot',
                    'id': slot_id,
                    # 'serviceCategory': [None],    TODO: Add default value
                    # 'serviceType': [None],    TODO: Add default value
                    # 'specialty': [None],    TODO: Add default value
                    'schedule': {'reference': f'Schedule/{practitioner_id}-schedule'},
                    'status': 'busy',
                    'start': get_iso_time(st, date, utc_offset),
                    'end': get_iso_time(tr, date, utc_offset),
                }
                slots.append(slot_obj)

                if save_dir:
                    json_save_fast(
                        os.path.join(save_dir, f'{slot_id}.fhir.json'),
                        slot_obj
                    )
            
            # Add slot as a `free` status
            for seg in free_schedule:
                slot_id = f'{practitioner_id}-slot{seg}'
                st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                slot_obj = {
                    'resourceType': 'Slot',
                    'id': slot_id,
                    # 'serviceCategory': [None],    TODO: Add default value
                    # 'serviceType': [None],    TODO: Add default value
                    # 'specialty': [None],    TODO: Add default value
                    'schedule': {'reference': f'Schedule/{practitioner_id}-schedule'},
                    'status': 'free',
                    'start': get_iso_time(st, date, utc_offset),
                    'end': get_iso_time(tr, date, utc_offset),
                }
                slots.append(slot_obj)
            
                if save_dir:
                    json_save_fast(
                        os.path.join(save_dir, f'{slot_id}.fhir.json'),
                        slot_obj
                    )

        return slots


    @staticmethod
    def data_to_appointment(data: dict, output_dir: Optional[str] = None) -> list[dict]:
        """
        Convert synthetic hospital data into `Appointment` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Appointment resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.

        Returns:
            list[dict]: A list of converted FHIR Appointment resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'appointment'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'appointment')

        hospital_name = data.get('metadata')['hospital_name']
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        date = data.get('metadata').get('date', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        appointments = list()

        for patient_name, patient_values in data['patient'].items():
            doctor_name = patient_values['attending_physician']
            practitioner_id = sanitize_id(f"{hospital_name}-{patient_values['department']}-{doctor_name}")
            patient_id = sanitize_id(f"{hospital_name}-{patient_values['department']}-{patient_name}")
            participant = [
                {"actor": {"reference": f"Practitioner/{practitioner_id}", "display": doctor_name}, "status": "accepted"},
                {"actor": {"reference": f"Patient/{patient_id}", "display": patient_name}, "status": "accepted"}
            ]

            # Filtering fixed schedule
            schedule_time_range = patient_values['schedule']
            schedule_segments = convert_time_to_segment(start_hour, end_hour, interval_hour, schedule_time_range)
            appointment_id = f'{practitioner_id}-appn{schedule_segments[0]}-{schedule_segments[-1]}'
            appointment_obj = {
                'resourceType': 'Appointment',
                'id': appointment_id,
                # 'serviceCategory': [None],    TODO: Add default value
                # 'serviceType': [None],    TODO: Add default value
                # 'specialty': [None],    TODO: Add default value
                # 'note: [{'text': 'sdfsf'}]    TODO: Add notes
                'status': 'booked',
                'start': get_iso_time(schedule_time_range[0], date, utc_offset),
                'end': get_iso_time(schedule_time_range[-1], date, utc_offset),
                'slot': [{'reference': f'Slot/{practitioner_id}-slot{seg}'} for seg in schedule_segments],
                'participant': participant
            }
            appointments.append(appointment_obj)

            if save_dir:
                json_save_fast(
                    os.path.join(save_dir, f'{appointment_id}.fhir.json'),
                    appointment_obj
                )
            
        return appointments
    

    def __call__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for data_file in tqdm(self.data_files, desc='Converting..'):
            data = json_load(data_file)
            practitioners =  DataConverter.data_to_practitioner(data, output_dir)
            schedules = DataConverter.data_to_schedule(data, output_dir)
            slots = DataConverter.data_to_slot(data, output_dir)
            patients =  DataConverter.data_to_patient(data, output_dir)
            appointments =  DataConverter.data_to_appointment(data, output_dir)
        
        return Information(
            practitioners=practitioners,
            schedules=schedules,
            slots=slots,
            patients=patients,
            appointments=appointments
        )