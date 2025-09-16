import os
from tqdm import tqdm
from typing import Optional

from utils import Information, log
from utils.fhir_utils import *
from utils.filesys_utils import json_load, json_save_fast, get_files
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
        self.data_files = get_files(data_dir, ext='json')
    

    @staticmethod
    def data_to_practitioner(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `Practitioner` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Practitioner resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Practitioner resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'practitioner'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'practitioner')
        
        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        practitioners = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[doctor_values['department']]['code'].lower(), 
                doctor_name
            )
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
                ],
                'gender': doctor_values['gender'],
                'telecom': doctor_values['telecom'],
                'birthDate': doctor_values['birthDate']
            }
            practitioners.append(practitioner_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{practitioner_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                json_save_fast(
                    save_path,
                    practitioner_obj
                )
        
        return practitioners
    

    @staticmethod
    def data_to_practitionerrole(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `PractitionerRole` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted PractitionerRole resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR PractitionerRole resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'practitionerrole'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'practitionerrole')
        
        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        practitionerroles = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[doctor_values['department']]['code'].lower(), 
                doctor_name
            )
            practitionerrole_id = get_practitionerrole_id(practitioner_id)
            practitionerrole_obj = {
                'resourceType': 'PractitionerRole',
                'id': practitionerrole_id,
                'active': True,
                'specialty': [
                    {
                        'coding': [{
                            'code': doctor_values['specialty']['code'],
                            'display': doctor_values['specialty']['name']
                        }],
                        'text': doctor_values['department']
                    }
                ],
                'characteristic': [
                    {
                        'coding': [{
                            'code': 'capacity_per_hour',
                            'display': str(doctor_values['capacity_per_hour'])
                        }],
                        'text': 'capacity_per_hour'
                    },
                    {
                        'coding': [{
                            'code': 'capacity',
                            'display': str(doctor_values['capacity'])
                        }],
                        'text': 'capacity'
                    }
                ],
                'practitioner': {'reference': f'Practitioner/{practitioner_id}'}
            }
            practitionerroles.append(practitionerrole_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{practitionerrole_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                json_save_fast(
                    save_path,
                    practitionerrole_obj
                )
        
        return practitionerroles


    @staticmethod
    def data_to_patient(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `Patient` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Patient resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Patient resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'patient'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'patient')
        
        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        patients = list()

        for patient_name, patient_values in data['patient'].items():
            patient_id = get_individual_id(
                hospital_name,
                department_data[patient_values['department']]['code'].lower(), 
                patient_name
            )
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
                ],
                'gender': patient_values['gender'],
                'telecom': patient_values['telecom'],
                'birthDate': patient_values['birthDate'],
                'identifier': patient_values['identifier'],
                'address': patient_values['address']
            }
            patients.append(patient_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{patient_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                json_save_fast(
                    save_path,
                    patient_obj
                )
        
        return patients


    @staticmethod
    def data_to_schedule(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `Schedule` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Schedule resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Schedule resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'schedule'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'schedule')

        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        start_date = data.get('metadata').get('start_date', None)
        end_date = data.get('metadata').get('end_date', start_date)
        start = get_iso_time(data.get('metadata')['time']['start_hour'], start_date, get_utc_offset(country_code, time_zone))
        end = get_iso_time(data.get('metadata')['time']['end_hour'], end_date, get_utc_offset(country_code, time_zone))
        schedules = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[doctor_values['department']]['code'].lower(), 
                doctor_name
            )
            schedule_id = get_schedule_id(practitioner_id)
            schedule_obj = {
                'resourceType': 'Schedule',
                'id': schedule_id,
                'active': True,
                'actor': [{'reference': f'Practitioner/{practitioner_id}'}],
                'planningHorizon': {'start': start, 'end': end}
            }
            schedules.append(schedule_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{schedule_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                json_save_fast(
                    save_path,
                    schedule_obj
                )

        return schedules


    @staticmethod
    def data_to_slot(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `Slot` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Slot resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Slot resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'slot'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'slot')

        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        entire_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)
        slots = list()

        for doctor_name, doctor_values in data['doctor'].items():
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[doctor_values['department']]['code'].lower(), 
                doctor_name
            )

            for date, schedules in doctor_values['schedule'].items():
                # Filtering fixed schedule
                fixed_schedule = []
                for schedule in schedules:
                    fixed_schedule += convert_time_to_segment(start_hour, end_hour, interval_hour, schedule)

                # Appointment available time segments
                free_schedule = sorted(list(set(entire_segments) - set(fixed_schedule)))

                # Add slot as a `busy` status
                for seg in fixed_schedule:
                    st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                    slot_id = get_slot_id(practitioner_id, date, seg)
                    slot_obj = {
                        'resourceType': 'Slot',
                        'id': slot_id,
                        'schedule': {'reference': f'Schedule/{get_schedule_id(practitioner_id)}'},
                        'status': 'busy',
                        'start': get_iso_time(st, date, utc_offset),
                        'end': get_iso_time(tr, date, utc_offset),
                    }
                    slots.append(slot_obj)

                    if save_dir:
                        save_path = os.path.join(save_dir, f'{slot_id}.fhir.json')
                        if sanity_check:
                            assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                        json_save_fast(
                            save_path,
                            slot_obj
                        )
                
                # Add slot as a `free` status
                for seg in free_schedule:
                    slot_id = get_slot_id(practitioner_id, date, seg)
                    st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                    slot_obj = {
                        'resourceType': 'Slot',
                        'id': slot_id,
                        'schedule': {'reference': f'Schedule/{get_schedule_id(practitioner_id)}'},
                        'status': 'free',
                        'start': get_iso_time(st, date, utc_offset),
                        'end': get_iso_time(tr, date, utc_offset),
                    }
                    slots.append(slot_obj)
                
                    if save_dir:
                        save_path = os.path.join(save_dir, f'{slot_id}.fhir.json')
                        if sanity_check:
                            assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                        json_save_fast(
                            save_path,
                            slot_obj
                        )

        return slots


    @staticmethod
    def data_to_appointment(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert synthetic hospital data into `Appointment` FHIR resources. 

        Args:
            data (dict): Synthetic hospital data containing doctor information.
            output_dir (Optional[str], optional): Directory path to save the converted Appointment resources 
                                                  as `.fhir.json` files. If None, the resources are not saved to disk.
                                                  Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Appointment resource objects.
        """
        save_dir = None
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'appointment'), exist_ok=True)
            save_dir = os.path.join(output_dir, 'appointment')

        hospital_name = data.get('metadata')['hospital_name']
        department_data = data.get('department')
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        appointments = list()

        for patient_name, patient_values in data['patient'].items():
            doctor_name = patient_values['attending_physician']
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[patient_values['department']]['code'].lower(), 
                doctor_name
            )
            patient_id = get_individual_id(
                hospital_name,
                department_data[patient_values['department']]['code'].lower(), 
                patient_name
            )
            participant = [
                {"actor": {"reference": f"Practitioner/{practitioner_id}", "display": doctor_name}, "status": "accepted"},
                {"actor": {"reference": f"Patient/{patient_id}", "display": patient_name}, "status": "accepted"}
            ]

            # Filtering fixed schedule
            date = patient_values['date']
            schedule_time_range = patient_values['schedule']
            schedule_segments = convert_time_to_segment(start_hour, end_hour, interval_hour, schedule_time_range)
            appointment_id = get_appointment_id(practitioner_id, date, schedule_segments[0], schedule_segments[-1])
            appointment_obj = {
                'resourceType': 'Appointment',
                'id': appointment_id,
                'status': 'booked',
                'start': get_iso_time(schedule_time_range[0], date, utc_offset),
                'end': get_iso_time(schedule_time_range[-1], date, utc_offset),
                'slot': [{'reference': f'Slot/{get_slot_id(practitioner_id, date, seg)}'} for seg in schedule_segments],
                'participant': participant
            }
            appointments.append(appointment_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{appointment_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), log(f"Same file exists: {save_path}", "error")
                json_save_fast(
                    save_path,
                    appointment_obj
                )
            
        return appointments
    

    def __call__(self, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[Information]:
        """
        Convert synthetic hospital data files into FHIR resources and optionally save them to disk.

        Args:
            output_dir (Optional[str], optional): Directory to save the converted FHIR resources as `.fhir.json` files.
                                                  If None, the resources will not be saved. Defaults to None.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[Information]: An object containing the converted FHIR resources, including practitioners, schedules, slots, patients, and appointments.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_resources = list()
        
        for data_file in tqdm(self.data_files, desc='Converting..'):
            data = json_load(data_file)
            practitioners = DataConverter.data_to_practitioner(data, output_dir, sanity_check)
            practitionerroles = DataConverter.data_to_practitionerrole(data, output_dir, sanity_check)
            schedules = DataConverter.data_to_schedule(data, output_dir, sanity_check)
            slots = DataConverter.data_to_slot(data, output_dir, sanity_check)
            patients = DataConverter.data_to_patient(data, output_dir, sanity_check)
            appointments = DataConverter.data_to_appointment(data, output_dir, sanity_check)

            information = Information(
                practitioners=practitioners,
                practitionerrole=practitionerroles,
                schedules=schedules,
                slots=slots,
                patients=patients,
                appointments=appointments
            )
            all_resources.append(information)
        
        return all_resources