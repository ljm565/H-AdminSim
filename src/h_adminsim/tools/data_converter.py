import os
from tqdm import tqdm
from typing import Optional

from h_adminsim.utils import Information, colorstr, log
from h_adminsim.utils.fhir_utils import *
from h_adminsim.registry.variables import PRIORITY_MAP
from h_adminsim.utils.filesys_utils import json_load, json_save_fast, get_files
from h_adminsim.utils.common_utils import (
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
                    assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
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
                    assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
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

        for patient_name, patient_data in data['patient'].items():
            patient_value = patient_data[0]
            patient_id = get_individual_id(
                hospital_name,
                department_data[patient_value['department']]['code'].lower(), 
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
                'gender': patient_value['gender'],
                'telecom': patient_value['telecom'],
                'birthDate': patient_value['birthDate'],
                'identifier': patient_value['identifier'],
                'address': patient_value['address']
            }
            patients.append(patient_obj)

            if save_dir:
                save_path = os.path.join(save_dir, f'{patient_id}.fhir.json')
                if sanity_check:
                    assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                json_save_fast(
                    save_path,
                    patient_obj
                )
        
        return patients
    

    @staticmethod
    def data_to_device(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert hospital test data into `Device` FHIR resources.

        Args:
            data (dict): Synthetic hospital data containing test information.
            output_dir (Optional[str], optional): Directory path to save the converted Device
                                                  resources as `.fhir.json` files. If None, the resources
                                                  are not saved to disk. Defaults to None.
            sanity_check (bool, optional): If True, asserts that no duplicate files exist when saving.
                                           Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR Device resource objects.
        """
        devices = list()
        test_data = data.get('test')
        
        # Activate only when test data exists, as Device resources are only needed for tests.
        if test_data:
            save_dir = None
            if output_dir:
                os.makedirs(os.path.join(output_dir, 'device'), exist_ok=True)
                save_dir = os.path.join(output_dir, 'device')
            
            hospital_name = data.get('metadata')['hospital_name']

            for tests in test_data.values():
                for info in tests:
                    for device_name in info['devices'].keys():
                        device_id = get_device_id(hospital_name, device_name)
                        device_obj = {
                            'resourceType': 'Device',
                            'id': device_id,
                            'status': 'active',
                            'displayName': info['name'],
                            'type': [
                                {
                                    'coding': [{
                                        'code': info['code'], 
                                        'display': info['name']
                                    }],
                                    'text': info['name']
                                }
                            ],
                            'property': [
                                {
                                    'type': {'text': 'duration'},
                                    'valueQuantity': {'value': info['duration_hour'], 'unit': 'h'}
                                },
                                {
                                    'type': {'text': 'priority'},
                                    'valueString': PRIORITY_MAP['priority_to_code'][info['priority']]
                                }
                            ]
                        }
                        devices.append(device_obj)

                        if save_dir:
                            save_path = os.path.join(save_dir, f'{device_id}.fhir.json')
                            if sanity_check:
                                assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                            json_save_fast(save_path, device_obj)

        return devices


    @staticmethod
    def data_to_healthcareservice(data: dict, output_dir: Optional[str] = None, sanity_check: bool = False) -> list[dict]:
        """
        Convert hospital test data into `HealthcareService` FHIR resources (one per test type).

        Each HealthcareService represents a logical test type and references the physical Device
        resources that can perform it (one-to-many via extensions).

        Args:
            data (dict): Synthetic hospital data containing test information.
            output_dir (Optional[str], optional): Directory path to save the converted HealthcareService
                                                  resources as `.fhir.json` files. If None, the resources
                                                  are not saved to disk. Defaults to None.
            sanity_check (bool, optional): If True, asserts that no duplicate files exist when saving.
                                           Defaults to False.

        Returns:
            list[dict]: A list of converted FHIR HealthcareService resource objects.
        """
        healthcareservices = list()
        test_data = data.get('test')

        # Activate only when test data exists, as HealthcareService resources are only needed for tests.
        if test_data:
            save_dir = None
            if output_dir:
                os.makedirs(os.path.join(output_dir, 'healthcareservice'), exist_ok=True)
                save_dir = os.path.join(output_dir, 'healthcareservice')

            hospital_name = data.get('metadata')['hospital_name']
            code_to_test_name = {test['code']: test['name'] for _, tests in data.get('test', {}).items() for test in tests}

            for tests in test_data.values():
                for info in tests:
                    healthcareservice_id = get_healthcareservice_id(hospital_name, info['code'])
                    eligibility = [
                        {
                            "code": {
                                "coding": [{"code": dep_code, "display": 'required'}],
                                "text": code_to_test_name[dep_code]
                            },
                            "comment": f"{dep_code} required"
                        } for dep_code in info['depends_on'] if dep_code in code_to_test_name
                    ] + [
                        {
                            "code": {
                                "coding": [{"code": avd_code, "display": 'avoid'}],
                                "text": code_to_test_name[avd_code]
                            },
                            "comment": f"Avoid {avd_code} in the same day"
                        } for avd_code in info['avoid_same_day'] if avd_code in code_to_test_name
                    ]
                    healthcareservice_obj = {
                        'resourceType': 'HealthcareService',
                        'id': healthcareservice_id,
                        'active': True,
                        'type': [
                            {
                                'coding': [{'code': info['code'], 'display': info['name']}],
                                'text': info['name']
                            }
                        ],
                        'name': info['name'],
                        'comment': info['description'],
                        'characteristic': [
                            {
                                'coding': [{'code': 'duration_hour', 'display': str(info['duration_hour'])}],
                                'text': 'duration_hour'
                            },
                            {
                                'coding': [{'code': 'priority', 'display': PRIORITY_MAP['priority_to_code'][info['priority']]}],
                                'text': 'priority'
                            },
                            {
                                'coding': [{'code': 'result_hours', 'display': str(info['result_hours'])}],
                                'text': 'result_hours'
                            }
                        ],
                        'eligibility': eligibility,
                        'serviceProvisionCode': [
                            {
                                'coding': [{'code': get_device_id(hospital_name, device_name), 'display': device_name}],
                                'text': device_name
                            }
                            for device_name in info['devices'].keys()
                        ],
                        'appointmentRequired': True,
                    }
                    healthcareservices.append(healthcareservice_obj)

                    if save_dir:
                        save_path = os.path.join(save_dir, f'{healthcareservice_id}.fhir.json')
                        if sanity_check:
                            assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                        json_save_fast(save_path, healthcareservice_obj)

        return healthcareservices


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
        test_data = data.get('test')
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        start_date = data.get('metadata').get('start_date', None)
        end_date = data.get('metadata').get('end_date', start_date)
        start = get_iso_time(data.get('metadata')['time']['start_hour'], start_date, get_utc_offset(country_code, time_zone))
        end = get_iso_time(data.get('metadata')['time']['end_hour'], end_date, get_utc_offset(country_code, time_zone))
        schedules = list()

        # Physician information
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
                    assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                json_save_fast(
                    save_path,
                    schedule_obj
                )
        
        # Test information: Activate only when test data exists.
        if test_data:
            for tests in test_data.values():
                for info in tests:
                    for device_name in info['devices'].keys():
                        device_id = get_device_id(hospital_name, device_name)
                        schedule_id = get_schedule_id(device_id)
                        schedule_obj = {
                            'resourceType': 'Schedule',
                            'id': schedule_id,
                            'active': True,
                            'actor': [{'reference': f'Device/{device_id}'}],
                            'planningHorizon': {'start': start, 'end': end}
                        }
                        schedules.append(schedule_obj)

                        if save_dir:
                            save_path = os.path.join(save_dir, f'{schedule_id}.fhir.json')
                            if sanity_check:
                                assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
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
        test_data = data.get('test')
        country_code = data.get('metadata').get('country_code', 'KR')
        time_zone = data.get('metadata').get('timezone', None)
        utc_offset = get_utc_offset(country_code, time_zone)
        start_hour = data.get('metadata')['time']['start_hour']
        end_hour = data.get('metadata')['time']['end_hour']
        interval_hour = data.get('metadata')['time']['interval_hour']
        entire_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)
        slots = list()

        # Physician fixed schedule
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
                            assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
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
                            assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                        json_save_fast(
                            save_path,
                            slot_obj
                        )

        # Test fixed schedule: Activate only when test data exists
        if test_data:
            for tests in test_data.values():
                for info in tests:
                    for device_name, device_info in info['devices'].items():
                        device_id = get_device_id(hospital_name, device_name)
                        for date, schedules in device_info['schedule'].items():
                            # Filtering fixed schedule
                            fixed_schedule = []
                            for schedule in schedules:
                                fixed_schedule += convert_time_to_segment(start_hour, end_hour, interval_hour, schedule)

                            # Appointment available time segments
                            free_schedule = sorted(list(set(entire_segments) - set(fixed_schedule)))

                            # Add slot as a `busy` status
                            for seg in fixed_schedule:
                                st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                                slot_id = get_slot_id(device_id, date, seg)
                                slot_obj = {
                                    'resourceType': 'Slot',
                                    'id': slot_id,
                                    'schedule': {'reference': f'Schedule/{get_schedule_id(device_id)}'},
                                    'status': 'busy',
                                    'start': get_iso_time(st, date, utc_offset),
                                    'end': get_iso_time(tr, date, utc_offset),
                                }
                                slots.append(slot_obj)

                                if save_dir:
                                    save_path = os.path.join(save_dir, f'{slot_id}.fhir.json')
                                    if sanity_check:
                                        assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                                    json_save_fast(
                                        save_path,
                                        slot_obj
                                    )
                            
                            # Add slot as a `free` status
                            for seg in free_schedule:
                                slot_id = get_slot_id(device_id, date, seg)
                                st, tr = convert_segment_to_time(start_hour, end_hour, interval_hour, [seg])
                                slot_obj = {
                                    'resourceType': 'Slot',
                                    'id': slot_id,
                                    'schedule': {'reference': f'Schedule/{get_schedule_id(device_id)}'},
                                    'status': 'free',
                                    'start': get_iso_time(st, date, utc_offset),
                                    'end': get_iso_time(tr, date, utc_offset),
                                }
                                slots.append(slot_obj)
                            
                                if save_dir:
                                    save_path = os.path.join(save_dir, f'{slot_id}.fhir.json')
                                    if sanity_check:
                                        assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
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
        def _emit_consultation(department: str, 
                               doctor_name: str, 
                               patient_id: str, patient_name: str, 
                               date: str, 
                               schedule_time_range: list[float]):
            """
            Build a physician-consultation `Appointment` resource (Practitioner + Patient) and append it to the enclosing `appointments` list.

            Args:
                department (str): Department key used to look up the department code in `department_data`. Drives the practitioner ID generation.
                doctor_name (str): Attending physician's full name; also used as the participant display label.
                patient_id (str): Pre-resolved patient FHIR ID (computed by the caller so the helper does not need `department_data` knowledge beyond the doctor side).
                patient_name (str): Patient's full name, used as the participant display label.
                date (str): Appointment date (ISO `YYYY-MM-DD`). Must be a concrete date; callers should skip emission when the consultation is not yet scheduled.
                schedule_time_range (list[float]): Start/end hour pair describing the consultation window (e.g. `[9.0, 9.5]`).
                                                   Discretized into segments via the enclosing `interval_hour`.
            """
            practitioner_id = get_individual_id(
                hospital_name,
                department_data[department]['code'].lower(),
                doctor_name
            )
            participant = [
                {"actor": {"reference": f"Practitioner/{practitioner_id}", "display": doctor_name}, "status": "accepted"},
                {"actor": {"reference": f"Patient/{patient_id}", "display": patient_name}, "status": "accepted"}
            ]
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
                    assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                json_save_fast(save_path, appointment_obj)

        # Initialize necessary things
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

        for patient_name, patient_data in data['patient'].items():
            for appn in patient_data:
                patient_id = get_individual_id(
                    hospital_name,
                    department_data[appn['department']]['code'].lower(),
                    patient_name
                )

                # First visit: only appointment with physician
                if appn['visit_type'] == 'first_visit':
                    _emit_consultation(
                        appn['department'],
                        appn['attending_physician'],
                        patient_id,
                        patient_name,
                        appn['date'],
                        appn['schedule']
                    )

                # Follow-up visit: appointment with physician + tests
                elif appn['visit_type'] == 'follow_up_visit':
                    # Tests: emit booked when a concrete date is present
                    required_tests = appn.get('required_tests', [])
                    for test in required_tests:
                        device_id = get_device_id(hospital_name, test['device_name'])
                        participant = [
                            {"actor": {"reference": f"Device/{device_id}", "display": test['device_name']}, "status": "accepted"},
                            {"actor": {"reference": f"Patient/{patient_id}", "display": patient_name}, "status": "accepted"}
                        ]
                        date = test['date']
                        schedule_time_range = test['schedule']
                        schedule_segments = convert_time_to_segment(start_hour, end_hour, interval_hour, schedule_time_range)
                        appointment_id = get_appointment_id(device_id, date, schedule_segments[0], schedule_segments[-1])
                        appointment_obj = {
                            'resourceType': 'Appointment',
                            'id': appointment_id,
                            'status': 'booked',
                            'start': get_iso_time(schedule_time_range[0], date, utc_offset),
                            'end': get_iso_time(schedule_time_range[-1], date, utc_offset),
                            'slot': [{'reference': f'Slot/{get_slot_id(device_id, date, seg)}'} for seg in schedule_segments],
                            'participant': participant
                        }
                        appointments.append(appointment_obj)

                        if save_dir:
                            save_path = os.path.join(save_dir, f'{appointment_id}.fhir.json')
                            if sanity_check:
                                assert not os.path.exists(save_path), colorstr("red", f"Same file exists: {save_path}")
                            json_save_fast(
                                save_path,
                                appointment_obj
                            )

                    # Follow-up consultation with physician.
                    # `appn['date']` is falsy when the consultation slot is out of the simulation day
                    if appn['date']:
                        _emit_consultation(
                            appn['department'],
                            appn['attending_physician'],
                            patient_id,
                            patient_name,
                            appn['date'],
                            appn['schedule']
                        )

                else:
                    raise ValueError(log(f"Invalid visit type: {appn['visit_type']}", "error"))

        return appointments


    @staticmethod
    def get_fhir_appointment(gt_resource_path: Optional[str] = None,
                             data: Optional[dict] = None) -> dict:
        """
        Load a FHIR Appointment resource from a file path if available, or generate it dynamically from the provided data.

        Args:
            gt_resource_path (Optional[str], optional):  
                Path to the ground-truth FHIR Appointment resource file.  
                If the file exists, it will be loaded and returned.  
                If not, a resource will be generated from the `data` argument.
            data (Optional[dict], optional):  
                Dictionary containing the metadata and patient information  
                needed to generate the Appointment resource.  
                Expected to include 'metadata' and 'information' keys.

        Returns:
            dict: A FHIR Appointment resource in dictionary form.
        """
        try:
            return json_load(gt_resource_path)
        except:
            metadata, info, department = data.get('metadata'), data.get('information'), data.get('department')
            schedule = info.get('schedule')
            if 'time' in schedule:
                schedule = schedule.get('time')
            
            gt_resource = DataConverter.data_to_appointment(
                {
                    'metadata': metadata,
                    'department': department,
                    'patient': {
                        info.get('patient'): {
                            'visit_type': info.get('visit_type'),
                            'department': info.get('department'),
                            'attending_physician': info.get('attending_physician'),
                            'date': info.get('date'),
                            'schedule': schedule
                        }
                    }
                }
            )[0]
            return gt_resource
    

    def __call__(self, 
                 output_dir: str,
                 sanity_check: bool = False) -> list[Information]:
        """
        Convert synthetic hospital data files into FHIR resources and optionally save them to disk.

        Args:
            output_dir (str): Directory to save the converted FHIR resources as `.fhir.json` files.
            sanity_check (bool, optional): If True, performs a sanity check to ensure the uniqueness of the generated FHIR data.
                                           This only applies when output_dir is specified. Defaults to False.

        Returns:
            list[Information]: An object containing the converted FHIR resources, including practitioners, schedules, slots, patients, and appointments.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_resources = list()
        
        for data_file in tqdm(self.data_files, desc='Converting to FHIR data..'):
            data = json_load(data_file)
            practitioners = DataConverter.data_to_practitioner(data, output_dir, sanity_check)
            practitionerroles = DataConverter.data_to_practitionerrole(data, output_dir, sanity_check)
            devices = DataConverter.data_to_device(data, output_dir, sanity_check)
            healthcareservices = DataConverter.data_to_healthcareservice(data, output_dir, sanity_check)
            schedules = DataConverter.data_to_schedule(data, output_dir, sanity_check)
            slots = DataConverter.data_to_slot(data, output_dir, sanity_check)
            patients = DataConverter.data_to_patient(data, output_dir, sanity_check)
            appointments = DataConverter.data_to_appointment(data, output_dir, sanity_check)

            information = Information(
                practitioners=practitioners,
                practitionerroles=practitionerroles,
                devices=devices,
                healthcareservices=healthcareservices,
                schedules=schedules,
                slots=slots,
                patients=patients,
                appointments=appointments
            )
            all_resources.append(information)
        
        return all_resources