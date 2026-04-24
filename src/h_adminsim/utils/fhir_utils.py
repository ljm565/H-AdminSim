import re

from h_adminsim.registry.variables import PRIORITY_MAP
from h_adminsim.utils.common_utils import (
    iso_to_hour,
    iso_to_date,
    sort_schedule,
    convert_time_list_to_merged_time,
)



def sanitize_id(s: str) -> str:
    """
    Sanitize a string to conform to the pattern: ^[A-Za-z0-9\-\.]{1,64}$

    Args:
        s (str): The input string to sanitize.

    Returns:
        str: A sanitized string containing only allowed characters,
             and no longer than 64 characters.
    """
    cleaned = re.sub(r'[^A-Za-z0-9\-\.]', '', s)
    return cleaned[:64]



def get_individual_id(hospital: str, department_code: str, individual_name: str) -> str:
    """
    Make an individual ID.

    Args:
        hospital (str): A hospital name.
        department_code (str): A department code.
        individual_name (str): An individual name.
    
    Returns:
        str: A sanitized individual ID.
    """
    return sanitize_id(f'{hospital}-{department_code.lower()}-{individual_name}')



def get_practitionerrole_id(individual_id: str) -> str:
    """
    Make a practitioner role ID for an individual.

    Args:
        individual_id (str): An individual ID.

    Returns:
        str: A practitioner role ID.
    """
    return f'{individual_id}-role'



def get_schedule_id(individual_id: str) -> str:
    """
    Make a schedule ID for an individual.

    Args:
        individual_id (str): An individual ID.

    Returns:
        str: A schedule ID.
    """
    return f'{individual_id}-schedule'



def get_slot_id(individual_id: str, date: str, time_segment_index: int) -> str:
    """
    Make a slot ID for an individual.

    Args:
        individual_id (str): An individual ID.
        date (str): A date in ISO format (YYYY-MM-DD).
        time_segment_index (int): An index of start time segment.

    Returns:
        str: A slot ID.
    """
    return f"{individual_id}-{date.replace('-', '')}-slot{time_segment_index}"



def get_appointment_id(individual_id: str, date: str, start_time_segment_index: int, end_time_segment_index: int) -> str:
    """
    Make an appointment ID for an individual.

    Args:
        individual_id (str): An individual ID.
        date (str): A date in ISO format (YYYY-MM-DD).
        start_time_segment_index (int): An index of start time segment.
        end_time_segment_index (int): An index of end time segment.

    Returns:
        str: An appointment ID.
    """
    return f"{individual_id}-{date.replace('-', '')}-appn{start_time_segment_index}-{end_time_segment_index}"



def get_device_id(hospital: str, device_id: str) -> str:
    """
    Make a Device ID from a test code.

    Args:
        hospital (str): A hospital name.
        device_id (str): A device ID (e.g., 'IMGAS-T01-0').

    Returns:
        str: A sanitized Device ID.
    """
    return sanitize_id(f'{hospital}-{device_id.lower()}')



def get_healthcareservice_id(hospital: str, test_code: str) -> str:
    """
    Make a HealthcareService ID from a test code.

    Args:
        hospital (str): A hospital name.
        test_code (str): A test code (e.g., 'IMGAS-T01').

    Returns:
        str: A sanitized HealthcareService ID.
    """
    return sanitize_id(f'{hospital}-{test_code.lower()}')



def get_all_doctor_info(practitioners: list[dict],
                        practitioner_roles: list[dict],
                        schedules: list[dict],
                        slots: list[dict],
                        appointments: list[dict],
                        **kwargs) -> dict:
    """
    Make a current state of doctoral information based on the FHIR server.

    Args:
        practitioners (list[dict]): Practitioner resources currently used in the hospital environment of the simulation.
        practitioner_roles (list[dict]): PractitionerRole resources currently used in the hospital environment of the simulation.
        schedules (list[dict]): Schedule resources currently used in the hospital environment of the simulation.
        slots (list[dict]): Slot resources currently used in the hospital environment of the simulation._
        appointments (list[dict]): Appointment resources currently used in the hospital environment of the simulation.

    Returns:
        dict: Current state of doctoral information. 
    """
    # Prepare several pre-required data
    doctor_information = dict()
    practitioner_ref_to_role = dict()
    practitioner_ref_to_schedules = dict()
    practitioner_ref_to_name = {
        f"Practitioner/{practitioner['resource']['id']}": \
            f"{practitioner['resource']['name'][0]['prefix'][0]} {practitioner['resource']['name'][0]['given'][0]} {practitioner['resource']['name'][0]['family']}" \
                for practitioner in practitioners
    }
    for practitioner_role in practitioner_roles:
        attributes = {attr['text']: attr['coding'][0]['display'] for attr in practitioner_role['resource']['characteristic']}
        practitioner_ref_to_role[practitioner_role['resource']['practitioner']['reference']] = {
            'department': practitioner_role['resource']['specialty'][0]['text'],
            'specialty': {
                'name': practitioner_role['resource']['specialty'][0]['coding'][0]['display'],
                'code': practitioner_role['resource']['specialty'][0]['coding'][0]['code']
            },
            'capacity_per_hour': int(attributes['capacity_per_hour']),
            'capacity': int(attributes['capacity']),
        }
    schedule_ref_to_practitioner_ref = {
        f"Schedule/{schedule['resource']['id']}": schedule['resource']['actor'][0]['reference'] for schedule in schedules
        if schedule['resource']['actor'][0]['reference'].startswith('Practitioner/')
    }

    # Append fixed schedules of a doctor
    for slot in slots:
        resource = slot['resource']
        practitioner_ref = schedule_ref_to_practitioner_ref.get(slot['resource']['schedule']['reference'])
        if practitioner_ref:
            date = iso_to_date(resource['start'])
            practitioner_dict = practitioner_ref_to_schedules.setdefault(practitioner_ref, {})
            practitioner_dict.setdefault(date, [])
            if not resource['status'] == 'free':
                practitioner_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])

    # Merge fixed schedule times
    if all(k in kwargs for k in ['start', 'end', 'interval']):
        for fixed_schedules in practitioner_ref_to_schedules.values():
            for date, time_list in fixed_schedules.items():
                fixed_schedules[date] = convert_time_list_to_merged_time(time_list=sort_schedule(time_list), **kwargs)
    
    # Append patient appointments of a doctor
    for appointment in appointments:
        resource = appointment['resource']
        for participant in resource['participant']:
            reference = participant['actor']['reference']
            if reference in practitioner_ref_to_name:
                date = iso_to_date(resource['start'])
                practitioner_dict = practitioner_ref_to_schedules.setdefault(reference, {})
                practitioner_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])
                break
        
    # Build the doctor information from FHIR
    for practitioner in practitioners:
        resource = practitioner['resource']
        ref = f"Practitioner/{resource['id']}"
        doctor_information[practitioner_ref_to_name[ref]] = {
            'department': practitioner_ref_to_role[ref]['department'],
            'specialty': practitioner_ref_to_role[ref]['specialty'],
            'schedule': sort_schedule(practitioner_ref_to_schedules.get(ref, [])),
            'capacity_per_hour': practitioner_ref_to_role[ref]['capacity_per_hour'],
            'capacity': practitioner_ref_to_role[ref]['capacity'],
            'gender': resource['gender'],
            'telecom': resource['telecom'],
            'birthDate': resource['birthDate']
        }

    return doctor_information


def get_all_test_info(practitioner_roles: list[dict],
                      healthcare_services: list[dict],
                      devices: list[dict],
                      schedules: list[dict],
                      slots: list[dict],
                      appointments: list[dict],
                      **kwargs) -> dict:
    """
    Make a current state of test information based on the FHIR server.

    Args:
        practitioner_roles (list[dict]): PractitionerRole resources currently used in the hospital environment of the simulation.
        healthcare_services (list[dict]): HealthcareService resources currently used in the hospital environment of the simulation.
        devices (list[dict]): Device resources currently used in the hospital environment of the simulation.
        schedules (list[dict]): Schedule resources currently used in the hospital environment of the simulation.
        slots (list[dict]): Slot resources currently used in the hospital environment of the simulation.
        appointments (list[dict]): Appointment resources currently used in the hospital environment of the simulation.

    Returns:
        dict: Current state of test information.
    """
    # Prepare several pre-required data
    test_information = dict()
    code_to_department = dict()
    device_ref_to_schedules = dict()
    device_schedule = dict()
    device_ref_to_device_key = dict()
    schedule_ref_to_device_ref = {
        f"Schedule/{schedule['resource']['id']}": schedule['resource']['actor'][0]['reference'] for schedule in schedules
        if schedule['resource']['actor'][0]['reference'].startswith('Device/')
    }

    # Gather department code information from PractitionerRole resources
    for entry in practitioner_roles:
        specialties = entry["resource"].get("specialty", [])
        for spec in specialties:
            full_code = spec["coding"][0]["code"]      # e.g., IMNEP-1
            dept_code = full_code.split("-")[0].lower()  # e.g., imnep
            dept_name = spec.get("text", "Unknown")
            code_to_department[dept_code] = dept_name

    # Gather basic test information according to the department
    for entry in healthcare_services:
        resource = entry["resource"]
        
        # Find the department
        department = None
        for c in code_to_department:
            if c in resource['id']:
                department = code_to_department[c]
                break

        if department not in test_information:
            test_information[department] = list()
        
        attributes = {attr['text']: attr['coding'][0]['display'] for attr in resource['characteristic']}
        eligibility_list = resource.get('eligibility', [])
        eligibility = {
            'depends_on': sorted([
                item['code']['coding'][0]['code']
                for item in eligibility_list
                if item['code']['coding'][0]['display'] == 'required'
            ]),
            'avoid_same_day': sorted([
                item['code']['coding'][0]['code']
                for item in eligibility_list
                if item['code']['coding'][0]['display'] == 'avoid'
            ])
        }
        test_information[department].append(
            {
                "name": resource['name'],
                "code": resource['type'][0]['coding'][0]['code'],
                "duration_hour": float(attributes['duration_hour']),
                "priority": PRIORITY_MAP['code_to_priority'][attributes['priority']],
                "result_hours": int(attributes['result_hours']),
                "depends_on": eligibility['depends_on'],
                "avoid_same_day": eligibility['avoid_same_day'],
                "description": resource['comment'],
                "device_n": len(resource['serviceProvisionCode']),
                "devices": {},
            }
        )

    # Append fixed schedules of a doctor
    for slot in slots:
        resource = slot['resource']
        device_ref = schedule_ref_to_device_ref.get(slot['resource']['schedule']['reference'])
        if device_ref:
            date = iso_to_date(resource['start'])
            device_dict = device_ref_to_schedules.setdefault(device_ref, {})
            device_dict.setdefault(date, [])
            if not resource['status'] == 'free':
                device_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])

    # Merge schedules
    if all(k in kwargs for k in ['start', 'end', 'interval']):
        for fixed_schedules in device_ref_to_schedules.values():
            for date, time_list in fixed_schedules.items():
                fixed_schedules[date] = convert_time_list_to_merged_time(time_list=sort_schedule(time_list), **kwargs)

    # Append patient appointments of a device
    for appointment in appointments:
        resource = appointment['resource']
        for participant in resource['participant']:
            reference = participant['actor']['reference']
            if reference in device_ref_to_schedules:
                date = iso_to_date(resource['start'])
                device_dict = device_ref_to_schedules.setdefault(reference, {})
                device_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])
                break

    # Test per device schedule
    for entry in devices:
        resource = entry["resource"]
        test_key = resource['type'][0]['coding'][0]['code']
        device_key = f"{test_key}-{resource['id'].split('-')[-1]}"
        device_ref = f"Device/{resource['id']}"
        device_ref_to_device_key[device_ref] = device_key
        device_schedule.setdefault(test_key, {})
        device_schedule[test_key][device_ref] = {"schedule": sort_schedule(device_ref_to_schedules[device_ref])}

    # Make test information with fixed device schedule
    for test_list in test_information.values():
        for test in test_list:
            for device_ref, schedule in device_schedule[test['code']].items():
                test['devices'][device_ref_to_device_key[device_ref]] = schedule

    return test_information
