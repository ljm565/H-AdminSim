import re
from typing import Union

from h_adminsim.utils.common_utils import (
    iso_to_hour,
    iso_to_date,
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



def convert_fhir_resources_to_doctor_info(practitioners: list[dict],
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
    def __sort_schedule(data: Union[dict, list]) -> Union[dict, list]:
        if isinstance(data, list):
            return sorted(data)
        return {k: sorted(v) for k, v in dict(sorted(data.items())).items()}

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
    schedule_ref_to_practioner_ref = {
        f"Schedule/{schedule['resource']['id']}": schedule['resource']['actor'][0]['reference'] for schedule in schedules
    }

    # Append fixed schedules of a doctor
    for slot in slots:
        resource = slot['resource']
        practitioner_ref = schedule_ref_to_practioner_ref[slot['resource']['schedule']['reference']]
        date = iso_to_date(resource['start'])
        practitioner_dict = practitioner_ref_to_schedules.setdefault(practitioner_ref, {})
        practitioner_dict.setdefault(date, [])
        if not resource['status'] == 'free':
            practitioner_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])

    # Merge fixed schedule times
    if all(k in kwargs for k in ['start', 'end', 'interval']):
        for fixed_schedules in practitioner_ref_to_schedules.values():
            for date, time_list in fixed_schedules.items():
                fixed_schedules[date] = convert_time_list_to_merged_time(time_list=__sort_schedule(time_list), **kwargs)

    # Append patient appointments of a doctor
    for appointment in appointments:
        resource = appointment['resource']
        for participant in resource['participant']:
            participant_ref = participant['actor']['reference']
            date = iso_to_date(resource['start'])
            practitioner_dict = practitioner_ref_to_schedules.setdefault(participant_ref, {})
            if participant_ref in practitioner_ref_to_name:
                practitioner_dict[date].append([iso_to_hour(resource['start']), iso_to_hour(resource['end'])])
                break
        
    # Build the doctor information from FHIR
    for practitioner in practitioners:
        resource = practitioner['resource']
        ref = f"Practitioner/{resource['id']}"
        doctor_information[practitioner_ref_to_name[ref]] = {
            'department': practitioner_ref_to_role[ref]['department'],
            'specialty': practitioner_ref_to_role[ref]['specialty'],
            'schedule': __sort_schedule(practitioner_ref_to_schedules.get(ref, [])),
            'schedule': {k: sorted(v) for k, v in dict(sorted(practitioner_ref_to_schedules.get(ref, []).items())).items()},
            'capacity_per_hour': practitioner_ref_to_role[ref]['capacity_per_hour'],
            'capacity': practitioner_ref_to_role[ref]['capacity'],
            'gender': resource['gender'],
            'telecom': resource['telecom'],
            'birthDate': resource['birthDate']
        }

    return doctor_information



def get_patient_from_appointment(resource: dict) -> Union[str, None]:
    """
    Extracts the patient's display name from a FHIR Appointment resource.

    Args:
        resource (dict): A FHIR Appointment resource represented as a dictionary. 
                         It should contain a 'participant' field, each with an 'actor' referencing a FHIR resource.

    Returns:
        Union[str, None]: The display name of the patient if found, otherwise None.
    """
    for participant in resource.get('participant', []):
        actor = participant.get('actor', {})
        if 'Patient' in actor.get('reference', ''):
            return actor.get('display')
    return None
