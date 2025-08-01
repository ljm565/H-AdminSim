import re
from utils.common_utils import get_time_hour



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



def get_individual_id(hospital: str, department: str, individual_name: str) -> str:
    """
    Make an individual ID.

    Args:
        hospital (str): A hospital name.
        department (str): A department name.
        individual_name (str): An individual name.
    
    Returns:
        str: A sanitized individual ID.
    """
    return sanitize_id(f'{hospital}-{department}-{individual_name}')



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



def get_slot_id(individual_id: str, time_segment_index: int) -> str:
    """
    Make a slot ID for an individual.

    Args:
        individual_id (str): An individual ID.
        time_segment_index (int): An index of start time segment.

    Returns:
        str: A slot ID.
    """
    return f'{individual_id}-slot{time_segment_index}'



def get_appointment_id(individual_id: str, start_time_segment_index: int, end_time_segment_index: int) -> str:
    """
    Make an appointment ID for an individual.

    Args:
        individual_id (str): An individual ID.
        start_time_segment_index (int): An index of start time segment.
        end_time_segment_index (int): An index of end time segment.

    Returns:
        str: An appointment ID.
    """
    return f'{individual_id}-appn{start_time_segment_index}-{end_time_segment_index}'



def convert_fhir_resources_to_doctor_info(practitioners: list[dict],
                                          practitioner_roles: list[dict],
                                          schedules: list[dict],
                                          slots: list[dict],
                                          appointments: list[dict]) -> dict:
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
    practitioner_ref_to_schedules = dict()
    practitioner_ref_to_name = {
        f"Practitioner/{practitioner['resource']['id']}": \
            f"{practitioner['resource']['name'][0]['prefix'][0]} {practitioner['resource']['name'][0]['given'][0]} {practitioner['resource']['name'][0]['family']}" \
                for practitioner in practitioners
    }
    practitioner_ref_to_role = {
        practitioner_role['resource']['practitioner']['reference']: \
            {
                'department': practitioner_role['resource']['specialty'][0]['text'],
                'specialty': {
                    'name': practitioner_role['resource']['specialty'][0]['coding'][0]['display'],
                    'code': practitioner_role['resource']['specialty'][0]['coding'][0]['code']
                }
            } for practitioner_role in practitioner_roles
    }
    schedule_ref_to_practioner_ref = {
        f"Schedule/{schedule['resource']['id']}": schedule['resource']['actor'][0]['reference'] for schedule in schedules
    }

    # Append fixed schedules of a doctor
    for slot in slots:
        resource = slot['resource']
        practitioner_ref = schedule_ref_to_practioner_ref[slot['resource']['schedule']['reference']]
        if not resource['status'] == 'free':
            practitioner_ref_to_schedules.setdefault(practitioner_ref, []).append([get_time_hour(resource['start']), get_time_hour(resource['end'])])

    # Append patient appointments of a doctor
    for appointment in appointments:
        resource = appointment['resource']
        for participant in resource['participant']:
            participant_ref = participant['actor']['reference']
            if participant_ref in practitioner_ref_to_name:
                practitioner_ref_to_schedules.setdefault(participant_ref, []).append([get_time_hour(resource['start']), get_time_hour(resource['end'])])
                break
        
    # Build the doctor information from FHIR
    for practitioner in practitioners:
        resource = practitioner['resource']
        ref = f"Practitioner/{resource['id']}"
        doctor_information[practitioner_ref_to_name[ref]] = {
            'department': practitioner_ref_to_role[ref]['department'],
            'specialty': practitioner_ref_to_role[ref]['specialty'],
            'schedule': sorted(practitioner_ref_to_schedules.get(ref, [])),
            'gender': resource['gender'],
            'telecom': resource['telecom'],
            'birthDate': resource['birthDate']
        }

    return doctor_information
