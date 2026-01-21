STATUS_CODES = {
    'format': 'incorrect format',
    'department': 'incorrect department',
    'patient': 'incorrect patient information',
    'department & patient': 'incorrect department and patient information',
    'simulation': 'incomplete simulation',
    'schedule': 'invalid schedule',
    'duration': 'wrong duration',
    'conflict': {
        'physician': 'physician conflict', 
        'time': 'time conflict'
    },
    'preference': {
        'physician': 'mismatched physician',
        'asap': 'not earliest schedule',
        'date': 'not valid date',
    },
    'tool': 'wrong tool selection or wrong type of argument',
    'cancel': {
        'identify': 'cancel: fail to identify requested schedule',
        'type': 'cancel: unexpected tool calling result'
    },
    'reschedule': {
        'identify': 'reschedule: fail to identify requested schedule',
        'schedule': 'reschedule: {status_code}',
        'type': 'reschedule: unexpected tool calling result'
    },
    'waiting list': 'fail to add to waiting list',
    # 'workload': 'workload balancing',
    'preceding': 'preceding task failed',
    'correct': 'pass',
}

# SCHEDULING_ERROR_CAUSE = {
#     'incorrect format': [
#         '* There is an issue with the output format. Please perform scheduling in the correct format.',
#     ],
#     'physician conflict': [
#         '* More than one doctor has been assigned. A schedule must be made with exactly one doctor.',
#     ],
#     'time conflict': [
#         "* The scheduling result overlaps with the doctor's existing schedule.",
#     ],
#     'mismatched physician': [
#         '* A different doctor was assigned even though the patient requested a specific doctor.',
#     ],
#     'not earliest schedule': [
#         '* The patient wants the earliest possible appointment in the department, but the assigned time is not the earliest available based on the current time.',
#         '* When scheduling, it is possible to assign an earlier date or time.',
#         "* The previous patient's schedule may have been cancelled. Therefore, it is necessary to carefully compare the hospital's start time with the doctor's schedule to identify available time slots.",
#     ],
#     'not valid date': [
#         '* The patient is available after a specific date and would like to make an appointment. Please choose the earliest possible time after that date.',
#     ],
#     'invalid schedule': [
#         "* The scheduling result may fall outside the hospital's operating hours.",
#         "* The scheduling result may be in the past relative to the current time.",
#         "* The scheduling result may not be a valid date.",
#         "* The assigned doctor may not belong to the department the patient should visit.",
#     ],
#     'wrong duration': [
#         "* The patient's schedule does not match the consultation duration required by the doctor.",
#     ],
#     # 'workload balancing': [
#     #     "* You must schedule the appointment with a doctor who has a lower workload than the current doctor.",
#     # ]
# }


class ToolCallingError(Exception):
    error_code = "TOOL_CALLING_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
