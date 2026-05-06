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
    'cancel': {
        'identify': 'cancel: fail to identify requested schedule',
        'type': 'cancel: unexpected tool calling result'
    },
    'reschedule': {
        'identify': 'reschedule: fail to identify requested schedule',
        'schedule': 'reschedule: {status_code}',
        'type': 'reschedule: unexpected tool calling result'
    },
    'test_retrieve': {
        'identify': 'test_retrieve: fail to identify requested schedule',
        'type': 'test_retrieve: unexpected tool calling result'
    },
    'test_schedule': {
        'format': 'test_schedule: incorrect format',
        'coverage': 'test_schedule: required tests not all covered',
        'schedule': 'test_schedule: invalid schedule',
        'duration': 'test_schedule: wrong duration',
        'conflict': 'test_schedule: device conflict',
        'dependency': 'test_schedule: dependency violated',
        'avoid_same_day': 'test_schedule: avoid_same_day violated',
        'preference': {
            'asap': 'test_schedule: not earliest result-ready time',
            'batch': 'test_schedule: not minimum visit-date count',
        },
        'fu_schedule': 'test_schedule: not earliest follow-up appointment',
    },
    'preceding': 'preceding task failed',
    'unexpected': "unexpected error: {e}",
    'correct': 'pass',
}


class ToolCallingError(Exception):
    error_code = "TOOL_CALLING_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DataNotFoundError(Exception):
    error_code = "DATA_NOT_FOUND_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SchedulingError(Exception):
    error_code = "SCHEDULING_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
