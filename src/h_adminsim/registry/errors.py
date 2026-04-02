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
    'preceding': 'preceding task failed',
    'unexpected': "unexpected error: {e}",
    'correct': 'pass',
}


class ToolCallingError(Exception):
    error_code = "TOOL_CALLING_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ScheduleNotFoundError(Exception):
    error_code = "SCHEDULE_NOT_FOUND_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SchedulingError(Exception):
    error_code = "SCHEDULING_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
