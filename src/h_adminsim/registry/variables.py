FIRST_NAMES = None
LAST_NAMES = None
DEPARTMENTS = None
HOSPITALS = None
COUNTRY_TIMEZONE_MAP = None
SYMPTOM_MAP = None
TELECOM_COUNTRY_CODE = None
SPECIALTIES = None
ADDRESSES = None
DEPARTMENT_TESTS = None
PRIORITY_MAP = {
    'priority_to_code': {
        0: 'initial',
        1: 'intermediate',
        2: 'advanced'
    },
    'code_to_priority': {
        'initial': 0,
        'intermediate': 1,
        'advanced': 2
    }
}
SCHEDULE_STATUS = {
    'scheduled': 'scheduled',
    'in_progress': 'in_progress',
    'completed': 'completed',
    'cancelled': 'cancelled',
    'not_yet': 'not_yet',
}
DEPARTMENT_NORMALIZATION = {
    'rheumatory': 'rheumatology',
    'pulmonology': 'pulmonary',
}
OCCUPATION = None