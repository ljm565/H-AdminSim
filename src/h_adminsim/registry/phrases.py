OPFV_PREFERENCE_PHRASE_PATIENT = {
    'asap': 'You want the earliest available doctor in the department for the outpatient visit.',
    'doctor': 'You have a preferred doctor for the outpatient visit.',
    'date': 'You want the earliest available doctor in the department for the outpatient visit, starting from **{date}**.'
}
OPFV_PREFERENCE_PHRASE_STAFF = {
    'asap': 'The patient wants the earliest available doctor in the department for the outpatient visit.',
    'doctor': 'The patient has a preferred doctor for the outpatient visit.',
    'date': 'The patient wants the earliest available doctor in the department for the outpatient visit, starting from **{date}**.'
}
OPFU_PREFERENCE_PHRASE_PATIENT = {
    'asap': 'You want to complete all required tests as soon as possible, regardless of the number of hospital visits.',
    'batch': 'You want to minimize the number of hospital visits by scheduling all required tests together on as few days as possible.'
}
OPFU_PREFERENCE_PHRASE_STAFF = {
    'asap': 'The patient wants to complete all required tests as soon as possible, regardless of the number of hospital visits.',
    'batch': 'The patient wants to minimize the number of hospital visits by scheduling all required tests together on as few days as possible.'
}

AGENT_DESCRIPTION = {
    'orchestrator': 'Oversees the entire patient intake and scheduling process, making high-level decisions and delegating tasks to sub-agents.',
    'first_visit_intake': 'Patient intake and department recommendation. Route the conversation to this agent whenever the patient mentions a disease, symptoms, or states that they are seeking medical care (e.g., doctor) due to a health concern.',
    'first_visit_scheduling': 'Schedule a first-visit appointment. Route the conversation to this agent whenever the patient indicates how they would like to schedule an appointment with a physician after completing the intake process. Also route to this agent whenever the patient requests to cancel or reschedule an existing appointment with a physician.',
    'follow_up_visit_scheduling': 'Schedule follow-up visit tests and appointments. Route the conversation to this agent whenever the patient mentions needing to schedule a diagnostic test after a physician visit or requests to arrange a test appointment. Also route to this agent whenever the patient requests to cancel or reschedule an existing diagnostic test schedule.'
}