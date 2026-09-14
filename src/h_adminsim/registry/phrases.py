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
    'throughput_max': 'You want to complete all required tests as soon as possible, regardless of the number of hospital visits.',
    'visit_min': 'You want to minimize the number of hospital visits by scheduling all required tests together on as few days as possible.',
    'stay_min': 'You want to minimize the time spent waiting at the hospital between tests, regardless of the number of hospital visits.',
    'indifferent': 'You have no specific preference regarding the scheduling of your follow-up tests, so you are willing to go along with whatever schedule the staff proposes.'
}
OPFU_PREFERENCE_PHRASE_STAFF = {
    'throughput_max': 'The patient wants to complete all required tests as soon as possible, regardless of the number of hospital visits.',
    'visit_min': 'The patient wants to minimize the number of hospital visits by scheduling all required tests together on as few days as possible.',
    'stay_min': 'The patient wants to minimize the time spent waiting at the hospital between tests, regardless of the number of hospital visits.',
    'indifferent': 'The patient has no specific scheduling preference for the follow-up tests, so the staff arranges all required tests as soon as possible, following the hospital\'s default policy.'
}

OPFU_PROPOSAL_PHRASE_STAFF = {
    'throughput_max': 'The staff has arranged every test as early as possible, across as many separate days as that takes.',
    'visit_min': 'The staff has packed every test into as few visit days as possible, even though the results then come back later.',
    'stay_min': 'The staff has spread the tests out so you never sit waiting between two tests on the same day, even though that means coming in on more days.',
    'indifferent': "The staff has arranged every test as early as possible, following the hospital's default policy.",
}

OPFU_UNAVAILABLE_PHRASE_PATIENT = {
    'none': 'There is no day or time you are unable to come in for the tests, so any slot the hospital offers works for you.',
    'day': 'You CANNOT come in for the tests at all on these dates: {dates}. Any other date is fine. Reason: {explanation}',
    'half_day': 'You CANNOT come in for the tests in the {half_day_word} ({half_day_range}) on these dates: {dates}. '
                'The rest of those days, and every other date, is fine. Reason: {explanation}',
}
OPFU_UNAVAILABLE_PHRASE_STAFF = {
    'none': 'The patient has not reported any day or time on which they cannot come in for the tests.',
    'day': 'The patient CANNOT come in for the tests at all on these dates: {dates}. Any other date is fine.',
    'half_day': 'The patient CANNOT come in for the tests in the {half_day_word} ({half_day_range}) on these dates: {dates}. '
                'The rest of those days, and every other date, is fine.',
}

AGENT_DESCRIPTION = {
    'orchestrator': 'Oversees the entire patient intake and scheduling process, making high-level decisions and delegating tasks to sub-agents.',
    'first_visit_intake': 'Patient intake and department recommendation. Route the conversation to this agent whenever the patient mentions a disease, symptoms, or states that they are seeking medical care (e.g., doctor) due to a health concern.',
    'first_visit_scheduling': 'Schedule a first-visit appointment. Route the conversation to this agent whenever the patient indicates how they would like to schedule an appointment with a physician after completing the intake process. Also route to this agent whenever the patient requests to cancel or reschedule an existing appointment with a physician.',
    'follow_up_visit_scheduling': 'Schedule follow-up visit tests and appointments. Route the conversation to this agent whenever the patient mentions needing to schedule a diagnostic test after a physician visit or requests to arrange a test appointment. Also route to this agent whenever the patient requests to cancel or reschedule an existing diagnostic test schedule.'
}