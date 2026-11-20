# H-AdminSim

---
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2602.05407-red)](https://arxiv.org/abs/2602.05407)
[![PyPI Version](https://img.shields.io/pypi/v/h-adminsim)](https://pypi.org/project/h-adminsim/)
[![Downloads](https://img.shields.io/pypi/dm/h-adminsim?cacheSeconds=86400)](https://pypi.org/project/h-adminsim/)

* Paper: [H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration
](https://arxiv.org/abs/2602.05407) (Best Paper Award)
* Project Page: https://ljm565.github.io/h_adminsim_project/
* PyPI Package: https://pypi.org/project/h-adminsim/

&nbsp;

---
## Recent updates 📣
* *v2.0.0 (November 2026)*: Added follow-up visit (test and consultation) scheduling simulation and a multi-agent administrative staff. **Contains breaking API changes** ([migration guide](#migrating-from-v1x)).
    * Added follow-up visit data synthesis (diagnostic tests, test devices, and device schedules) and the `OutpatientFollowUpScheduling` task.
    * Added follow-up patient scheduling preferences (`throughput_max`, `visit_min`, `stay_min`, `indifferent`) with occupation-based unavailable times.
    * Added staff-side negotiation policies that steer follow-up patients toward hospital-efficient schedules, plus urgent-patient capacity protection.
    * Replaced the single admin staff agent with a multi-agent system (`HospitalMAS`): an orchestrator routes each conversation to task-specific staff agents.
    * `Simulator` now takes a `task` dictionary and can run first-visit and follow-up tasks together or separately.
    * Added FHIR integration for test schedules, and resume support for follow-up simulations.
    * `Evaluator` now covers both first-visit and follow-up (negotiation and capacity protection) evaluations.
* *v1.2.4&mdash;v1.2.6 (April 2026)*: Improved scheduling simulation realism and stability.
* *v1.2.3 (April 2026)*: Add support for custom department and disease-symptom data.
    <!-- * Added `--department_info_path` and `--disease_symptom_file_path` arguments to allow users to plug in custom department and disease-symptom data in standalone mode.
    * Fixed minor bugs. -->
* *v1.2.1, v1.2.2 (April 2026)*: Update license to Apache License 2.0.
* *v1.2.0 (April 2026)*: Bug fixes and simulation improvements.
* *v1.1.0 (March 2026)*: Merge administrative simulation code from `patientsim` into `h-adminsim`.
* *v1.0.5 (March 2026)*: Minor improvements.
* *v1.0.4 (March 2026)*: Simulation improvements.
* *v1.0.3 (February 2026)*: Update error exception codes for observable traceback.
* *v1.0.2 (February 2026)*: Update dependencies for stable simulation.
* *v1.0.1 (February 2026)*: H-AdminSim now supports concurrent simulation and open-source tool calling.
* *v1.0.0 (February 2026)*: H-AdminSim package has been released.
<!-- * *v0.7.2 (December 2025)*: Rule-based and tool calling-based scheduling logics have been supported.
* *v0.7.1 (November 2025)*: Self-corrective feedback logic has been supported.
* *v0.7.0 (October 2025)*: Simulation has been improved reflecting feedbacks from experts.
* *v0.6.0 (September 2025)*: Simulation has been improved reflecting feedbacks from experts. -->
<!-- * *August 2025 (v0.5.2)*: We has supported vLLM inference of the Hugging Face models.
* *August 2025 (v0.5.1)*: Now you can easily set the virtual environment using Poetry.
* *August 2025 (v0.5.0)*: We integrated the FHIR server to retrieve hospital information during hospital administration office agent simulation. -->
<!-- * *July 2025 (v0.4.2)*: We have supported LangChain's JsonOutputParser funtion as well as naive LLM API methods.
* *July 2025 (v0.4.1)*: Add functionality to schedule appointments based on the hospital's current time (the time the patient contacted for booking).
* *July 2025 (v0.4.0)*: Added a hospital simulation environment to enable rescheduling based on patient priority, flexibility, and other constraints.
* *July 2025 (v0.3.1)*: Added agent results evaluation codes.
* *July 2025 (v0.3.0)*: This repository has supported Gemini- and GPT-based LLM agent task testing: 'department', 'schedule', 'fhir_resource', 'fhir_api'.
* *June 2025 (v0.2.2)*: Added *PractitionerRole* resource type and function to make more realistic data.
* *June 2025 (v0.2.1)*: Fixed *Appointment* resource type error and added function to show failed files during creating data to FHIR.
* *June 2025 (v0.2.0)*: Added function to map synthetic data to some *workflow* resource types in FHIR.
* *June 2025 (v0.1.2)*: The data synthesis speed has been improved, and a sanity check feature has been added during synthesis.
* *June 2025 (v0.1.1)*: Added random patient data synthesizing codes and completed sanity check.
* *June 2025 (v0.1.0)*: Added random hospital data synthesizing codes and completed sanity check.
* *June 2025 (v0.0.5)*: Enhanced and fixed the FHIR manager operations.
* *June 2025 (v0.0.4)*: Updated documents (environment setting, CRUD guides)
* *June 2025 (v0.0.3)*: Now we has supported FHIR CRUD.
* *June 2025 (v0.0.2)*: Created chat demo `README.md`.
* *June 2025 (v0.0.1)*: Created chat demo codes using FastAPI communication. -->

&nbsp;

&nbsp;

---


## Overview 📚
H-AdminSim is an official Python package for simulating interactions between hospital administrative staff and outpatients (first-visit and follow-up) using LLM agents.
It provides a standardized evaluation testbed for assessing LLM performance across key administrative tasks across multiple care levels (primary, secondary, and tertiary), with optional [FHIR](https://www.hl7.org/fhir/) integration and support for heterogeneous deployment environments, allowing flexible simulation workflows tailored to diverse hospital systems.

Large hospitals often handle 10,000+ outpatient encounters per day, and prior reports indicate limited specialization among administrative staff despite high workload.
H-AdminSim is designed to help address these challenges by offering a realistic, reproducible simulation environment that supports future hospital automation and LLM-assisted administrative workflows.


&nbsp;



### 1. Care level-specific data synthesis
We provide [configuration examples](https://github.com/ljm565/H-AdminSim/blob/master/src/h_adminsim/assets/configs/) for simulating primary, secondary, and tertiary care settings.
Each configuration reflects key characteristics of its hospital level:

* Hospital time granularity: `tertiary` < `secondary` = `primary` (coarser in lower levels)
* Number of departments: `primary` < `secondary` < `tertiary`
* Number of physicians: `primary` < `secondary` < `tertiary`
* Patient referral rate: `primary` < `secondary` < `tertiary`
* Proportion of patients with preferred `physician`/`date`: `primary` < `secondary` = `tertiary`

You may also define your own conditions using a custom configuration file (e.g., [data_synthesis.yaml](https://github.com/ljm565/H-AdminSim/blob/master/config/data_synthesis.yaml))


&nbsp;

### 2. Hospital Administration Simulation
#### 2.1. Patient Intake Simulation
We extend the previously emergency department-focused [PatientSim](https://github.com/ljm565/patientsim-pkg) to enable realistic conversations between administrative staff and first-visit outpatients with diverse backgrounds.
* **Disease profile**: One of 194 disease–symptom pairs across 9 internal-medicine departments (`gastroenterology`, `cardiology`, `pulmonology`, `endocrinology/metabolism`, `nephrology`, `hematology/oncology`, `allergy`, `infectious diseases`, `rheumatology`)
* **Medical referral status**: Dialogue flow adapts based on whether the patient has a referral
* **Tasks**: Department recommendation, information extraction, structured data construction


#### 2.2. Appointment Scheduling Simulation
We simulate realistic scheduling interactions between administrative staff and patients, reflecting diverse scheduling behaviors and hospital-level constraints.
* **Time flow**: Users can define the simulation period and starting point, enabling the agent to perform time-related tasks based on the progression of simulated time.
* **Patient preferences**: `ASAP` (earliest slot), `physician` (specific physician requested), `date` (preferred date range start)
* **Random requests**: `cancellation`, `rescheduling`
* **Tasks**: New appointment scheduling, rescheduling, schdule cancellation


#### 2.3. Follow-up Visit Scheduling Simulation
After the first visit, patients often need several diagnostic tests before a follow-up consultation. We simulate how administrative staff book these test sequences on shared test devices.
* **Test scheduling**: Each patient needs several tests (possibly from other departments). The tests must be booked on limited test devices, followed by a consultation with the attending physician.
* **Patient preferences**: `throughput_max` (get results as early as possible), `visit_min` (fewest hospital visit days), `stay_min` (least waiting between tests), `indifferent` (no preference)
* **Patient constraints**: Occupation-based unavailable times and home-to-hospital distance
* **Staff negotiation**: Configurable policies decide when the staff tries to move a patient from their own preference toward a more hospital-efficient schedule.
* **Capacity protection**: Optionally holds back part of each test's capacity for `urgent` patients
* **Tasks**: Follow-up test and consultation scheduling, rescheduling, schedule cancellation


#### 2.4. Multi-Agent Administrative Staff
The administrative staff is a multi-agent system (`HospitalMAS`). An orchestrator agent reads each patient message and routes it to the staff agent for that task (`first_visit_intake`, `first_visit_scheduling`, or `follow_up_visit_scheduling`). After intake, the conversation is handed over to first-visit scheduling.


#### 2.5. FHIR Integration
We provide optional support for integrating with FHIR, allowing the simulator to operate flexibly across heterogeneous hospital environments as long as FHIR-compatible data is available. For instructions on running a FHIR server, please refer to the [FHIR Server Execution](https://github.com/ljm565/fhir_server) repository.

&nbsp;

&nbsp;

---
## Quick Starts 🚀
### 1. Installation
```bash
pip install h_adminsim
```
```python
import h_adminsim
print(h_adminsim.__version__)
```

&nbsp;

### 2. Environment Variables
Before using the LLM API, you need to provide the API key (or the required environment variables for each model) either directly or in a `.env` file.
```bash
# For GPT API without Azure
OPENAI_API_KEY=${YOUR_OPENAI_KEY}

# For Gemini API
GOOGLE_API_KEY=${YOUR_GEMINI_API_KEY}
```

&nbsp;

### 3. Simulation
```python
from h_adminsim.pipeline import DataGenerator, Simulator, HospitalMAS
from h_adminsim.task.opfv_task import OutpatientFirstIntake, OutpatientFirstScheduling
from h_adminsim.task.opfu_task import OutpatientFollowUpScheduling
from h_adminsim.utils.mas_utils import init_mas_system

# 1. Data synthesis (first-visit + follow-up visit)
data_generator = DataGenerator(task=['first_visit', 'follow_up_visit'])
data_generator.build(convert_to_fhir=True)
agent_data_dir = data_generator.save_dir / 'agent_data'
output_dir = data_generator.save_dir / 'simulation_results'

# 2. Administrative staff MAS (orchestrator + one staff agent per task)
staff_mas = HospitalMAS(
    init_mas_system(
        orchestrator_model='gpt-5-mini',
        orchestrator_vllm_endpoint=None,
        first_visit_intake_model='gpt-5-mini',
        first_visit_scheduling_model='gpt-5-mini',
        follow_up_visit_scheduling_model='gpt-5-mini',
    )
)

# 3. Tasks (keys must be each task's `name`)
task = {
    'first_visit_intake': OutpatientFirstIntake(
        patient_model='gpt-5-nano',
        admin_staff_mas=staff_mas,
    ),
    'first_visit_scheduling': OutpatientFirstScheduling(
        patient_model='gpt-5-nano',
        admin_staff_mas=staff_mas,
    ),
    'follow_up_visit_scheduling': OutpatientFollowUpScheduling(
        patient_model='gpt-5-nano',
        admin_staff_mas=staff_mas,
    ),
}

# 4. Simulation
simulator = Simulator(task=task)
simulator.run(
    simulation_data_path=agent_data_dir,
    output_dir=output_dir,
    resume=False,
    verbose=True
)
```

&nbsp;

&nbsp;

---
## Components Details ⚙️
### 1. Data synthesis
```python
from h_adminsim.pipeline import DataGenerator

# 1. Generator Initialization
# 1.1. Default usaage
data_generator = DataGenerator()    # Default: primary care
# data_generator = DataGenerator(care_level='secondary') # For secondary care
# data_generator = DataGenerator(care_level='tertiary')  # For tertiary care

# 1.2. You can synthesize data with your own configuration
data_generator = DataGenerator(config='data_config.yaml')

# 1.3. Follow-up visit data (diagnostic tests and test devices) on top of the first-visit data
# ('first_visit' is added automatically when only 'follow_up_visit' is given)
data_generator = DataGenerator(task=['first_visit', 'follow_up_visit'], care_level='tertiary')


# 2. Synthesizing Data
# 2.1. Default usage
data_generator.build()

# 2.2. When you want the synthesized data returned along with its FHIR-converted version (optional)
data_generator.build(convert_to_fhir=True)

# 2.3. Using Custom Hospital Data
# (For the expected file format, see the examples below.)
data_generator.build(
    department_info_path="department_info.json",
    symptom_file_path="disease_symptom_pair.json",
)

# 2.4. When you want to upload the synthesized data to your own FHIR server (optional)
# Provide your FHIR server URL
data_generator.upload_to_fhir(
    fhir_data_dir=data_generator.save_dir / "fhir_data",
    fhir_url=${FHIR_URL},
)       
```
<details>
<summary>Configuration example for data synthesis</summary>

```yaml
# Base
seed: 9999

# FHIR server url
fhir_url: http://localhost:8080/fhir    # Optional: set your FHIR server URL here

# Data configs
project: ./synthetic_data/
data_name: hospital_small    # Output path: ./synthetic_data/hospital_small/data
hospital_data:
    hospital_n: 10           # Number of hospitals to synthesize
    start_date:
        min: 2025-03-17      # ISO format: YYYY-MM-DD
        max: 2025-09-21      
    days: 7                  # Simulation period (in days)
    interval_hour: 0.25      # Time unit expressed in hours
    start_hour:              # Possible hospital opening hours
        min: 9
        max: 10
    end_hour:                # Possible hospital closing hours
        min: 18
        max: 19
    department_per_hospital:
        min: 7
        max: 9
    doctor_per_department:
        min: 1
        max: 1
    working_days:                   # Number of days each doctor works during the simulation period
        min: 5
        max: 7
    doctor_capacity_per_hour:
        min: 1
        max: 4
    doctor_has_schedule_prob: 0     # Probability that a doctor has at least one fixed schedule
    schedule_coverage_ratio:        # Proportion of fixed schedules relative to total working hours
        min: 0.4
        max: 0.6
    appointment_coverage_ratio:   # Proportion of appointments scheduled outside fixed schedules
        min: 0.2
        max: 0.5
    first_visit:
        preference:
            type: ['asap', 'doctor', 'date']    # Types of patient scheduling preferences
            probs: [0.4, 0.4, 0.2]              # Probability distribution for each preference type
        symptom:
            type: ['simple', 'with_history']    # 'simple' = no referral; 'with_history' = referral case
            probs: [0.7, 0.3]                   # Probability distribution for symptom types
    follow_up_visit:                            # Required only when `follow_up_visit` is in the DataGenerator task
        doctor_schedule_extra_days_before: 7    # Clinic days added before `start_date` to hold each patient's prior first visit (must exceed `simulation_start_day_before` of the Simulator)
        doctor_schedule_extra_days_after: 7     # Clinic days added after the period so a late follow-up consultation still has a clinic day
        n_machines_per_test:                    # Number of devices per test
            min: 1
            max: 2
        test_per_department:                    # Number of diagnostic tests per department
            min: 2
            max: 4
        test_has_schedule_prob: 0.5             # Probability that a test has at least one fixed schedule
        test_fixed_schedule_ratio:              # Proportion of a test's fixed schedules relative to total working hours
            min: 0.05
            max: 0.1
        test_day_closed_prob: 0.3               # Probability that a test is closed on a given day
        test_per_patient:                       # Number of tests each follow-up patient needs
            trial: 20                           # Max attempts to sample a feasible test combination
            min: 2
            max: 3
        test_schedule_appointment_ratio:        # Proportion of a test's vacant slots pre-filled with other appointments
            min: 0.2
            max: 0.4
        cross_department_test_prob: 0.2         # Probability of tests from other departments
        include_consultation: True              # Whether to include a post-test doctor consultation
        preference:
            type: ['throughput_max', 'visit_min', 'stay_min', 'indifferent']    # Types of follow-up patient preferences
```
</details>
<details>
<summary>Custom data examples</summary>

* `department_info.json`
>```json
>{   
>    "metadata": {
>        "reference": "https://emergency.or.kr/bbs/general-notice/2513"
>    },
>    "specialty": {
>        "internal medicine": {
>            "code": "IM",
>            "subspecialty": {
>                "gastroenterology": {
>                    "code": "IMGAS",
>                    "field": [
>                        "Hepatology",
>                        "Pancreatobiliary Medicine",
>                        "Inflammatory Bowel Disease"
>                    ]
>                },
>                "cardiology": {
>                    "code": "IMCAR",
>                    "field": [
>                        "Interventional Cardiology",
>                        "Electrophysiology",
>                        "Heart Failure and Transplantation"
>                    ]
>                }
>            }
>        
>        }
>    }
>}
>```
* `disease_symptom_pair.json`
>It is recommended that this file contains disease-symptom pairs corresponding to the subspecialties defined in `department_info.json`.

>```json
>{
>    "gastroenterology": [
>        {
>            "Acute cholecystitis": {
>                "department": [
>                    "gastroenterology"
>                ],
>                "symptom": [
>                    "Sudden sharp pain in the upper right side of the abdomen that spreads towards the right shoulder",
>                    "Tender abdomen",
>                    "Worsening pain with deep breathing",
>                    "Persistent pain",
>                    "High temperature (fever)",
>                    "Nausea and vomiting",
>                    "Sweating",
>                    "Loss of appetite",
>                    "Yellowing of the skin and whites of the eyes (jaundice)",
>                    "Bulge in the abdomen"
>                ]
>            }
>        }
>    ],
>    "cardiology": [
>        {
>            "Abdominal aortic aneurysm": {
>                "department": [
>                    "cardiology"
>                ],
>                "symptom": [
>                    "Pulsating feeling in the abdomen, usually near the belly button",
>                    "Persistent back pain",
>                    "Persistent abdominal pain",
>                    "Sudden and severe pain in the abdomen (if ruptured)",
>                    "Pain radiating down into the scrotum (in men, if ruptured)",
>                    "Dizziness (if ruptured)",
>                    "Sweaty and clammy skin (if ruptured)",
>                    "Rapid heartbeat (tachycardia, if ruptured)",
>                    "Shortness of breath (if ruptured)",
>                    "Feeling faint (if ruptured)",
>                    "Loss of consciousness (if ruptured)"
>                ]
>            }
>        }
>    ]
>}
>```
</details>


&nbsp;

### 2. Administrative Staff MAS
Every task gets its staff agent from a shared `HospitalMAS`. Build the MAS first, and include a staff agent for each task you plan to run.
```python
from h_adminsim.pipeline import HospitalMAS
from h_adminsim.utils.mas_utils import init_mas_system

# 2.1. Default usage
# Pass a model only for the tasks you run; tasks set to None get no staff agent
mas_structure = init_mas_system(
    orchestrator_model='gpt-5-mini',            # Routes each patient message to a staff agent
    orchestrator_vllm_endpoint=None,
    first_visit_intake_model='gpt-5-mini',
    first_visit_scheduling_model='gemini-2.5-flash',
    follow_up_visit_scheduling_model='gpt-5-mini',
    intake_max_inference=5,
)
staff_mas = HospitalMAS(mas_structure)
##############################################################

# 2.2. Advanced usage: vLLM
mas_structure = init_mas_system(
    orchestrator_model='meta-llama/Llama-3.3-70B-Instruct',
    orchestrator_vllm_endpoint='http://0.0.0.0:8000',
    first_visit_intake_model='meta-llama/Llama-3.3-70B-Instruct',
    first_visit_intake_vllm_endpoint='http://0.0.0.0:8000',
    first_visit_scheduling_model='meta-llama/Llama-3.3-70B-Instruct',
    first_visit_scheduling_vllm_endpoint='http://0.0.0.0:8000',
)
staff_mas = HospitalMAS(mas_structure)
##############################################################
```
<details>
<summary>Building a custom MAS structure</summary>

`init_mas_system` returns a nested dictionary like the one below. You can build it yourself to use your own agents or descriptions. It must have exactly one root (the orchestrator), and each child node must be named after the task it serves.

```python
from h_adminsim.agent import OrchestratorAgent, IntakeAdminStaffAgent, SchedulingAdminStaffAgent
from h_adminsim.pipeline import HospitalMAS
from h_adminsim.registry import AGENT_DESCRIPTION

mas_structure = {
    'orchestrator': {
        'agent': OrchestratorAgent(model='gpt-5-mini'),
        'description': AGENT_DESCRIPTION['orchestrator'],
        'subagent': {
            'first_visit_intake': {
                'agent': IntakeAdminStaffAgent(model='gpt-5-mini', department_list=None),  # Departments are set per hospital at runtime
                'description': AGENT_DESCRIPTION['first_visit_intake'],
                'next_step': 'first_visit_scheduling',  # Hand over to scheduling once intake is done
            },
            'first_visit_scheduling': {
                'agent': SchedulingAdminStaffAgent(target_task='first_visit_scheduling', model='gpt-5-mini'),
                'description': AGENT_DESCRIPTION['first_visit_scheduling'],
            },
            'follow_up_visit_scheduling': {
                'agent': SchedulingAdminStaffAgent(target_task='follow_up_visit_scheduling', model='gpt-5-mini'),
                'description': AGENT_DESCRIPTION['follow_up_visit_scheduling'],
            },
        },
    }
}
staff_mas = HospitalMAS(mas_structure)
```
</details>

&nbsp;

### 3. Task Initialization
#### 3.1. Patient Intake
```python
from h_adminsim.agent import SupervisorAgent
from h_adminsim.task.opfv_task import OutpatientFirstIntake

# 1. Patient Intake
# 1.1. Default usage (Staff-only)
intake_task = OutpatientFirstIntake(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    intake_max_inference=5,  # Default: up to 5 rounds (10 turns) of dialogue
)
##############################################################

# 1.2. Role separation
# Staff: dialogue handling, Supervisor: data collection and structuring
supervisor_agent = SupervisorAgent(
    target_task='first_visit_intake',
    model='gemini-2.5-flash',
    api_key=${YOUR_API_KEY},  # You may set the API key here instead of using a .env file
)
intake_task = OutpatientFirstIntake(
    patient_model='gemini-2.5-flash',
    admin_staff_mas=staff_mas,
    supervisor_agent=supervisor_agent,
    intake_max_inference=8,
)
##############################################################

# 1.3. Advanced usage: vLLM
# (Staff agents' vLLM endpoints are set in the MAS; see 2.2.)
supervisor_agent = SupervisorAgent(
    target_task='first_visit_intake',
    model='meta-llama/Llama-3.3-70B-Instruct',
    use_vllm=True,              # Use a vLLM-hosted model as the supervisor
    vllm_endpoint='http://0.0.0.0:8000',  # vLLM server endpoint
)
intake_task = OutpatientFirstIntake(
    patient_model='meta-llama/Llama-3.3-70B-Instruct',
    admin_staff_mas=staff_mas,
    supervisor_agent=supervisor_agent,
    intake_max_inference=5,
    patient_vllm_endpoint='http://0.0.0.0:8000',
)
##############################################################
```

&nbsp;

#### 3.2. First-visit Appointment Scheduling
```python
from h_adminsim.task.opfv_task import OutpatientFirstScheduling

# 2. First-visit Appointment Scheduling
# 2.1. Default usage (Tool-calling with reasoning fallbacks)
fv_scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    schedule_cancellation_prob=0.05,        # Cancellation event
    request_early_schedule_prob=0.1,        # Rescheduling event
    preference_rejection_prob=0.3,          # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay=0.5,    # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='tool_calling',
    fhir_integration=False,
)
##############################################################

# 2.2. LLM reasoning-based scheduling without tool-calling
fv_scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    scheduling_strategy='reasoning',
    fhir_integration=False,
)
##############################################################

# 2.3. HIS upload via FHIR
fv_scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    scheduling_strategy='tool_calling',
    fhir_integration=True,
)
##############################################################

# 2.4. Advanced usage: vLLM patient
fv_scheduling_task = OutpatientFirstScheduling(
    patient_model='meta-llama/Llama-3.3-70B-Instruct',
    admin_staff_mas=staff_mas,
    scheduling_strategy='tool_calling',
    patient_vllm_endpoint='http://0.0.0.0:8000',
)
##############################################################
```

&nbsp;

#### 3.3. Follow-up Visit Scheduling
Follow-up scheduling needs data built with `DataGenerator(task=['first_visit', 'follow_up_visit'])`.
```python
from h_adminsim.task.opfu_task import OutpatientFollowUpScheduling

# 3. Follow-up Visit (Test and Consultation) Scheduling
# 3.1. Default usage (no staff negotiation: the staff follows the patient's preference)
fu_scheduling_task = OutpatientFollowUpScheduling(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    schedule_cancellation_prob=0.05,        # Cancellation event
    request_early_schedule_prob=0.075,      # Rescheduling event
    preference_rejection_prob=0.3,          # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay=0.5,    # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='tool_calling',     # ['tool_calling', 'reasoning']
    fhir_integration=False,
)
##############################################################

# 3.2. Staff negotiation and urgent capacity protection
negotiation_params = {
    'type': 'result_gain',                  # Negotiation policy type
    'mode': 'threshold',                    # ['threshold', 'hospital-side', 'patient-side']
    'tcl_temperature': 1.0,
    'trigger_temperature_visit': 1.0,       # τ for `visit_min` patients
    'trigger_temperature_stay': 1.0,        # τ for `stay_min` patients
    'negotiation_trigger_threshold': 1.0,   # Higher = stricter, fewer negotiations (`threshold` mode only)
    'capacity_protection': {                # Hold back part of each test's capacity for `urgent` patients
        'enabled': True,
        'ratio': 0.2,                       # Share of each (test, date) capacity reserved, in [0, 1)
        'horizon_days': 3,                  # Protect only dates opening within this many days
        'release_hours': 24,                # Release a protected date to normal patients within this many hours
    },
}
fu_scheduling_task = OutpatientFollowUpScheduling(
    patient_model='gpt-5-nano',
    admin_staff_mas=staff_mas,
    negotiation_params=negotiation_params,
)
##############################################################
```
<details>
<summary>Negotiation modes</summary>

* `patient-side`: The staff never negotiates and always follows the patient's preference. This is the default when `negotiation_params=None`.
* `hospital-side`: The staff negotiates whenever the hospital has anything to gain.
* `threshold`: The staff negotiates once the trigger index reaches `negotiation_trigger_threshold`.

The policy used (type, mode, and every parameter) is saved under `_metadata` in each result file, so a run can be reproduced from its results.
</details>

&nbsp;

### 4. Simulation
`Simulator` takes a dictionary of tasks keyed by task name. Any combination works, with one rule: when you run intake together with follow-up scheduling, `first_visit_scheduling` must be included too.
```python
from h_adminsim.pipeline import Simulator

# 4. Simulator initialization
# 4.1. Default usage (end-to-end: first-visit intake -> first-visit scheduling -> follow-up scheduling)
# A follow-up patient is scheduled only after their first-visit appointment has finished in simulated time.
simulator = Simulator(
    task={
        'first_visit_intake': intake_task,
        'first_visit_scheduling': fv_scheduling_task,
        'follow_up_visit_scheduling': fu_scheduling_task,
    },
    simulation_start_day_before=3,
    fhir_integration=False,
    fhir_url=None,
    fhir_max_connection_retries=5,
    random_seed=9999,
)
##############################################################

# 4.2. FHIR integration
# (If enabled, every scheduling task must be initialized with `fhir_integration=True`)
simulator = Simulator(
    task={
        'first_visit_intake': intake_task,
        'first_visit_scheduling': fv_scheduling_task,
    },
    simulation_start_day_before=3,
    fhir_integration=True,
    fhir_url='http://localhost:8080/fhir',
    fhir_max_connection_retries=5,
    random_seed=9999,
)
##############################################################

# 4.3. Running a subset of tasks
# 4.3.1. First-visit tasks only
simulator = Simulator(
    task={
        'first_visit_intake': intake_task,
        'first_visit_scheduling': fv_scheduling_task,
    },
)

# 4.3.2. Follow-up scheduling only
# Each patient's prior first visit is created automatically as a completed past appointment.
simulator = Simulator(
    task={'follow_up_visit_scheduling': fu_scheduling_task},
)
##############################################################
```
```python
# Run the initialized simulator
simulator.run(
    simulation_data_path='hospital_data/primary/agent_data',
    output_dir='hospital_data/primary/simulation_results',
    resume=False,   # If the simulation stopped unexpectedly, set resume=True with the same paths
    verbose=True,
)
```

&nbsp;

### 5. Evaluation
```python
from h_adminsim.pipeline import Evaluator

# 5.1. First-visit evaluation (one results directory)
evaluator = Evaluator(
    fv_path='hospital_data/primary/simulation_results',
    model_name='gpt-5-mini',    # Required only for `fv_token`
)
evaluator.evaluate([
    'fv_task',          # Accuracy of each first-visit task
    'fv_department',    # Department recommendation accuracy
    'fv_rounds',        # Average number of intake dialogue rounds
    'fv_token',         # API cost from the token usages
])
##############################################################

# 5.2. Follow-up evaluation (compares runs on the same data with different negotiation settings)
evaluator = Evaluator(
    fu_hospital_side_path='results/hospital_side',      # Run with `mode: hospital-side`
    fu_patient_side_path='results/patient_side',        # Run with `mode: patient-side`
    fu_interpolated_paths=[                             # Runs with `mode: threshold`, in the order to show
        'results/threshold_1',
        'results/threshold_2',
    ],
    fu_protection_paths=[                               # Runs with and without capacity protection
        'results/hospital_side',
        'results/hospital_side_protection_0.2',
    ],
    fu_agent_data_path='hospital_data/primary/agent_data',  # Agent data the runs used (for `fu_protection`); default: `<results dir>/../agent_data`
    model_name='gpt-5-mini',
)
outputs = evaluator.evaluate(
    ['fu_tau', 'fu_dashboard', 'fu_protection'],
    save=True,      # Save the dashboard and protection results as files
    tau_n=4,        # Number of τ seeds per preference for `fu_tau`
)
taus = outputs['fu_tau']    # τ values to use for the `threshold` runs
##############################################################
```
<details>
<summary>Follow-up evaluation types</summary>

* `fu_tau`: Computes τ (`trigger_temperature_visit`, `trigger_temperature_stay`) values from the `hospital-side` run, to use for the `threshold` runs.
* `fu_dashboard`: Compares the negotiation results across the `hospital-side`, `threshold`, and `patient-side` runs.
* `fu_protection`: Compares the effect of urgent capacity protection across `fu_protection_paths`. Each run's mode and protection ratio are read from its `_metadata`.

All runs must share the same hospital operating hours and time unit.
</details>

&nbsp;

### Migrating from v1.x
| v1.x | v2.0.0 |
|---|---|
| `OutpatientFirstIntake(admin_staff_model=...)` | `OutpatientFirstIntake(admin_staff_mas=HospitalMAS(...))` |
| `OutpatientFirstScheduling(admin_staff_model=...)` | `OutpatientFirstScheduling(admin_staff_mas=HospitalMAS(...))` |
| `admin_staff_vllm_endpoint=...` on each task | `*_vllm_endpoint=...` in `init_mas_system(...)` |
| `Simulator(intake_task=..., scheduling_task=...)` | `Simulator(task={'first_visit_intake': ..., 'first_visit_scheduling': ...})` |
| `hospital_data.preference` / `hospital_data.symptom` in the data config | `hospital_data.first_visit.preference` / `hospital_data.first_visit.symptom` |
| `from h_adminsim import SupervisorAgent` | `from h_adminsim.agent import SupervisorAgent` |
| `Evaluator(path).evaluate(['task', 'department', 'rounds'])` | `Evaluator(fv_path=path).evaluate(['fv_task', 'fv_department', 'fv_rounds'])` |


&nbsp;

&nbsp;



---
## License
This project is licensed under the Apache License 2.0.

&nbsp;

&nbsp;


---
## Citation
For `H-AdminSim` and `PatientSim` outpatient simulation, please cite the following.
```
@inproceedings{lee2026hadminsimmultiagentsimulatorrealistic,
  title         = {H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration}, 
  author        = {Jun-Min Lee and Meong Hi Son and Edward Choi},
  booktitle     = {Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  year          = {2026},
  eprint        = {2602.05407},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2602.05407}, 
}
```


<!-- 

## More Details on Module Tests 🔍
1. [Getting Started](https://github.com/ljm565/H-AdminSim/blob/master/docs/1_getting_started.md)
2. [How to perform CRUD on FHIR](https://github.com/ljm565/H-AdminSim/blob/master/docs/2_fhir_crud.md)
3. [Hospital Data Synthesis](https://github.com/ljm565/H-AdminSim/blob/master/docs/3_data_synthesis.md)
4. [Agent Simulation](https://github.com/ljm565/H-AdminSim/blob/master/docs/4_agent_test.md) -->

&nbsp;
