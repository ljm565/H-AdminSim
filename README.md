# H-AdminSim

<!-- ---
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/patientsim)
![PyPI Version](https://img.shields.io/pypi/v/patientsim)
![Downloads](https://img.shields.io/pypi/dm/patientsim)
![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2505.17818-blue)
--- -->

&nbsp;

&nbsp;

## Overview 📚
H-AdminSim is an official Python package for simulating interactions between hospital administrative staff and first-visit outpatients using LLM agents.
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


#### 2.3. FHIR Integration
We provide optional support for integrating with FHIR, allowing the simulator to operate flexibly across heterogeneous hospital environments as long as FHIR-compatible data is available. For instructions on running a FHIR server, please refer to the [FHIR Server Execution](https://github.com/ljm565/fhir_server) repository.

&nbsp;

&nbsp;

## Recent updates 📣
* *December 2025 (v1.0.0)*: H-AdminSim package has been released.
* *December 2025 (v0.7.2)*: Rule-based and tool calling-based scheduling logics have been supported.
* *November 2025 (v0.7.1)*: Self-corrective feedback logic has been supported.
* *October 2025 (v0.7.0)*: Simulation has been improved reflecting feedbacks from experts.
* *September 2025 (v0.6.0)*: Simulation has been improved reflecting feedbacks from experts.
* *August 2025 (v0.5.2)*: We has supported vLLM inference of the Hugging Face models.
* *August 2025 (v0.5.1)*: Now you can easily set the virtual environment using Poetry.
* *August 2025 (v0.5.0)*: We integrated the FHIR server to retrieve hospital information during hospital administration office agent simulation.
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
GOOGLE_API_KEY=${YOUR_GEMINI_API_KEY"}
```

&nbsp;

### 3. Simulation
```python
from h_adminsim import AdminStaffAgent, SupervisorAgent
from h_adminsim.pipeline import DataGenerator, Simulator
from h_adminsim.task.agent_task import OutpatientFirstIntake, OutpatientFirstScheduling

data_generator = DataGenerator()
data_generator.build(convert_to_fhir=True)
agent_data_dir = data_generator.save_dir / 'agent_data'
output_dir = data_generator.save_dir / 'simulation_results'

# Intake task
intake_task = OutpatientFirstIntake(
    patient_model='gpt-5-nano',
    admin_staff_model='gemini-2.5-flash',
)

# Scheduling task
scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_model='gpt-5-mini',
)

# Simulation
simulator = Simulator(
    intake_task=intake_task,
    scheduling_task=scheduling_task,
)
simulator.run(
    simulation_data_path=agent_data_dir,
    output_dir=output_dir,
    resume=False,
    verbose=True
)
```

&nbsp;

&nbsp;

## Components Details ⚙️
### 1. Data synthesis
```python
from h_adminsim.pipeline import DataGenerator

# 1. Generator Initialization
# 1.1. Default usaage
data_generator = DataGenerator()    # Default: primary care
# data_generator = DateGenerator(level='secondary') # For secondary care
# data_generator = DateGenerator(level='tertiary')  # For tertiary care

# 1.2. You can synthesize data with your own configuration
data_generator = DataGenerator(config='data_config.yaml')


# 2. Synthesizing Data
# 2.1. Default usage
data_generator.build()

# 2.2. When you want the synthesized data returned along with its FHIR-converted version (optional)
data_generator.build(convert_to_fhir=True)

# 2.3. When you want to upload the synthesized data to your own FHIR server (optional)
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
        min: 3
        max: 4
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
    preference:
        type: ['asap', 'doctor', 'date']    # Types of patient scheduling preferences
        probs: [0.4, 0.4, 0.2]              # Probability distribution for each preference type
    symptom:
        type: ['simple', 'with_history']    # 'simple' = no referral; 'with_history' = referral case
        probs: [0.7, 0.3]                   # Probability distribution for symptom types
```
</details>

&nbsp;

### 2. Task Initialization
#### 2.1. Patient Intake
```python
from h_adminsim import SupervisorAgent
from h_adminsim.task.agent_task import OutpatientFirstIntake

# 1. Patient Intake
# 1.1. Default usage (Staff-only)
intake_task = OutpatientFirstIntake(
    patient_model='gpt-5-nano',
    admin_staff_model='gpt-5-mini',
    intake_max_inference=5,  # Default: up to 5 rounds (10 turns) of dialogue
)
##############################################################

# 1.2. Role separation
# Staff: dialogue handling, Supervisor: data collection and structuring
supervisor_agent = SupervisorAgent(
    target_task='first_outpatient_scheduling',
    model='gemini-2.5-flash',
    api_key=${YOUR_API_KEY},  # You may set the API key here instead of using a .env file
)
intake_task = OutpatientFirstIntake(
    patient_model='gemini-2.5-flash',
    admin_staff_model='gpt-5',
    supervisor_agent=supervisor_agent,
    intake_max_inference=8,
)
##############################################################

# 1.3. Advanced usage: vLLM
supervisor_agent = SupervisorAgent(
    target_task='first_outpatient_scheduling',
    model='meta-llama/Llama-3.3-70B-Instruct',
    use_vllm=True,              # Use a vLLM-hosted model as the supervisor
    vllm_endpoint='http://0.0.0.0:8000',  # vLLM server endpoint
)
intake_task = OutpatientFirstIntake(
    patient_model='meta-llama/Llama-3.3-70B-Instruct',
    admin_staff_model='meta-llama/Llama-3.3-70B-Instruct',
    supervisor_agent=supervisor_agent,
    intake_max_inference=5,
    patient_vllm_endpoint='http://0.0.0.0:8000',
    admin_staff_vllm_endpoint='http://0.0.0.0:8000',
)
##############################################################
```

&nbsp;

#### 2.2. Appointment Scheduling
```python
from h_adminsim import AdminStaffAgent, SupervisorAgent
from h_adminsim.task.agent_task import OutpatientFirstScheduling

# 2. Appointment Scheduling
# 2.1. Default usage (Tool-calling with reasoning fallbacks)
scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_model='gemini-2.5-flash',
    schedule_cancellation_prob=0.05,    # Cancellation event
    request_early_schedule_prob=0.1,    # Rescheduling event
    preference_rejection_prob = 0.3,        # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay = 0.5,  # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='tool_calling',
    fhir_integration=False,
)
##############################################################

# 2.2. LLM reasoning-based scheduling without tool-calling
scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_model='gpt-5-mini',
    schedule_cancellation_prob=0.05,    # Cancellation event
    request_early_schedule_prob=0.1,    # Rescheduling event
    preference_rejection_prob = 0.3,        # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay = 0.5,  # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='reasoning',
    fhir_integration=False,
)
##############################################################

# 2.3. HIS upload via FHIR
scheduling_task = OutpatientFirstScheduling(
    patient_model='gpt-5-nano',
    admin_staff_model='gemini-2.5-flash',
    schedule_cancellation_prob=0.05,    # Cancellation event
    request_early_schedule_prob=0.1,    # Rescheduling event
    preference_rejection_prob = 0.3,        # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay = 0.5,  # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='tool_calling',
    fhir_integration=True,
)
##############################################################

# 2.4. Advanced usage: vLLM
scheduling_task = OutpatientFirstScheduling(
    patient_model='meta-llama/Llama-3.3-70B-Instruct',
    admin_staff_model='gpt-5-mini',
    schedule_cancellation_prob=0.05,    # Cancellation event
    request_early_schedule_prob=0.1,    # Rescheduling event
    preference_rejection_prob = 0.3,        # Prob. of rejecting the first-priority scheduling preference
    preference_rejection_prob_decay = 0.5,  # Decay factor for the preference rejection prob.
    scheduling_max_inference=5,
    scheduling_strategy='tool-calling',    # Currently, we do not support tool-calling from vLLM
    fhir_integration=False,
    patient_vllm_endpoint='http://0.0.0.0:8000',
    
)
##############################################################
```

&nbsp;

### 3. Simulation
```python
from h_adminsim.pipeline import Simulator

# 3. Simulator initialization
# 3.1. Default usage
simulator = Simulator(
    intake_task=intake_task,
    scheduling_task=scheduling_task,
    simulation_start_day_before=3,
    fhir_integration=False,      
    fhir_url=None,
    fhir_max_connection_retries=5,
    random_seed=9999,
)
##############################################################

# 3.2. FHIR integration 
# (If enabled, scheduling task must be initialized with `fhir_integration=True`)
simulator = Simulator(
    intake_task=intake_task,
    scheduling_task=scheduling_task,
    simulation_start_day_before=3,
    fhir_integration=True,
    fhir_url='http://localhost:8080/fhir',
    fhir_max_connection_retries=5,
    random_seed=9999,
)
##############################################################


# 3.3. Running a single task
# 3.3.1. Intake task only
simulator = Simulator(
    intake_task=intake_task,
    scheduling_task=None,
    simulation_start_day_before=3,
    fhir_integration=False,
    fhir_url=None,
    fhir_max_connection_retries=5,
    random_seed=9999,
)

# 3.3.2. Scheduling task only
simulator = Simulator(
    intake_task=None,
    scheduling_task=scheduling_task,
    simulation_start_day_before=3,
    fhir_integration=False,
    fhir_url=None,
    fhir_max_connection_retries=5,
    random_seed=9999,
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

&nbsp;

## Citation
For `H-AdminSim` and `PatientSim` outpatient simulation, please cite the following.
```
@misc{lee2026hadminsimmultiagentsimulatorrealistic,
      title={H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration}, 
      author={Jun-Min Lee and Meong Hi Son and Edward Choi},
      year={2026},
      eprint={2602.05407},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.05407}, 
}
```


<!-- 

## More Details on Module Tests 🔍
1. [Getting Started](https://github.com/ljm565/H-AdminSim/blob/master/docs/1_getting_started.md)
2. [How to perform CRUD on FHIR](https://github.com/ljm565/H-AdminSim/blob/master/docs/2_fhir_crud.md)
3. [Hospital Data Synthesis](https://github.com/ljm565/H-AdminSim/blob/master/docs/3_data_synthesis.md)
4. [Agent Simulation](https://github.com/ljm565/H-AdminSim/blob/master/docs/4_agent_test.md) -->

&nbsp;
