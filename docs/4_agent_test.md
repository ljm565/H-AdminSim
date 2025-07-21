# Agent test
We introduce how to test the capabilities required of agents in a hospital booking system.
To begin, you should first build the agent test data by following [this guideline 2.2](./3_data_synthesis.md).
<br><br>We support [Gemini-based LLMs](https://ai.google.dev/gemini-api/docs/models) and [GPT-based LLMs](https://platform.openai.com/docs/pricing).

&nbsp;

## Agent Test Guides
### 0. Preliminary
We need to complete the `config/agent_test.yaml` file.
```yaml
# LLM model
model: gpt-4 # gemini-2.5-flash-preview-05-20

# Agent test data and converted FHIR data folder
agent_test_data: synthetic_data/hospital_easy_small/agent_data
fhir_data: synthetic_data/hospital_easy_small/fhir_data
fhir_url: http://localhost:8080/fhir

# Prompt paths
department_task:
    system_prompt: asset/prompts/department_system.txt
    user_prompt: asset/prompts/department_user.txt
schedule_task:
    system_prompt: asset/prompts/schedule_system.txt
    user_prompt: asset/prompts/schedule_user.txt
fhir_resource_task:
    system_prompt: asset/prompts/fhir_resource_system.txt
    user_prompt: asset/prompts/fhir_resource_user.txt
fhir_api_task:
    system_prompt: asset/prompts/fhir_api_system.txt
    user_prompt: asset/prompts/fhir_api_user.txt
```
> * `model`: The model to be used as the agent.
> * `agent_test_data`: Path to the pre-built agent test data folder.
> * `fhir_data`: Path to the folder containing pre-converted FHIR data.
> * `fhir_url`: The base URL of the FHIR server.
> * `*prompt`: Paths to the prompt data.


&nbsp;

### 1. Agent Tasks Execution
#### 1.1 Overview
We evaluate four core capabilities of an agent in the hospital booking system:
> 1. `department`: Task to assign the appropriate medical department based on the patient's symptoms.
> 2. `schedule`: After the `department` task, this task assigns a attending physician and schedules an appointment based on the doctor’s availability and the patient’s required duration.
> 3. `fhir_resource`: Task to generate a FHIR Appointment resource based on the scheduling results.
> 4. `fhir_api`: Task to generate a `curl` command that creates the FHIR resource on the FHIR server.


<br>These tasks are sequentially dependent.
For example, if the `department` prediction fails, the subsequent tasks cannot be completed properly.
Likewise, even if the `schedule` task succeeds, a failure in the `fhir_resource` task will cause the `fhir_api` task to fail as well.

Therefore, this framework allows for both individual evaluation of each capability and sequential evaluation of the entire pipeline.
When evaluating the `fhir_resource` task individually, it is performed under the assumption that both the `department` and `schedule` tasks have produced correct outputs.
In contrast, when evaluating the entire process sequentially, the result of each previous task directly affects the outcome of the current task.

&nbsp;

#### 1.2 Execution
You can execute the agent tasks using the below commands:
```bash
# Evaluating only the `department` capability
python3 -u src/run/agent_simulate.py --config config/agent_simulate.yaml --type department --output_dir ${SYNTHETIC_DATA_FOLDER}/agent_results

# Evaluating only the `fhir_resource` capability
python3 -u src/run/agent_simulate.py --config config/agent_simulate.yaml --type fhir_resource --output_dir ${SYNTHETIC_DATA_FOLDER}/agent_results

# Sequentially evaluating the `department` and `schedule` tasks
python3 -u src/run/agent_simulate.py --config config/agent_simulate.yaml --type department schedule --output_dir ${SYNTHETIC_DATA_FOLDER}/agent_results

# Sequentially evaluating all tasks
python3 -u src/run/agent_simulate.py --config config/agent_simulate.yaml --type department schedule fhir_resource fhir_api --output_dir ${SYNTHETIC_DATA_FOLDER}/agent_results
```

&nbsp;

#### 1.3 Evaluation
You can evaluate the agent task results using the below command:
```bash
python3 -u src/run/evaluate.py --path ${SYNTHETIC_DATA_FOLDER}/agent_results
```

&nbsp;