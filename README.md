# Reservation Agents


### Recent updates 📣
* *October 2025 (v0.7.0)*: Simulation has been improved reflecting feedbacks from experts.
* *September 2025 (v0.6.0)*: Simulation has been improved reflecting feedbacks from experts.
* *August 2025 (v0.5.2)*: We has supported vLLM inference of the Hugging Face models.
* *August 2025 (v0.5.1)*: Now you can easily set the virtual environment using Poetry.
* *August 2025 (v0.5.0)*: We integrated the FHIR server to retrieve hospital information during hospital administration office agent simulation.
* *July 2025 (v0.4.2)*: We have supported LangChain's JsonOutputParser funtion as well as naive LLM API methods.
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
* *June 2025 (v0.0.1)*: Created chat demo codes using FastAPI communication.

&nbsp;

&nbsp;



## Overview 📚
<!-- This repository is designed to make it easy for anyone to tune models available on Hugging Face.
When a new model is released, anyone can easily implement a model wrapper to perform instruction-tuning and fine-tuning.
For detailed usage instructions, please refer to the description below.
* Universal LLM trainer supports full-training.
* Universal LLM trainer supports LoRA fine-tuning.
* Universal LLM trainer supports QLoRA fine-tuning.
* Universal LLM trainer supports DDP and FSDP training strategies.-->

&nbsp;

&nbsp;



## Quick Starts 🚀
### Environment setup
We have to install PyTorch and other requirements. Please refer to more [detailed setup](./docs/1_getting_started.md) including Docker.
```bash
# PyTorch install
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Requirements install
pip3 install -r docker/requirements.txt
```

&nbsp;

### Synthesizing hospital data
```bash
python3 src/run/synthesize_data.py --config config/data_synthesis.yaml
```

&nbsp;

&nbsp;



## Tutorials & Documentations 🔍
1. [Getting Started](./docs/1_getting_started.md)
2. [How to perform CRUD on FHIR](./docs/2_fhir_crud.md)
3. [Hospital Data Synthesis](./docs/3_data_synthesis.md)
4. [Agent Simulation](./docs/4_agent_test.md)

&nbsp;
