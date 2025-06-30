# Hospital Data Synthesis
Here, we introduce how to synthesize hospital data, including basic information about hospitals, departments, doctors, and patients.

&nbsp;

## Synthesizing Guides
### 0. Preliminary
We need to complete the `config/data_synthesis.yaml` file.
```yaml
# Base
seed: 9999

# FHIR server url
fhir_url: http://localhost:8080/fhir

# Data configs
project: ./synthetic_data/
data_name: hospital_easy
hospital_data:
    hospital_n: 10000
    interval_hour: 0.5
    start_hour:
        min: 6
        max: 8
    end_hour:
        min: 18
        max: 20
    department_per_hospital:
        min: 2
        max: 5    
    doctor_per_department:
        min: 1
        max: 5
    doctor_has_schedule_prob: 0.3   # The probability that a doctor has at least one fixed schedule.
    schedule_coverage_ratio:        # If the doctor has a fixed schedule, the proportion of that schedule relative to the total working hours.
        min: 0.01
        max: 0.2
    appointment_coverage_ratio:   # Proportion of appointment time scheduled with patients outside the doctor's fixed schedule.
        max_chunk_size: 4         # Maximum number of consecutive segments per appointment (e.g., max duration = interval_hour * max_chunk_size).
        min: 0.9
        max: 1.0

```
> * `project`, `data_name`: The generated data will be saved to the path `${project}/${data_name}`. This path is generated automatically, so you don't need to create it manually.
> * `hospital_n`: Number of hosptial data you want to generate.
> * `interval_hour`: The defualt time unit in the gnerated data. It is applied to time-related items such as schedules.
> * `start_hour`: Hospital opening time.
> * `end_hour`: Hospital closing time.
> * `department_per_hospital`: Number of departments in the hospital
> * `doctor_per_department`: Number of doctors in each department.
> * `doctor_has_schedule_prob`: Probability that doctors have at least one fixed schedule other than patient appointments.
> * `schedule_coverage_ratio`: When doctors have at least one fixed schedule, the proportion of their working hours occupied by these fixed schedules.
> * `appointment_coverage_ratio`: Among doctors' available hours excluding fixed schedules, the proportion allocated to patient appointments.
> * `max_chunk_size`: Maximum duration of an individual patient appointment (e.g., max duration = interval_hour * max_chunk_size).



&nbsp;

### 1. Synthesize Data
You can synthesize hospital data using the following command:
```bash
python3 src/run/synthesize_data.py --config config/data_synthesis.yaml

# f you want to check whether the generated data are compatible with the Hospital object, 
# you can use the --sanity_check option.
python3 src/run/synthesize_data.py --config config/data_synthesis.yaml --sanity_check
```

&nbsp;


### 2. Convert the Synthesized Data to FHIR Format
You can convert the synthesized JSON data to FHIR format.
> [!NOTE]
> If the data synthesis is completed, an `args.yaml` file will be generated in the synthesized data folder, and this file is required.
```bash
# Converting command example
python3 src/run/convert_to_fhir.py --config ${SYNTHETIC_DATA_FOLDER}/args.yaml --output_dir ${SYNTHETIC_DATA_FOLDER}/fhir_data
```
Supported resource types:
> - `Practitioner`
> - `Patient`
> - `Schedule`
> - `Slot`
> - `Appointment`

&nbsp;