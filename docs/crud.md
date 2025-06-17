# FHIR CRUD
Here, we introduce how to  perform CRUD operations on FHIR.

&nbsp;

## CRUD
### 0. Preliminary
We assume that the FHIR server is already running.
After that, we need to complete the `config/crud.yaml` file.
```yaml
# Base
seed: 9999

# FHIR server url
fhir_url: http://localhost:8080/fhir

# Data configs
create_data_path: ./data  #  Path of data to create. We can set it to individual JSON files or a directory containing multiple JSON files.
```
> * `create_data_path`: This argument is for the "create" operation. It can be set to a folder containing JSON files or to an individual JSON path.

&nbsp;

### 1. Create
You can perform create operation using the following command:
```bash
python3 src/run/crud.yaml --config config/crud.yaml --mode create
```

&nbsp;

### 2. Read
You can perform read operation using the following command:
```bash
python3 src/run/crud.yaml --config config/crud.yaml --mode read --resource_type ${RESOURCE_TYPE} --id ${ID}
```
> * `${RESOURCE_TYPE}`: Specify the type of FHIR resource you want to read (e.g., `Patient`, `Schedule`, etc.).
> * `${ID}`: Specify the unique identifier of the resource you want to read. This can be a UUID or any other unique string.


&nbsp;

### 3. Update
You can perform update operation using the following command:
```bash
python3 src/run/crud.yaml --config config/crud.yaml --mode update --resource_type ${RESOURCE_TYPE} --id ${ID} --update_data_path ${UPDATE_DATA_PATH}
```
> * `${RESOURCE_TYPE}`: Specify the type of FHIR resource you want to update (e.g., `Patient`, `Schedule`, etc.).
> * `${ID}`: Specify the unique identifier of the resource you want to update. This can be a UUID or any other unique string.
> * `${UPDATE_DATA_PATH}`: JSON data file that you want to update.

&nbsp;

### 4. Delete
You can perform delete operation using the following command:
```bash
python3 src/run/crud.yaml --config config/crud.yaml --mode delete --resource_type ${RESOURCE_TYPE} --id ${ID}
```
> * `${RESOURCE_TYPE}`: Specify the type of FHIR resource you want to delete (e.g., `Patient`, `Schedule`, etc.).
> * `${ID}`: Specify the unique identifier of the resource you want to delete. This can be a UUID or any other unique string.

&nbsp;