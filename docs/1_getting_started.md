# Getting Started
Here, we configure the environment first.

&nbsp;

## Docker
### 1. Image Build
You can build a Docker image using the following command:
```bash
cd docker
docker build -t ${IMAGE_NAME} .
```
&nbsp;

### 2. Run Docker Container
Then, you can make a Docker container using the following command:
```bash
docker run -it -d --name ${CONTAINER_NAME} --gpus all --shm-size=2g -v ${LOCAL_REPOSITORY_PATH}:${MOUNT_PATH} -v ${PATH_OF_HUGGINGFACE_HUB_CACHE_FOLDER}:/root/.cache/ ${IMAGE_NAME}

docker exec -it ${CONTAINER_NAME} /bin/bash
```

If you want to prevent auto-kill from your server, please add the below option:
```bash
--oom-score-adj=-1000
```

&nbsp;

&nbsp;


## Anaconda (Deprecated)
### 0. Preliminary
It is assumed that the Conda environment and Python and PyTorch related libaray are all installed.
PyTorch version 2.7.1 or higher is recommended.
```bash
# torch install example
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

### 1. Package Installation
You can install packages using the following command:
```bash
cd docker
pip3 install -r requirements.txt
```

&nbsp;

&nbsp;


## Poetry
### 1. Package Installation
It is assumed that Poetry has already been installed in your environment.
```bash
poetry install
poetry shell
```

&nbsp;