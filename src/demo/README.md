# Demo Operation
Herein, we introduce chat demo operation steps.

&nbsp;

## LLM Agent Server Operation
### 1. Complete `.env` File
You should complete `.env` file in the main project directory.
The below is the example:
```bash
OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}
GOOGLE_API_KEY=${YOUR_GCP_API_KEY}
HF_TOKEN=${YOUR_HF_TOKEN}
```

&nbsp;

### 2. Complete Server Configuration
You should complete `config/llm_server.yaml` to operate the LLM server:
```yaml
# Uvicorn
host: 127.0.0.1
port: 8501

# LLM model
model: gemini-2.5-flash-preview-05-20
```
* The default host and port are `localhost` and `8501`, respectively.
* All GCP Gemini models include the word 'gemini' in their names, so the code can automatically recognize whether the model is from GCP or OpenAI.

&nbsp;

### 3. Running the Server using FastAPI and Uvicorn
Now, you can operate the LLM server.
The server will communicate via FastAPI and managed by Uvicorn.
You can execute the sever using the following command:
```bash
# Base execution
python3 src/demo/llm_server.py --config config/llm_server.yaml

# You can use automated Uvicorn reload option
python3 src/demo/llm_server.py --config config/llm_server.yaml --is_develop
```
* `--is_develop`: If you want to use automated Uvicorn reload option, you should use this option.

&nbsp;

&nbsp;


## Frontend Operation
### Using streamlit
You can run frontend via streamlit:
```bash
# Base execution
streamlit run src/demo/front.py

# Using an other port
streamlit run src/demo/front.py --server.port 20000
```