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

### 2. Running the Backend Server using FastAPI and Uvicorn
Now, you can operate the LLM server.
The server will communicate via FastAPI and managed by Uvicorn.
You can execute the sever using the following command:
```bash
# Base execution
python3 demo/llm_server.py --model gpt-5-nano

# You can use automated Uvicorn reload option
python3 demo/llm_server.py --model gemini-2.5-flash --is_develop

# You can use your own address
python3 demo/llm_server.py --model gpt-5-mini --backend_host 124.12.33.12 --port 8732 --is_develop
```
* `--is_develop`: If you want to use automated Uvicorn reload option, you should use this option.
* `--backend_host`: If you want to use your own host, you can use this option. Defaults to `127.0.0.1`.
* `--backend_port`: If you want to use your own port, you can use this option. Defaults to `8778`.

&nbsp;

&nbsp;


## Frontend Operation
### Using streamlit
You can run frontend via streamlit:
```bash
# Base execution
streamlit run demo/front.py

# Using an other port
streamlit run demo/front.py --server.port 20000

# When you set the backend server with your own address
# Don't forget the `--` in the middle of the command
streamlit run demo/front.py --server.port 20000 -- --backend_host 124.12.33.12 --port 8732
```
* `--backend_host`: When you deploy backend server with your own host, you can use this option. Defaults to `127.0.0.1`.
* `--backend_port`: When you deploy backend server with your own port, you can use this option. Defaults to `8778`.