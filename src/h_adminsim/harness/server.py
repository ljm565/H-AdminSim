"""
OpenAPI tool server exposing the policy harness to an LLM client (e.g. Open WebUI): every endpoint is one tool.

File access is confined to the policy directory (read and write) and the reference directories (read only), and files
are named by bare file names, so the LLM cannot reach other paths. Run it with the ``h-adminsim-policy-server`` command
(or ``python -m h_adminsim.harness.server``).
"""
import os
import argparse
from typing import Optional
from dotenv import load_dotenv, find_dotenv

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from h_adminsim.registry.harness import (
    HARNESS_DEFAULT_POLICY_DIR,
    HARNESS_CHAT_ID_HEADERS,
    HARNESS_SERVER_DESCRIPTION,
    HARNESS_TOOL_DESCRIPTIONS,
)
from h_adminsim.harness.report import Report
from h_adminsim.harness.schemas import (
    SaveRequest,
    ReadRequest,
    ApiSpecRequest,
    TemplateRequest,
    MetricTemplateRequest,
    ValidatePolicyRequest,
    ValidateConfigRequest,
)
from h_adminsim.harness.data import describe_data
from h_adminsim.harness.samples import sample_contexts
from h_adminsim.harness.scaffold import policy_template, metric_template
from h_adminsim.harness.validate import validate_policy, validate_config
from h_adminsim.harness.spec import policy_api_spec, system_prompt, bundled_policy_dir
from h_adminsim.harness.workspace import save_policy, chat_policy_dir, list_policy_files, resolve_policy_file



def _report(report: Report) -> dict:
    """
    A report as the tool result: the plain-text summary first, then the details.

    Args:
        report (Report): The harness report.

    Returns:
        dict: ``{'summary', 'ok', 'issues', 'info'}``.
    """
    return {'summary': report.summary(), **report.to_dict()}



def create_app(agent_data_path: str,
               policy_dir: str = HARNESS_DEFAULT_POLICY_DIR,
               reference_dirs: Optional[list] = None,
               api_key: Optional[str] = None) -> FastAPI:
    """
    The tool server application.

    Args:
        agent_data_path (str): Agent data the policies are written for: its patient fields, and the smoke-test samples
                               built from it (cached in `policy_dir`).
        policy_dir (str, optional): Directory of the policies; each chat writes to its own folder in it (see
                                    `chat_policy_dir`), named by the chat id the client sends in an ``X-Chat-Id`` (or
                                    ``X-OpenWebUI-Chat-Id``) header, and sees only that folder and the reference
                                    configs. Defaults to 'policies'.
        reference_dirs (Optional[list], optional): Read-only directories of existing policy configs; None uses the
                                                   configs shipped with the package. Defaults to None.
        api_key (Optional[str], optional): Bearer key every request must carry; None leaves the server open.
                                           Defaults to None.

    Returns:
        FastAPI: The application.
    """
    policy_dir = os.path.abspath(policy_dir)
    reference_dirs = [os.path.abspath(d) for d in (reference_dirs if reference_dirs is not None else [bundled_policy_dir()])]
    agent_data_path = os.path.abspath(agent_data_path)
    os.makedirs(policy_dir, exist_ok=True)

    def authorize(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> None:
        if api_key and (credentials is None or credentials.credentials != api_key):
            raise HTTPException(status_code=401, detail='Invalid or missing API key.')


    def chat_dir(request: Request) -> str:
        chat_id = next((request.headers[h] for h in HARNESS_CHAT_ID_HEADERS if request.headers.get(h)), None)
        return chat_policy_dir(policy_dir, chat_id)


    def resolve(file_name: str, own_dir: str) -> str:
        try:
            path = resolve_policy_file(file_name, own_dir, reference_dirs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if path is None:
            raise HTTPException(status_code=404, detail=f"No file {file_name!r}; see `list_policies`.")
        return path


    app = FastAPI(title='H-AdminSim policy harness', version='1', description=HARNESS_SERVER_DESCRIPTION,
                  dependencies=[Depends(authorize)])


    @app.post('/get_policy_api', operation_id='get_policy_api', description=HARNESS_TOOL_DESCRIPTIONS['get_policy_api'])
    def get_policy_api(body: ApiSpecRequest) -> dict:
        """
        `get_policy_api` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (ApiSpecRequest): The request.

        Returns:
            dict: ``{'markdown'}``: the reference.
        """
        return {'markdown': policy_api_spec(agent_data_path=agent_data_path, include_sources=body.include_examples)}


    @app.post('/describe_data', operation_id='describe_data', description=HARNESS_TOOL_DESCRIPTIONS['describe_data'])
    def describe_data_tool() -> dict:
        """
        `describe_data` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Returns:
            dict: ``{'patient_data', 'always_available'}``.
        """
        return describe_data(agent_data_path, sample_dir=policy_dir)


    @app.post('/list_policies', operation_id='list_policies', description=HARNESS_TOOL_DESCRIPTIONS['list_policies'])
    def list_policies(own_dir: str = Depends(chat_dir)) -> dict:
        """
        `list_policies` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            own_dir (str, optional): This chat's folder. Defaults to the folder of the request's chat.

        Returns:
            dict: ``{'chat_folder', 'files': [{'file_name', 'editable', 'path'}, ...]}``.
        """
        return {'chat_folder': own_dir, 'files': list_policy_files(own_dir, reference_dirs)}


    @app.post('/read_policy', operation_id='read_policy', description=HARNESS_TOOL_DESCRIPTIONS['read_policy'])
    def read_policy(body: ReadRequest, own_dir: str = Depends(chat_dir)) -> dict:
        """
        `read_policy` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (ReadRequest): The request.
            own_dir (str, optional): This chat's folder. Defaults to the folder of the request's chat.

        Returns:
            dict: ``{'file_name', 'content'}``.
        """
        with open(resolve(body.file_name, own_dir)) as f:
            return {'file_name': body.file_name, 'content': f.read()}


    @app.post('/make_policy_template', operation_id='make_policy_template', description=HARNESS_TOOL_DESCRIPTIONS['make_policy_template'])
    def make_policy_template(body: TemplateRequest) -> dict:
        """
        `make_policy_template` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (TemplateRequest): The request.

        Raises:
            HTTPException: 400 if the request is invalid.

        Returns:
            dict: ``{'file_name', 'code', 'config'}``.
        """
        try:
            return policy_template(body.class_name, hooks=body.hooks, requires=body.requires, params=body.params)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


    @app.post('/make_metric_template', operation_id='make_metric_template', description=HARNESS_TOOL_DESCRIPTIONS['make_metric_template'])
    def make_metric_template(body: MetricTemplateRequest) -> dict:
        """
        `make_metric_template` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (MetricTemplateRequest): The request.

        Raises:
            HTTPException: 400 if the request is invalid.

        Returns:
            dict: ``{'file_name', 'code', 'config'}``.
        """
        try:
            return metric_template(body.class_name, requires=body.requires, params=body.params)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


    @app.post('/save_policy', operation_id='save_policy', description=HARNESS_TOOL_DESCRIPTIONS['save_policy'])
    def save_policy_tool(body: SaveRequest, own_dir: str = Depends(chat_dir)) -> dict:
        """
        `save_policy` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (SaveRequest): The request.
            own_dir (str, optional): This chat's folder, where the files go. Defaults to the folder of the request's chat.

        Returns:
            dict: The validation report.
        """
        return _report(save_policy(body.files, body.config, body.config_name, directory=own_dir, overwrite=body.overwrite,
                                   agent_data_path=agent_data_path, sample_dir=policy_dir))


    @app.post('/validate_policy', operation_id='validate_policy', description=HARNESS_TOOL_DESCRIPTIONS['validate_policy'])
    def validate_policy_tool(body: ValidatePolicyRequest, own_dir: str = Depends(chat_dir)) -> dict:
        """
        `validate_policy` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (ValidatePolicyRequest): The request.
            own_dir (str, optional): This chat's folder. Defaults to the folder of the request's chat.

        Returns:
            dict: The validation report.
        """
        return _report(validate_policy(resolve(body.file_name, own_dir), class_name=body.class_name, params=body.params or None,
                                       agent_data_path=agent_data_path, sample_dir=policy_dir))


    @app.post('/validate_policy_config', operation_id='validate_policy_config', description=HARNESS_TOOL_DESCRIPTIONS['validate_policy_config'])
    def validate_policy_config(body: ValidateConfigRequest, own_dir: str = Depends(chat_dir)) -> dict:
        """
        `validate_policy_config` tool (see `HARNESS_TOOL_DESCRIPTIONS`).

        Args:
            body (ValidateConfigRequest): The request.
            own_dir (str, optional): This chat's folder. Defaults to the folder of the request's chat.

        Returns:
            dict: The validation report.
        """
        return _report(validate_config([resolve(n, own_dir) for n in body.config_files],
                                       agent_data_path=agent_data_path, sample_dir=policy_dir))

    return app



def main() -> None:
    """
    Command-line entry point (``h-adminsim-policy-server``): serve the tools, or print the system prompt.
    """
    parser = argparse.ArgumentParser(prog='h-adminsim-policy-server', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (172.17.0.1: reachable from Docker containers on this machine only).')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--policy_dir', default=HARNESS_DEFAULT_POLICY_DIR, help='Directory policies are saved to and read from.')
    parser.add_argument('--reference_dir', action='append', default=None,
                        help='Read-only directory of existing policy configs (repeatable). Default: the configs shipped with h_adminsim.')
    parser.add_argument('--agent_data', default=None,
                        help='Agent data file or directory the policies are written for (required to serve).')
    parser.add_argument('--api_key', default=None,
                        help='Bearer key required from clients (default: POLICY_API_KEY from .env or the environment; none = open).')
    parser.add_argument('--print_system_prompt', action='store_true', help='Print the system prompt for the LLM and exit.')
    args = parser.parse_args()

    if args.print_system_prompt:
        print(system_prompt())
        return
    
    if not args.agent_data:
        parser.error('--agent_data is required to serve the tools')
    
    if not args.api_key:
        dotenv_path = find_dotenv(usecwd=True)
        load_dotenv(dotenv_path, override=True)
        args.api_key = os.environ.get("POLICY_API_KEY", None)
    
    if not args.api_key and args.host not in ('127.0.0.1', 'localhost'):
        print(f"WARNING: no API key while listening on {args.host}; anyone who can reach it can write policy files.")
    
    app = create_app(args.agent_data, args.policy_dir, args.reference_dir, args.api_key)
    review, restrict = sample_contexts(os.path.abspath(args.agent_data), os.path.abspath(args.policy_dir))   # build once
    print(f"Smoke-test samples ready: {len(review)} review, {len(restrict)} restrict contexts.")
    uvicorn.run(app, host=args.host, port=args.port)



if __name__ == '__main__':
    main()
