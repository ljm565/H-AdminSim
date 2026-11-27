# Policy harness in Open WebUI

Let an LLM in Open WebUI write H-AdminSim follow-up scheduling policies. `h_adminsim` ships an OpenAPI tool server
(`h-adminsim-policy-server`) that exposes the `h_adminsim.harness` tools; it runs where h_adminsim is installed, and
Open WebUI only calls it over HTTP, so nothing is installed in the Open WebUI container.

| Tool | What it does |
|---|---|
| `get_policy_api` | Policy API reference built from the installed package |
| `describe_data` | What the agent data holds per patient field, what the simulator always provides, and built-in metric ranges in the samples |
| `list_policies` / `read_policy` | This chat's policy files (editable), the reference configs and the bundled negotiation prompts (read only) |
| `make_policy_template` / `make_metric_template` | Runnable policy or user-defined metric skeleton and its config |
| `save_policy` | Save the policy file, metric files (`<name>_metric.py`) and text assets (e.g. a prompt `.txt`) with their one config to this chat's folder, and validate them together |
| `validate_policy` / `validate_policy_config` | Validate one file, or several configs combined in priority order |

The LLM names files by bare name (or by the exact path `list_policies` gives): it cannot read or write outside the
policy and reference directories.

## 1. Start the tool server

```bash
pip install h_adminsim
echo "POLICY_API_KEY=<choose a key>" >> .env
h-adminsim-policy-server --host 172.17.0.1 --port 8765 --policy_dir ./policies --agent_data ./agent_data
```

- `--host 172.17.0.1` (the `docker0` bridge) makes the server reachable from Docker containers on this machine only.
- `--policy_dir`: where generated policies are saved (default `./policies`).
- `--reference_dir`: read-only configs the LLM may combine with (default: the configs shipped with h_adminsim,
  `tau.yaml` and `capacity_protection.yaml`; repeatable).
- `--agent_data` (required): the agent data the policies are written for. `describe_data` reports its patient fields,
  and the smoke tests run on samples built from it with the simulator's own scheduler (no LLM): review samples spread
  over preference, urgency and gain, and capacity snapshots at the start, later in time, and on a busy day (a synthetic
  load). They are built once per data (up to about a minute for a large hospital) and cached as
  `<policy_dir>/samples.json`.
- `--api_key`: the bearer key every request must carry; set the same key in Open WebUI. Without it, the server reads
  `POLICY_API_KEY` from a `.env` file (searched from the current directory upward; it takes precedence) or the
  environment, which also keeps the key out of the process list. With neither, the server is open.
- From a clone of the repository, `python src/run/policy_server.py ...` does the same.
- Interactive API docs: `http://172.17.0.1:8765/docs`.

## 2. Run Open WebUI

```bash
docker run -d --name open-webui-policy \
    -p 3001:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui-policy:/app/backend/data \
    --restart unless-stopped \
    ghcr.io/open-webui/open-webui:0.13-slim
```

Open `http://<server>:3001`, create the admin account, and add an LLM connection (any OpenAI-compatible endpoint) under
the admin connection settings. Use a model with good function calling.

## 3. Register the tool server and the model

1. Admin settings → tool servers → add an OpenAPI server:
   URL `http://host.docker.internal:8765`, spec path `openapi.json`, Bearer key = the server's key (`POLICY_API_KEY` or `--api_key`), and the
   custom header `X-Chat-Id: {{CHAT_ID}}` so each chat gets its own policy folder (see below).
2. Workspace → Models → new model on top of your LLM:
   - System prompt: the output of `h-adminsim-policy-server --print_system_prompt`.
   - Tools: enable the tool server.
   - Advanced params → Function calling: `Native` (if the model supports it).

## 4. Use the generated policy

Each chat saves into its own folder, `<policy_dir>/<YYYYMMDD>_<first 8 chars of the chat id>/` (requests without a
chat id use `<policy_dir>/default/`). Chats are independent: a chat sees and edits only its own folder, next to the
read-only reference configs, so two chats may write policies with the same name or rule. Replaced files are
kept in the folder's `.backup/`.

`save_policy` writes the policy's files and its config into the chat's folder. A policy with its own negotiation prompt
keeps it as a `.txt` file declared under the config's `assets:` and passed as `asset:<id>`:

```yaml
api_version: 1
assets:
  soft: soft_prompt.txt              # relative to this config
policies:
  - id: soft_negotiate
    source: file:soft_negotiate.py:SoftNegotiate
    params: {prompt_path: 'asset:soft'}
```

A negotiation prompt is the staff agent's system prompt, filled with `str.format` when a negotiation starts. The
validation checks every prompt a policy negotiates with: placeholders the simulator does not fill (or unescaped braces)
and a missing `#ACCEPT` / `#KEEP_PREFERENCE` or `#FORCE_ACCEPT` instruction are errors, since such a negotiation
fails or never closes. `get_policy_api` lists the placeholders, and the LLM starts from a bundled prompt
(`staff_negotiation_policy_common_system.txt`).

When a simulation runs, asset files are copied into `<run>/policy/` with the policies, and the run metadata records
the path and sha256 of every file a policy parameter names (bundled prompts included). List the YAML under
`schedule_task.policy_config` in the simulation config, in priority order (first = highest):

```yaml
schedule_task:
  policy_config:
    - policies/20261126_1a2b3c4d/urgent_keep.yaml   # the chat's folder; `list_policies` gives full paths
    - <h_adminsim>/assets/policy/tau.yaml     # path: python -c "from h_adminsim.harness import bundled_policy_dir; print(bundled_policy_dir())"
```
