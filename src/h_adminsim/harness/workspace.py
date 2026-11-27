import os
import re
import shutil
from datetime import date, datetime
from typing import Optional

from h_adminsim.registry.harness import (
    HARNESS_DEFAULT_POLICY_DIR,
    HARNESS_BACKUP_DIR,
    HARNESS_DEFAULT_CHAT_DIR,
    HARNESS_CHAT_MARKER,
    HARNESS_ASSET_EXTS,
)
from h_adminsim.harness.report import Report
from h_adminsim.harness.validate import validate_policy, validate_config

_POLICY_FILE_EXTS = ('.py', '.yaml', '.yml') + HARNESS_ASSET_EXTS



def chat_policy_dir(policy_dir: str, chat_id: Optional[str] = None, today: Optional[date] = None) -> str:
    """
    The folder of one chat inside `policy_dir`, created on first use as ``<YYYYMMDD>_<first 8 chars of the chat id>``
    (the date of that first use). The full chat id is kept in a marker file, so the chat finds its folder again on
    later days and two chats sharing an id prefix get separate folders. Without a chat id, ``default`` is used.

    Args:
        policy_dir (str): The policy directory.
        chat_id (Optional[str], optional): The chat id; characters other than letters, digits, ``-`` and ``_`` are
                                           dropped. Defaults to None.
        today (Optional[date], optional): Date used to name a new folder. Defaults to None (today).

    Returns:
        str: The absolute path of the chat's folder (existing afterwards).
    """
    policy_dir = os.path.abspath(policy_dir)
    chat_id = re.sub(r'[^A-Za-z0-9_-]', '', chat_id or '')
    if not chat_id:
        path = os.path.join(policy_dir, HARNESS_DEFAULT_CHAT_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    os.makedirs(policy_dir, exist_ok=True)
    for name in sorted(os.listdir(policy_dir)):
        marker = os.path.join(policy_dir, name, HARNESS_CHAT_MARKER)
        if os.path.isfile(marker):
            with open(marker) as f:
                if f.read().strip() == chat_id:
                    return os.path.join(policy_dir, name)

    base = f"{(today or date.today()).strftime('%Y%m%d')}_{chat_id[:8]}"
    name, n = base, 1
    while os.path.exists(os.path.join(policy_dir, name)):
        name, n = f"{base}_{n}", n + 1
    path = os.path.join(policy_dir, name)
    os.makedirs(path)
    with open(os.path.join(path, HARNESS_CHAT_MARKER), 'w') as f:
        f.write(chat_id)
    return path



def list_policy_files(own_dir: str, reference_dirs: Optional[list] = None) -> list[dict]:
    """
    Policy files (``.py``), configs (``.yaml``) and text assets (``.txt`` / ``.md``) one chat can see: its own folder (editable), then the reference
    directories (read only; a name already listed is skipped). Chats are independent: other chats' folders are not
    shown, so a chat may write a policy with the same name or rule as another chat.

    Args:
        own_dir (str): The chat's folder (see `chat_policy_dir`).
        reference_dirs (Optional[list], optional): Read-only directories of existing configs. Defaults to None.

    Returns:
        list[dict]: ``{'file_name', 'editable', 'path'}`` per file.
    """
    files, seen = [], set()
    for directory, editable in [(own_dir, True)] + [(d, False) for d in reference_dirs or []]:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith(_POLICY_FILE_EXTS) and os.path.isfile(path) and name not in seen:
                seen.add(name)
                files.append({'file_name': name, 'editable': editable, 'path': path})
    return files



def resolve_policy_file(file_name: str, own_dir: str, reference_dirs: Optional[list] = None) -> Optional[str]:
    """
    Path of a file named as `list_policy_files` names it: looked up in the chat's folder, then in the reference
    directories.

    Args:
        file_name (str): The file name.
        own_dir (str): The chat's folder.
        reference_dirs (Optional[list], optional): Read-only directories of existing configs. Defaults to None.

    Raises:
        ValueError: If the name is not a bare file name (empty, absolute or with a directory part).

    Returns:
        Optional[str]: The absolute path, or None if no such file exists.
    """
    if not file_name or os.path.basename(file_name) != file_name or file_name in ('.', '..'):
        raise ValueError(f"Use a file name from `list_policies`, without directories, got {file_name!r}.")
    candidates = [os.path.join(os.path.abspath(d), file_name) for d in [own_dir] + list(reference_dirs or [])]
    return next((p for p in candidates if os.path.isfile(p)), None)



def save_policy(files: dict,
                config: Optional[str] = None,
                config_name: Optional[str] = None,
                directory: str = HARNESS_DEFAULT_POLICY_DIR,
                overwrite: bool = False,
                *,
                agent_data_path: str,
                sample_dir: Optional[str] = None) -> Report:
    """
    Write policy files and their policy config into `directory` in one go and validate them together: policy and metric
    code in separate ``.py`` files, and text assets (e.g. a negotiation prompt, ``.txt`` / ``.md``) that the config
    declares under ``assets:``. The files are kept even when the validation fails, so they can be fixed and saved again
    with ``overwrite=True``; a replaced file is first copied to ``<directory>/.backup/<name>.<timestamp><ext>``.

    Args:
        files (dict): ``{file name: content}``; bare names ending with ``.py`` or a text asset extension.
        config (Optional[str], optional): The policy config YAML loading them. Without one, `files` must be a single
                                          ``.py`` policy file, validated alone. Defaults to None.
        config_name (Optional[str], optional): File name of the config (``<name>.yaml``). Defaults to the first ``.py``
                                               file's name with ``.yaml``.
        directory (str, optional): The directory to write to (created if needed). Defaults to 'policies'.
        overwrite (bool, optional): Whether to replace existing files (backed up first). Defaults to False.
        agent_data_path (str): Agent data the validation's smoke-test samples are built from.
        sample_dir (Optional[str], optional): Directory caching those samples (see `sample_contexts`). Defaults to None.

    Returns:
        Report: The validation findings (`validate_config` with a config, else `validate_policy`); ``info['files']``
                holds the written paths and ``info['backups']`` the copies of replaced files.
    """
    report = Report()
    names = list(files or {})
    allowed = ('.py',) + HARNESS_ASSET_EXTS
    bad = [n for n in names if os.path.basename(n) != n or not n.endswith(allowed) or n in [e for e in allowed]]
    code_names = [n for n in names if n.endswith('.py')]
    if not names or bad:
        report.error('bad_file_name', f"`files` must map bare file names ending with {list(allowed)} to their content, "
                                      f"got {bad or 'none'}.")
        return report
    if config is None and (len(names) != 1 or not code_names):
        report.error('config_required', "Several files (or a non-Python file) are saved together with the policy config "
                                        "that loads them; pass `config`.")
        return report
    if config is not None:
        config_name = config_name or (os.path.splitext(code_names[0])[0] + '.yaml' if code_names else None)
        if not config_name or os.path.basename(config_name) != config_name or not config_name.endswith(('.yaml', '.yml')):
            report.error('bad_file_name', f"`config_name` must be a bare '<name>.yaml', got {config_name!r}.")
            return report

    directory = os.path.abspath(directory)
    targets = {os.path.join(directory, n): content for n, content in files.items()}
    config_path = os.path.join(directory, config_name) if config is not None else None
    if config_path:
        targets[config_path] = config
    existing = [p for p in targets if os.path.exists(p)]
    if existing and not overwrite:
        report.error('file_exists', f"Files already exist: {existing}. Replace them (overwrite=True) only if you saved "
                                    f"them in this conversation or the user agreed; otherwise ask the user, or use "
                                    f"other file names.")
        return report

    os.makedirs(directory, exist_ok=True)
    backups = []
    if existing:
        backup_dir = os.path.join(directory, HARNESS_BACKUP_DIR)
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        for path in existing:
            stem, ext = os.path.splitext(os.path.basename(path))
            backups.append(os.path.join(backup_dir, f"{stem}.{stamp}{ext}"))
            shutil.copy2(path, backups[-1])
    for path, content in targets.items():
        with open(path, 'w') as f:
            f.write(content)

    check = validate_config(config_path, agent_data_path=agent_data_path, sample_dir=sample_dir) if config_path else \
        validate_policy(os.path.join(directory, code_names[0]), agent_data_path=agent_data_path, sample_dir=sample_dir)
    report.issues.extend(check.issues)
    report.info.update(check.info)
    report.info['files'] = {'written': sorted(os.path.join(directory, n) for n in names), 'config': config_path}
    if backups:
        report.info['backups'] = backups
    return report
