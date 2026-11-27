from typing import Any, Optional

from pydantic import BaseModel, Field



class ApiSpecRequest(BaseModel):
    include_examples: bool = Field(True, description="Include the built-in policies' full source code as examples.")


class TemplateRequest(BaseModel):
    class_name: str = Field(..., description="Policy class name, e.g. 'UrgentKeepPolicy'.")
    hooks: list[str] = Field(['review'], description="Hooks to implement: 'restrict' and/or 'review'.")
    requires: list[str] = Field([], description="What the hooks read: 'patient.<field>' / 'metric.<id>'.")
    params: dict[str, Any] = Field({}, description='Parameters with default values (bool, int, float or str).')


class MetricTemplateRequest(BaseModel):
    class_name: str = Field(..., description="Metric class name, e.g. 'TestDaySpread'.")
    requires: list[str] = Field([], description="What `compute` reads: 'patient.<field>' / 'metric.<id>'.")
    params: dict[str, Any] = Field({}, description='Parameters with default values (bool, int, float or str).')


class SaveRequest(BaseModel):
    files: dict[str, str] = Field(..., description="{file name: full content}: the policy file, metric files "
                                                   "('<name>_metric.py') and text assets ('<name>.txt').")
    config: Optional[str] = Field(None, description="Policy config YAML loading the files: `file:<file>:<Class>` "
                                                    "sources, `metrics:` before the policies reading them, `assets:` "
                                                    "for text files. Required when saving more than one file.")
    config_name: Optional[str] = Field(None, description="Config file name, e.g. 'urgent_keep.yaml' (default: the "
                                                         "first Python file's name with .yaml).")
    overwrite: bool = Field(False, description='Replace existing files: files you saved in this conversation, or others once the user agreed.')


class ReadRequest(BaseModel):
    file_name: str = Field(..., description='A file name from `list_policies`.')


class ValidatePolicyRequest(BaseModel):
    file_name: str = Field(..., description='Python file name from `list_policies`.')
    class_name: Optional[str] = Field(None, description="Policy class (default: the file's only Policy subclass).")
    params: dict[str, Any] = Field({}, description='Parameters to build the policy with.')


class ValidateConfigRequest(BaseModel):
    config_files: list[str] = Field(..., description='Policy config YAML file names in priority order (first = highest).')
