import re
import math
from collections import Counter
from typing import Any, Optional
from dataclasses import dataclass, field

from h_adminsim.registry.harness import HARNESS_ISSUE_LEVELS

_ANSI = re.compile(r'\x1b\[[0-9;]*m')



def plain_text(value: Any) -> str:
    """
    A message without ANSI color codes (the package colors its error messages for the terminal).

    Args:
        value (Any): A message or an exception.

    Returns:
        str: The plain message.
    """
    return _ANSI.sub('', str(value))



def summarize_values(values: list) -> dict:
    """
    Compact summary of metric values: their number, the finite range (min / median / max) and the count of infinities
    for numbers, the counts of other values, how many are None, and how many distinct values occur.

    Args:
        values (list): The values.

    Returns:
        dict: ``{'samples', 'distinct'[, 'min', 'median', 'max', 'inf', 'none', 'values']}``.
    """
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    finite = sorted(v for v in numbers if math.isfinite(v))
    others = [v for v in values if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool))]
    summary = {'samples': len(values), 'distinct': len({repr(v) for v in values})}
    if finite:
        summary.update(min=round(finite[0], 4), median=round(finite[len(finite) // 2], 4), max=round(finite[-1], 4))
    if len(numbers) > len(finite):
        summary['inf'] = len(numbers) - len(finite)
    if any(v is None for v in values):
        summary['none'] = sum(v is None for v in values)
    if others:
        summary['values'] = {k: n for k, n in Counter(str(v) for v in others).most_common(10)}
    return summary



@dataclass(frozen=True)
class Issue:
    """
    One finding of a harness check.

    Fields:
        level: ``'error'`` (the policy cannot run as is) or ``'warning'`` (it runs, but likely not as intended).
        code: Short kind of the finding (e.g. ``'syntax_error'``, ``'hook_raised'``).
        message: What is wrong and how to fix it, in plain text.
        where: Location, e.g. ``'my_policy.py:42'`` or ``'review sample 3'`` (None if not tied to one).
    """
    level: str
    code: str
    message: str
    where: Optional[str] = None


    def __post_init__(self):
        """
        Raises:
            ValueError: If `level` is not one of `HARNESS_ISSUE_LEVELS`.
        """
        if self.level not in HARNESS_ISSUE_LEVELS:
            raise ValueError(f"Issue level must be one of {list(HARNESS_ISSUE_LEVELS)}, got {self.level!r}")


    def to_dict(self) -> dict:
        """
        JSON-serializable form.

        Returns:
            dict: ``{'level', 'code', 'message', 'where'}``.
        """
        return {'level': self.level, 'code': self.code, 'message': self.message, 'where': self.where}



@dataclass
class Report:
    """
    Result of a harness check. Checks do not raise on a broken policy; they report it here.

    Fields:
        issues: The findings, in the order found.
        info: JSON-serializable facts about what was checked (e.g. the policy's hooks and smoke-test outcomes).
    """
    issues: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


    @property
    def ok(self) -> bool:
        """Whether no error was found (warnings allowed)."""
        return not self.errors


    @property
    def errors(self) -> list:
        """The error issues."""
        return [i for i in self.issues if i.level == 'error']


    @property
    def warnings(self) -> list:
        """The warning issues."""
        return [i for i in self.issues if i.level == 'warning']


    def error(self, code: str, message: Any, where: Optional[str] = None) -> None:
        """
        Add an error.

        Args:
            code (str): Short kind of the finding.
            message (Any): What is wrong (an exception is turned into its plain message).
            where (Optional[str], optional): Location. Defaults to None.
        """
        self.issues.append(Issue('error', code, plain_text(message), where))


    def warning(self, code: str, message: Any, where: Optional[str] = None) -> None:
        """
        Add a warning.

        Args:
            code (str): Short kind of the finding.
            message (Any): What is likely wrong (an exception is turned into its plain message).
            where (Optional[str], optional): Location. Defaults to None.
        """
        self.issues.append(Issue('warning', code, plain_text(message), where))


    def to_dict(self) -> dict:
        """
        JSON-serializable form.

        Returns:
            dict: ``{'ok', 'issues', 'info'}``.
        """
        return {'ok': self.ok, 'issues': [i.to_dict() for i in self.issues], 'info': self.info}


    def summary(self) -> str:
        """
        Short plain-text summary: the verdict, then one line per issue.

        Returns:
            str: The summary.
        """
        head = 'OK' if self.ok else 'FAILED'
        counts = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        lines = [f"{head}: {counts}"]
        for i in self.issues:
            where = f" ({i.where})" if i.where else ''
            lines.append(f"- [{i.level}] {i.code}{where}: {i.message}")
        return '\n'.join(lines)
