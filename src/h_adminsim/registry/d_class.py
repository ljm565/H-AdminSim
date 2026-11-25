from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from h_adminsim.agent.base import BaseAgent



@dataclass
class ConversationState:
    """Dialog messages so far and the MAS agent currently handling the conversation."""
    messages: list[dict] = field(default_factory=list)
    current_agent: str | None = None



@dataclass
class MASNode:
    """
    A single node in the Multi-Agent System tree.

    A node with no ``children`` is a leaf worker; otherwise its ``agent`` acts as
    a router that delegates to one child per turn.
    """
    name: str
    agent: "BaseAgent"
    children: dict[str, "MASNode"] = field(default_factory=dict)
    parent: Optional["MASNode"] = None
    description: Optional[str] = None
    is_complete: bool = False
    next_step: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no children (a worker, not a router)."""
        return not self.children
