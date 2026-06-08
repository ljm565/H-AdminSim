from typing import Optional

from h_adminsim.agent import BaseAgent
from h_adminsim.registry import ConversationState, MASNode
from h_adminsim.utils import colorstr, log



class HospitalMAS:
    def __init__(self, mas_structure: dict):
        self._structure_check(mas_structure)
        self.nodes = {}
        self.root = self._build_tree(mas_structure)
        self.state = ConversationState()
        self.path = [self.root]
        # Safety cap on routing hops within a single turn (guards against two
        # routers oscillating without ever reaching a leaf).
        self.max_hops_per_turn = 50


    def _structure_check(self, mas_structure: dict):
        """
        Check whether the MAS structure is valid. Each node should contain 'agent' and 'subagent' keys.

        Args:
            mas_structure (dict): A nested dictionary representing the MAS structure. Each node should contain 'agent' and 'subagent' keys.
                                  An optional 'description' string per node helps routers decide.
        """
        def _recursive_check(structure: dict) -> None:
            """
                Check whether all agents have `agent` and `subagent` keys.
                {
                    'orchestrator': {
                        'agent': OrchestratorAgent(...),    # a BaseMASAgent instance
                        'description': '...'   # optional, for router decision-making
                        'subagent': {
                            'intake': {
                                'agent': IntakeAdminStaffAgent(...),
                                'subagent': {}
                            }, ...
                        },
                    }
                }
                """
            for agent_name, agent_info in structure.items():
                if not isinstance(agent_name, str):
                    raise ValueError(f"Agent name must be a string. Found: {type(agent_name)} type.")

                if not isinstance(agent_info, dict):
                    raise ValueError(
                        f"Agent '{agent_name}' configuration must be a dictionary."
                    )

                if 'agent' not in agent_info:
                    raise ValueError(
                        f"Agent '{agent_name}' must contain 'agent' key."
                    )

                if not isinstance(agent_info['agent'], BaseAgent):
                    raise ValueError(
                        f"Agent '{agent_name}' must be a BaseAgent instance, "
                        f"got {type(agent_info['agent'])}."
                    )

                if 'subagent' not in agent_info or not isinstance(agent_info['subagent'], dict):
                    agent_info['subagent'] = {}
                
                if 'description' not in agent_info or not isinstance(agent_info['description'], str):
                    agent_info['description'] = ''

                _recursive_check(agent_info["subagent"])

        if not isinstance(mas_structure, dict):
            raise ValueError("MAS structure must be a dictionary.")

        if len(mas_structure) != 1:
            raise ValueError("Only one orchestrator is allowed.")
        
        _recursive_check(mas_structure)


    def _build_tree(self, structure: dict, parent: Optional[MASNode] = None) -> MASNode:
        """
        Convert a validated structure dict into a ``MASNode`` tree.

        Args:
            structure (dict): A single-key structure dict (one node and its subtree).
            parent (Optional[MASNode]): The parent node, for upward (bubble-up) links.

        Returns:
            MASNode: The constructed node, with children wired recursively.
        """
        (name, info), = structure.items()
        node = MASNode(
            name=name,
            agent=info['agent'],
            parent=parent,
            description=info.get('description'),
        )
        self.nodes[name] = node
        for child_name, child_info in info['subagent'].items():
            node.children[child_name] = self._build_tree({child_name: child_info}, parent=node)
        return node


    def get_agent(self, name: str) -> BaseAgent:
        """
        Fetch a subagent from the MAS tree by node name.

        Args:
            name (str): The node name (e.g. ``'first_visit_intake'``).

        Returns:
            BaseAgent: The agent registered under that node.
        """
        return self.nodes[name].agent


    def reset(self, verbose: bool = True):
        """
        Reset the MAS for a fresh conversation: clear the shared state, reset the
        routing path to the root, and reset every agent's conversation history.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.state = ConversationState()
        self.path = [self.root]
        for node in self.nodes.values():
            node.agent.reset_history(verbose=verbose)


    def aggregate_token_usages(self) -> dict:
        """
        Merge every agent's ``client.token_usages`` (a dict of lists) into a single
        dict by concatenating the per-key lists across all agents.

        Returns:
            dict: Combined token usage across all MAS agents.
        """
        aggregated: dict = {}
        for node in self.nodes.values():
            for key, values in getattr(node.agent.client, 'token_usages', {}).items():
                aggregated.setdefault(key, []).extend(values)
        return aggregated


    def chat(self,
             user_prompt: str,
             using_multi_turn: bool = True,
             verbose: bool = True,
             **kwargs) -> tuple[str, bool]:
        """
        Feed one user message into the MAS and return the resulting reply.

        Args:
            user_prompt (str): The user prompt to send to an agent.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            tuple[str, bool]: ``(reply, is_done)`` — the reply produced by whichever
                agent handled the turn, and whether that agent signalled completion.
        """
        self.state.messages.append({"role": "Patient", "content": user_prompt})
        return self._advance(
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            **kwargs
        )


    def _advance(self,
                 using_multi_turn: bool = True,
                 verbose: bool = True,
                 **kwargs) -> tuple[str, bool]:
        """
        Route down to a leaf and run it for the current turn.

        Args:
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            tuple[str, bool]: ``(reply, is_done)`` for the turn.
        """
        node = self.path[-1]

        # Descend through routers until we reach a leaf worker.
        hops = 0
        while not node.is_leaf:
            hops += 1
            if hops > self.max_hops_per_turn:
                log(colorstr("yellow", f"Routing hop cap ({self.max_hops_per_turn}) hit; replying directly."))
                return self._say("Could you tell me again?"), False

            sub_agents = {n: c.description for n, c in node.children.items()}
            response = node.agent(
                user_prompt=self.state.messages[-1]["content"],
                using_multi_turn=using_multi_turn,
                verbose=verbose,
                sub_agents=sub_agents,
                **kwargs
            )

            # The router must return a JSON object; anything else ends the turn.
            if not isinstance(response, dict):
                return self._say('Could you tell me again?'), False

            choice = response.get('route')

            # This subtree is finished; bubble up to the parent router.
            if choice == node.agent.ROUTE_DONE:
                if not self._pop():
                    self.state.is_complete = True
                    return self._say(response.get('reply') or "Is there anything else I can help you with?"), True
                node = self.path[-1]
                continue

            # Delegation case.
            if choice:
                if choice not in node.children:
                    log(f'{choice} agent cannot be supported.', level='warning')
                    continue    # Retry if the router hallucinates an invalid route (defensive)
                node = node.children[choice]
                self.path.append(node)
                continue

            # Direct reply case (no route): the router answers the user itself.
            return self._say(response.get('reply') or 'Could you tell me again?'), False

        # Leaf worker handles the turn.
        reply, is_done = node.agent.act(
            user_prompt=self.state.messages[-1]["content"],
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            sub_agents=None,    # Because of leaf node
            **kwargs
        )
        self.state.current_agent = node.name
        if is_done:
            self._pop()     # return control to the parent router
        return self._say(reply), is_done


    def _pop(self) -> bool:
        """Pop the active node, returning control to its parent. False if at root."""
        if len(self.path) > 1:
            self.path.pop()
            return True
        return False


    def _say(self, reply: str) -> str:
        """Record an assistant reply in the shared state and return it."""
        self.state.messages.append({"role": "Staff", "content": reply})
        return reply
