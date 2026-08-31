import os
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, TYPE_CHECKING
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim.registry import STATUS_CODES
from h_adminsim.registry.errors import (
    SchedulingError,
    DataNotFoundError,
    AgentSelectionError,
)
from h_adminsim.tools import scheduling_tool_calling
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.simulation.op_simulation import OPSimulation
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.prompt_utils import load_prompt
from h_adminsim.utils.common_utils import init_result_dict, preprocess_dialog, run_with_retry, staff_role

if TYPE_CHECKING:
    from h_adminsim.tools import SchedulingRule
    from h_adminsim.agent import SchedulingAdminStaffAgent
    from h_adminsim.environment.hospital import HospitalEnvironment



class TurnLimitReached(Exception):
    """
    Raised when a simulation loop exhausts `max_inferences` without the staff
    agent producing a schedule, so the shared error tail can record it like any
    other terminal condition instead of returning early from inside the loop.
    """



class OPSchedulingSimulation(OPSimulation, ABC):
    """
    Shared mechanics for the scheduling simulations
    (``OPFVSchedulingSimulation``, ``OPFUSchedulingSimulation``).

    Subclasses declare the dialogue histories they keep through ``HISTORY_KEYS``
    and supply their own domain logic (staff-reply rendering, post-processing,
    sanity checks). Everything gathered here is identical across the subclasses.

    The subclass contract is spelled out below: two abstract members the shared
    dialogues call, plus the state their ``__init__`` is expected to set.
    """
    # --- Class attributes declared by the subclass -------------------------
    # Packaged schedule-rejection system prompt for this visit type.
    REJECTION_PROMPT: str

    # Reply shown when the retrieval tool cannot find what the patient is asking about;
    # declared by subclasses because it names what they book (appointments vs tests).
    NOT_FOUND_MESSAGE: str

    # --- Instance state the subclass `__init__` is expected to set ---------
    _chief_agent_name: str
    environment: "HospitalEnvironment"
    rules: "SchedulingRule"
    sanity_checker: Optional[SanityChecker]
    scheduling_strategy: str
    fhir_integration: bool


    @property
    @abstractmethod
    def scheduling_agent(self) -> "SchedulingAdminStaffAgent":
        """
        The scheduling worker this simulation drives.

        Subclasses fetch their own leaf worker from the MAS, so the default raises
        rather than returning a value a `super()` call would silently propagate.

        Returns:
            SchedulingAdminStaffAgent: The leaf worker handling scheduling for this visit type.
        """
        raise NotImplementedError(
            colorstr("red", f"{type(self).__name__} does not expose scheduling_agent.")
        )


    @abstractmethod
    def _render_staff_reply(self, prediction: dict, reply_type: str, *args, **kwargs) -> str:
        """
        Turn a structured staff result into the utterance shown to the patient.

        Subclasses may take further arguments; the shared cancellation and rescheduling
        dialogues call this with ``(prediction, reply_type)`` alone. Rendering is
        visit-type specific, so the default raises rather than returning a value a
        `super()` call would record as a staff turn.

        Args:
            prediction (dict): The structured tool-calling/reasoning result for this turn.
            reply_type (str): Which reply the staff agent is making ('cancel', 'reschedule', ...).

        Returns:
            str: The staff utterance to show the patient.
        """
        raise NotImplementedError(
            colorstr("red", f"{type(self).__name__} does not implement _render_staff_reply().")
        )


    def canceling(self,
                  client: AgentExecutor,
                  patient_intention: str,
                  chat_history: list = []) -> dict:
        """
        Handle a multi-turn cancellation request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a cancellation request.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            TypeError: If the prediction or inputs are of an unsupported type.

        Returns:
            dict: Cancelling processed result. Recording the outcome is left to the
                  simulation, which owns the result dictionary.
        """
        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        # Canceling result
        if prediction['type'] == 'tool':
            # Nothing found -> ask the patient to check again
            if prediction['result']['status'] is None and prediction['result']['index']['pred'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = self.NOT_FOUND_MESSAGE
            return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from canceling method."))


    def rescheduling(self,
                     client: AgentExecutor,
                     patient_intention: str,
                     doctor_information: Optional[dict] = None,
                     chat_history: list = [],
                     **kwargs) -> dict:
        """
        Handle a multi-turn rescheduling request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            doctor_information (Optional[dict], optional): Accepted for call-site symmetry; the
                                                           rescheduling pipeline already captures
                                                           it, and the simulation guards it before
                                                           the dialogue starts. Defaults to None.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            TypeError: If the returned type is not supported.

        Returns:
            dict: Rescheduling processed result.
        """
        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        if prediction['type'] == 'tool':
            res = prediction['result']

            # Nothing found -> ask the patient to check again
            if res['status'] is None and res['index']['pred'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = self.NOT_FOUND_MESSAGE
                return prediction

            # Retrieval outcome; the pipeline action below may override it
            result_dict = self._retrieval_result(
                prediction, 'reschedule', STATUS_CODES['reschedule']['identify']
            )
            if res['status'] is False:  # Identification failed -> nothing left to reschedule
                prediction['result_dict'] = result_dict
                prediction['tmp_flag'] = 'retrieve'
                return prediction

            # Translate pipeline action into tmp_flag + result_dict updates
            action = res.get('action')
            if action == 'reschedule':
                result_dict['pred'] = [res['new_schedule']]
                prediction['tmp_flag'] = 'reschedule'
            elif action == 'waiting_list':
                prediction['tmp_flag'] = 'waiting_list'
            elif action == 'schedule_fail':
                result_dict['pred'] = [res['new_schedule']]
                result_dict['status'] = [False]
                result_dict['status_code'] = [STATUS_CODES['reschedule']['schedule'].format(
                    status_code=res.get('schedule_status_code') or STATUS_CODES['format'])]
                prediction['tmp_flag'] = 'schedule'

            prediction['result_dict'] = result_dict
            return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))


    def _cancel_simulate(self,
                         key: str,
                         gt_idx: Optional[int] = None,
                         doctor_information: Optional[dict] = None,
                         patient_schedules: Optional[list[dict]] = None,
                         verbose: bool = True,
                         max_inferences: int = 5,
                         patient_kwargs: dict = {},
                         tool_data: Optional[dict] = None,
                         **kwargs) -> Tuple[dict, dict]:
        """
        Drive a multi-turn cancellation dialogue and record its outcome.

        Args:
            key (str): Dialogue-history key of the running simulation.
            gt_idx (Optional[int], optional): Ground-truth index of the booking to be cancelled.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            tool_data (Optional[dict], optional): Extra domain data this visit type's tools
                                                   need, handed to `build_agent`.
            **kwargs: Additional keyword arguments passed to the patient agent.

        Returns:
            Tuple[dict, dict]: Resolved doctor information and the result dictionary.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
            **(tool_data or {}),
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}

        # Staff turn closure: the tool-calling agent is built once, so binding it here is safe
        def staff_turn(user_prompt: str) -> Tuple[str, dict]:
            prediction = self.canceling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                chat_history=self._to_lc_history(key),
            )
            return self._render_staff_reply(prediction, key), prediction

        # Start conversation
        self._open_staff_turn(key)

        try:
            for _ in range(max_inferences):
                patient_response = self._patient_turn(key, 'cancel', **merged_patient_kwargs)

                # Canceling from staff
                output, prediction = self._staff_turn(patient_response, staff_turn)

                # Naive reply turn
                if prediction['type'] == 'text':
                    self._record_staff_turn(key, output)

                # Record this turn's retrieval outcome
                elif prediction['type'] == 'tool':
                    result_dict = self._retrieval_result(
                        prediction, 'cancel', STATUS_CODES['cancel']['identify']
                    )

                    # A wrong identification cancels nothing -> surface as a not-found failure
                    if prediction['result']['status'] is False:
                        raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))

                    # Successful cancellation closes the dialogue
                    self._record_staff_turn(key, output)
                    self._closing_patient_turn(key, 'cancel', natural_express=False)
                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history[key]))
                    break

            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = self._failure_result(key, {'cancel': gt_idx}, STATUS_CODES['cancel']['identify'])

        except Exception as e:
            result_dict = self._resolve_simulation_error(
                e, key, {'cancel': gt_idx},
                error_codes={
                    AgentSelectionError: STATUS_CODES['agent'],       # Wrong agent activated
                    TypeError: STATUS_CODES['cancel']['type'],        # Tool calling error
                },
                result_dict=result_dict,
                dialog_only=(DataNotFoundError,),                     # Schedule identification error
            )

        log("Simulation completed.", color=True)
        self._finish_scheduling_turn(key, verbose)
        return doctor_information, result_dict


    def _reschedule_simulate(self,
                             key: str,
                             pipeline_factory: Callable,
                             gt_idx: Optional[int] = None,
                             doctor_information: Optional[dict] = None,
                             patient_schedules: Optional[list[dict]] = None,
                             verbose: bool = True,
                             max_inferences: int = 5,
                             patient_kwargs: dict = {},
                             staff_kwargs: dict = {},
                             tool_data: Optional[dict] = None,
                             **kwargs) -> Tuple[dict, dict]:
        """
        Drive a multi-turn rescheduling dialogue and record its outcome.

        Args:
            key (str): Dialogue-history key of the running simulation.
            pipeline_factory (Callable): ``(doctor_information, **tool_data, **staff_kwargs) -> pipeline``
                                          building the post-retrieval rescheduling pipeline for this
                                          visit type.
            gt_idx (Optional[int], optional): Ground-truth index of the booking to be rescheduled.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            tool_data (Optional[dict], optional): Extra domain data this visit type's tools need.
                                                   Both the tool-calling agent and the rescheduling
                                                   pipeline receive it, so a subclass declares it once
                                                   instead of binding it into `pipeline_factory`.
            **kwargs: Additional keyword arguments passed to both agents.

        Returns:
            Tuple[dict, dict]: Resolved doctor information and the result dictionary.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
            reschedule_pipeline=pipeline_factory(
                doctor_information, **(tool_data or {}), **merged_staff_kwargs
            ),
            **(tool_data or {}),
        )

        # Staff turn closure: the tool-calling agent is built once, so binding it here is safe
        def staff_turn(user_prompt: str) -> Tuple[str, dict]:
            prediction = self.rescheduling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                chat_history=self._to_lc_history(key),
            )
            return self._render_staff_reply(prediction, key), prediction

        # Start conversation
        self._open_staff_turn(key)

        try:
            for _ in range(max_inferences):
                patient_response = self._patient_turn(key, 'move', **merged_patient_kwargs)

                # Rescheduling from staff
                output, prediction = self._staff_turn(patient_response, staff_turn)

                # Naive reply turn
                if prediction['type'] == 'text':
                    self._record_staff_turn(key, output)

                # Record this turn's rescheduling outcome
                elif prediction['type'] == 'tool':
                    # Tool-calling failures resolve nothing -> surface them before recording a staff turn.
                    tmp_flag = prediction.get('tmp_flag')
                    if tmp_flag == 'retrieve':
                        result_dict = prediction['result_dict']
                        raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))
                    elif tmp_flag == 'schedule':
                        result_dict = prediction['result_dict']
                        raise SchedulingError(colorstr("red", "Error: Scheduling error."))
                    elif tmp_flag not in ('waiting_list', 'reschedule'):
                        raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))

                    # Successful reschedule / waiting-list closes the dialogue
                    result_dict = prediction['result_dict']
                    self._record_staff_turn(key, output)
                    self._closing_patient_turn(key, 'move', natural_express=False)
                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history[key]))
                    break

            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = self._failure_result(key, {'reschedule': gt_idx}, STATUS_CODES['reschedule']['identify'])

        except Exception as e:
            result_dict = self._resolve_simulation_error(
                e, key, {'reschedule': gt_idx},
                error_codes={
                    AgentSelectionError: STATUS_CODES['agent'],           # Wrong agent activated
                    TypeError: STATUS_CODES['reschedule']['type'],        # Tool calling error
                },
                result_dict=result_dict,
                # Identification / scheduling failures already recorded their own result
                dialog_only=(DataNotFoundError, SchedulingError),
            )

        log("Simulation completed.", color=True)
        self._finish_scheduling_turn(key, verbose)
        return doctor_information, result_dict
    

    @abstractmethod
    def _rejection_prompt_fields(self,
                                 patient_condition: dict,
                                 rejected_preference: str) -> dict:
        """
        Build the fields that fill this visit type's schedule-rejection prompt.

        The preference vocabularies differ per visit type — first-visit preferences name a
        doctor or a date, follow-up ones name a test-scheduling objective — so each subclass
        phrases its own and returns what its template asks for.

        Args:
            patient_condition (dict): Patient ground-truth condition including current preference.
            rejected_preference (str): The scheduling preference the staff agent proposed in the
                                        previous turn that the patient must explicitly reject.

        Returns:
            dict: Keyword arguments for `rejection_system_prompt_template.format`, excluding
                  `personality`, which the caller supplies.
        """
        raise NotImplementedError(
            colorstr("red", f"{type(self).__name__} does not implement _rejection_prompt_fields().")
        )


    def _update_patient_system_prompt(self,
                                      patient_condition: Optional[dict] = None,
                                      rejected_preference: Optional[str] = None,
                                      new_system_prompt: Optional[str] = None):
        """
        Swap the patient agent's system prompt, either into the rejection scenario or to a
        caller-supplied one.

        Args:
            patient_condition (Optional[dict], optional): Patient ground-truth condition including
                                                           current preference. Given together with
                                                           `rejected_preference` to enter the
                                                           rejection scenario.
            rejected_preference (Optional[str], optional): The scheduling preference proposed by the
                                                           staff agent in the previous turn that the
                                                           patient must explicitly reject.
            new_system_prompt (Optional[str], optional): System prompt to switch to instead, used to
                                                          steer the patient's closing reaction.

        Raises:
            ValueError: If neither the rejection pair nor a replacement prompt was given.
        """
        # Rejection scenario: the subclass phrases its own preference vocabulary
        if patient_condition is not None and rejected_preference is not None:
            system_prompt = self.rejection_system_prompt_template.format(
                personality=self.patient_agent.personality,
                **self._rejection_prompt_fields(patient_condition, rejected_preference),
            )
        else:
            if not new_system_prompt:
                raise ValueError(colorstr("red", f"`new_system_prompt` must be provided."))
            system_prompt = new_system_prompt

        # Apply it to the agent, and to the system turn already sitting in its history
        self.patient_agent.system_prompt = system_prompt
        if len(self.patient_agent.client.histories) and \
            isinstance(self.patient_agent.client.histories[0], dict) and \
                self.patient_agent.client.histories[0].get('role') == 'system':
            self.patient_agent.client.histories[0]['content'][0]['text'] = system_prompt



    def _open_staff_turn(self, key: str, greet: Optional[str] = None) -> None:
        """
        Take the staff turn that opens a dialogue, seeding the orchestrator's own
        message log alongside the dialogue history so the greeting is part of both.

        Args:
            key (str): Dialogue-history key of the running simulation.
            greet (Optional[str], optional): Greeting to open with. Defaults to the
                                              orchestrator's, which every flow uses except
                                              first-visit scheduling — that one opens with
                                              the scheduling worker's own appointment greeting.
        """
        staff_greet = greet or self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history[key].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(role=self.admin_staff_mas.path[-1].name):<25}: {staff_greet}")


    def _patient_turn(self,
                      key: str,
                      label: str,
                      prompt: Optional[str] = None,
                      max_retries: Optional[int] = None,
                      **patient_kwargs) -> str:
        """
        Take one patient turn and record it.

        Args:
            key (str): Dialogue-history key of the running simulation.
            label (str): Short tag shown next to 'Patient' in the log — the operation for the
                          cancellation and rescheduling flows ('cancel', 'move'), the patient's
                          current scheduling preference for the scheduling ones.
            prompt (Optional[str], optional): What to send the patient agent. Defaults to the
                                               last utterance in the dialogue; the streaming flow
                                               overrides it to steer the patient's closing reaction.
            max_retries (Optional[int], optional): Retry the agent call this many times. Demo
                                                    paths retry so a transient API error cannot end
                                                    the stream; evaluation paths let it surface and
                                                    be recorded. Defaults to None (no retry).
            **patient_kwargs: Additional keyword arguments forwarded to the patient agent.

        Returns:
            str: The patient utterance.
        """
        prompt = self.dialog_history[key][-1]["content"] if prompt is None else prompt
        call_kwargs = dict(using_multi_turn=True, verbose=False, **patient_kwargs)

        if max_retries is None:
            patient_response = self.patient_agent(prompt, **call_kwargs)
        else:
            patient_response = run_with_retry(
                self.patient_agent, prompt, max_retries=max_retries, **call_kwargs
            )

        self.dialog_history[key].append({"role": "Patient", "content": patient_response})
        role = f"{colorstr('green', 'Patient')} ({label})"
        log(f"{role:<25}: {patient_response}")
        return patient_response


    def _closing_patient_turn(self,
                              key: str,
                              label: str,
                              natural_express: bool = True,
                              max_retries: Optional[int] = None,
                              **patient_kwargs) -> str:
        """
        Close a scheduling dialogue with the patient's reaction to the proposed schedule.

        Args:
            key (str): Dialogue-history key of the running simulation.
            label (str): Short tag shown next to 'Patient' in the log.
            natural_express (bool, optional): Whether to have the patient react in their own words.
                                               When False the dialogue closes on the fixed thank-you
                                               without spending a turn on the agent — what the
                                               cancellation and rescheduling flows use, since the
                                               tool has already settled the request and the patient
                                               has nothing left to weigh up. Defaults to True.
            max_retries (Optional[int], optional): Retry the agent call this many times; see
                                                    `_patient_turn`. Defaults to None (no retry).
            **patient_kwargs: Additional keyword arguments forwarded to the patient agent.

        Returns:
            str: The patient's closing utterance.
        """
        # Fixed closing: nothing to generate, just record it
        if not natural_express:
            self.dialog_history[key].append({"role": "Patient", "content": self.end_phrase})
            role = f"{colorstr('green', 'Patient')} ({label})"
            log(f"{role:<25}: {self.end_phrase}")
            return self.end_phrase

        # Have the patient react to the schedule the staff just proposed
        self._update_patient_system_prompt(new_system_prompt=self.patient_satisfaction_system_prompt)
        return self._patient_turn(
            key,
            label,
            prompt=self.natural_end_phrase.format(schedule=self.dialog_history[key][-1]['content']),
            max_retries=max_retries,
            **patient_kwargs,
        )


    def _record_staff_turn(self, key: str, output) -> str:
        """
        Record the staff utterance produced by a MAS turn.

        Args:
            key (str): Dialogue-history key of the running simulation.
            output: The MAS output carrying the rendered reply and the agent that produced it.

        Returns:
            str: The recorded staff utterance, which the streaming flow also yields.
        """
        staff_response, _role = output.response, output.agent
        self.dialog_history[key].append({"role": "Staff", "content": staff_response})
        log(f"{staff_role(role=_role):<25}: {staff_response}")
        return staff_response


    def _init_prompt(self, schedule_rejection_prompt_path: Optional[str] = None):
        """
        Initialize the patient-side prompts used across the scheduling simulations.

        Args:
            schedule_rejection_prompt_path (Optional[str], optional): Path to a custom schedule
                                                                      rejection system prompt file.
                                                                      If not provided, the subclass's
                                                                      `REJECTION_PROMPT` is used.
                                                                      Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Rejection scenario system prompt: packaged per visit type, overridable by the caller
        if not schedule_rejection_prompt_path:
            self.rejection_system_prompt_template = load_prompt(self.REJECTION_PROMPT)
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                self.rejection_system_prompt_template = f.read()

        # Shared prompts driving the patient's reaction to a proposed schedule
        self.patient_satisfaction_system_prompt = load_prompt('opfvfu_schedule_patient_satisfied_system.txt')
        self.natural_end_phrase = load_prompt('opfvfu_schedule_patient_satisfied_user.txt')
        self.patient_evaluation_system_prompt = load_prompt('opfvfu_schedule_patient_evaluation_system.txt')
        self.patient_schedule_evaluation_phrase = load_prompt('opfvfu_schedule_patient_evaluation_user.txt')
        self.end_phrase = "Thank you."


    def _staff_turn(self,
                    user_prompt: str,
                    respond: Callable[[str], Tuple[str, dict]],
                    force_on_misroute: bool = False) -> Tuple[object, dict]:
        """
        Route one staff turn through the MAS and recover both of its results.

        ``admin_staff_mas.chat`` drives the staff turn through a callback that may only
        return the utterance to show the patient, but the simulation also needs the
        structured prediction behind it. This carries that prediction out of the callback
        so callers get both, and applies the simulation's misrouting policy.

        Throughout the simulations, ``staff_response`` names the utterance shown to the
        patient and ``prediction`` names the structured tool-calling/reasoning result.

        Args:
            user_prompt (str): The patient utterance that opens this turn.
            respond (Callable[[str], Tuple[str, dict]]): Produces ``(staff_response,
                                                          prediction)`` for a prompt.
            force_on_misroute (bool, optional): Whether to deterministically re-run the turn on
                                                 the chief agent when the orchestrator picks the
                                                 wrong one, instead of raising. Demo/streaming
                                                 paths must not surface a misroute, whereas
                                                 evaluation paths record it as a failure.
                                                 Defaults to False.

        Raises:
            AgentSelectionError: If the orchestrator activated the wrong agent and
                                 `force_on_misroute` is False.

        Returns:
            Tuple[object, dict]: The MAS output and the structured prediction.
        """
        holder = {}

        def callback(prompt: str) -> Tuple[str, bool]:
            reply, prediction = respond(prompt)
            holder['prediction'] = prediction
            return reply, False

        output = self.admin_staff_mas.chat(
            user_prompt=user_prompt,
            callback=callback,
            using_multi_turn=False,
            verbose=False,
        )

        if output.agent != self._chief_agent_name:
            if not force_on_misroute:
                raise AgentSelectionError(
                    colorstr('red', f'Wrong agent activated, expected {self._chief_agent_name} but got {output.agent}')
                )
            log(f"Wrong agent ({output.agent}) activated; forcing {self._chief_agent_name}.", level="warning")
            output = self.admin_staff_mas.force_chat(
                self._chief_agent_name,
                user_prompt=user_prompt,
                callback=callback,
                using_multi_turn=False,
                verbose=False,
            )

        return output, holder.pop('prediction')


    def _retrieval_result(self,
                          prediction: dict,
                          key: str,
                          identify_code: str) -> dict:
        """
        Translate a retrieval tool's verdict into a result dictionary.

        The retrieval tools decide correctness themselves — ``gt_idx`` is baked into the
        agent at build time — so nothing is judged here; the verdict the tool already
        returned is only reshaped into the recording format.

        Args:
            prediction (dict): A ``type == 'tool'`` staff result whose ``'result'`` carries
                               the tool's ``'status'`` verdict and ``'index'`` pair.
            key (str): Result-dict key naming the retrieval ('cancel', 'reschedule', ...).
            identify_code (str): Status code recorded when identification failed.

        Returns:
            dict: Result dictionary holding this turn's retrieval outcome.
        """
        status = prediction['result']['status']
        index = prediction['result']['index']

        result_dict = init_result_dict()
        result_dict['gt'].append({key: None if status is None else index['gt']})
        result_dict['pred'].append({key: index['pred']})
        result_dict['status'].append(status)
        result_dict['status_code'].append(
            None if status is None
            else (identify_code if status is False else STATUS_CODES['correct'])
        )
        return result_dict


    def _failure_result(self,
                        key: str,
                        gt,
                        status_code: str) -> dict:
        """
        Build the standard single-entry failure result for a simulation run.

        Args:
            key (str): Dialogue-history key of the running simulation.
            gt: Ground-truth entry to record alongside the failure.
            status_code (str): Status code describing why the run failed.

        Returns:
            dict: Result dictionary in the shape the evaluation pipeline expects.
        """
        return {
            'gt': [gt],
            'pred': [None],
            'status': [False],
            'status_code': [status_code],
            'dialog': [preprocess_dialog(self.dialog_history[key])],
        }


    def _resolve_simulation_error(self,
                                  e: Exception,
                                  key: str,
                                  gt,
                                  error_codes: dict,
                                  result_dict: Optional[dict] = None,
                                  dialog_only: tuple = ()) -> dict:
        """
        Translate an exception that ended a simulation loop into a result dictionary.

        Args:
            e (Exception): The exception that ended the loop.
            key (str): Dialogue-history key of the running simulation.
            gt: Ground-truth entry to record alongside the failure.
            error_codes (dict): ``{exception type: status code}`` for this simulation,
                                 ordered most specific first. Anything not listed falls
                                 through to ``STATUS_CODES['unexpected']``.
            result_dict (Optional[dict], optional): Result accumulated so far; required
                                                     only when `dialog_only` is used.
            dialog_only (tuple, optional): Exception types whose `result_dict` was already
                                            populated by the inner tool-calling step, so the
                                            dialogue is appended instead of replacing it.

        Returns:
            dict: The result dictionary to return from the simulation.
        """
        # The loop already recorded the outcome; only the dialogue is missing.
        if dialog_only and isinstance(e, dialog_only):
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history[key]))
            return result_dict

        for exc, status_code in error_codes.items():
            if isinstance(e, exc):
                if isinstance(e, AgentSelectionError):
                    log(str(e), level='warning')
                return self._failure_result(key, gt, status_code)

        # Unexpected failure: surface the exception text in the status code.
        status_code = STATUS_CODES['unexpected'].format(e=e)
        log(status_code, level='warning')
        return self._failure_result(key, gt, status_code)


    def _to_lc_history(self, key: str) -> list:
        """
        Convert the dialog history for the given key into LangChain message objects.

        Args:
            key (str): Key identifying which dialog history to convert.

        Returns:
            list: A list of LangChain HumanMessage and AIMessage objects.
        """
        msgs = []
        for m in self.dialog_history[key]:
            if m["role"] == "Patient":
                msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "Staff":
                msgs.append(AIMessage(content=m["content"]))
        return msgs


    def _accumulate_staff_tokens(self,
                                 prediction: dict,
                                 staff_token_stats: dict,
                                 staff_token_callback: TokenUsageCallback) -> dict:
        """
        Merge this turn's staff token usage into the running stats and return them.

        Args:
            prediction (dict): The staff scheduling result (carries ``'token'`` for reasoning).
            staff_token_stats (dict): The running per-key token usage to update.
            staff_token_callback (TokenUsageCallback): Cumulative callback for the tool-calling path.

        Returns:
            dict: The updated token statistics.
        """
        if self.scheduling_strategy == 'tool_calling':
            return staff_token_callback.token_usage
        for k, v in prediction['token'].items():
            if k not in staff_token_stats:
                staff_token_stats[k] = deepcopy(v)
            else:
                staff_token_stats[k].extend(v)
        return staff_token_stats


    def _finish_scheduling_turn(self,
                                reply_type: str,
                                verbose: bool = False):
        """
        Hand the floor back to the orchestrator once the scheduling eval loop is done.

        Args:
            reply_type (str): Reply types of the scheduling agent.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        if len(self.admin_staff_mas.path) <= 1:
            return

        closing = self.dialog_history[reply_type][-1]['content']
        messages = self.admin_staff_mas.state.messages

        # When end abnormally
        if messages and messages[-1]['content'] == closing:
            return

        # When end normally
        self.admin_staff_mas.chat(
            user_prompt=closing,
            using_multi_turn=False,
            verbose=verbose,
            is_done=True,
        )
