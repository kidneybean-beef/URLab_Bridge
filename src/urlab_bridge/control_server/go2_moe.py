from __future__ import annotations

import argparse
import logging
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .commands import TwistSlewConfig, TwistSlewLimiter
from .metrics import ControlLoopMetrics
from .models import WebPolicyTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Go2MoeDependencies:
    action_size: int = 12
    select_articulation: Callable[[Any, str], str] | None = None
    format_pose_sample: Callable[[dict[str, float]], object] | None = None
    push_gains: Callable[..., int] | None = None
    resolve_torque_limits: Callable[[float | None], object] | None = None
    sync_command_source_runtime_ui: Callable[[Any, Any, str], None] | None = None
    build_compatibility_report: Callable[..., Any] | None = None
    format_compatibility_report: Callable[[Any], str] | None = None
    safety_abort_reason: Callable[..., str | None] | None = None
    capture_actuated_joint_pose: Callable[[Any], dict[str, float]] | None = None
    validate_target_pose: Callable[[dict[str, float]], str | None] | None = None
    load_policy: Callable[..., Any] | None = None
    build_observation: Callable[..., Any] | None = None
    reset_history: Callable[..., None] | None = None
    infer_action: Callable[..., tuple[Any, Any]] | None = None
    action_abort_reason: Callable[..., str | None] | None = None
    apply_action_limit: Callable[..., tuple[Any, bool]] | None = None
    action_to_target_pose: Callable[[Any, Any], dict[str, float]] | None = None
    maybe_rate_limit_target_pose: Callable[..., dict[str, float]] | None = None
    target_pose_to_action: Callable[[dict[str, float]], Any] | None = None
    target_delta_abs_max: Callable[[dict[str, float], dict[str, float]], float] | None = None
    signal_handler: Callable[..., Any] = signal.signal

    def require(self, name: str) -> Any:
        value = getattr(self, name)
        if value is None:
            raise RuntimeError(f"missing Go2 MoE dependency: {name}")
        return value


class _PolicyState:
    def __init__(
        self,
        *,
        articulation: str,
        prefix: str,
        art: Any,
        command_source: Any,
        action_size: int,
        command_slew_config: TwistSlewConfig,
    ) -> None:
        self.articulation = articulation
        self.prefix = prefix
        self.art = art
        self.command_source = command_source
        self.command_limiter = TwistSlewLimiter(command_slew_config)
        self.desired_command = (0.0, 0.0, 0.0)
        self.applied_command = (0.0, 0.0, 0.0)
        self.last_action = np.zeros(action_size, dtype=np.float32)
        self.current_pose: dict[str, float] = {}
        self.previous_target: dict[str, float] = {}
        self.pending_previous_target: dict[str, float] | None = None
        self.pending_last_action: Any | None = None
        self.iters = 0


class Go2MoePolicyRuntime:
    def __init__(
        self,
        *,
        policy_path: str,
        device: str,
        dependencies: Go2MoeDependencies,
        log: logging.Logger = logger,
    ) -> None:
        self.policy_path = policy_path
        self.device = device
        self.dependencies = dependencies
        self._logger = log
        self._policy: Any | None = None

    def load(self, states: Sequence[_PolicyState]) -> None:
        load_policy = self.dependencies.require("load_policy")
        self._policy = load_policy(self.policy_path, device=self.device)
        self._logger.info(
            "Go2 MoE policy runtime: policy=%s device=%s robots=%d batching=unified",
            self.policy_path,
            self.device,
            len(states),
        )

    def reset_histories(
        self,
        states: Sequence[_PolicyState],
        observations: Sequence[Any],
    ) -> None:
        reset_history = self.dependencies.require("reset_history")
        reset_history(
            self._require_policy(),
            np.stack(observations).astype(np.float32, copy=False),
            device=self.device,
        )

    def infer(
        self,
        states: Sequence[_PolicyState],
        observations: Sequence[Any],
    ) -> tuple[list[Any], list[Any]]:
        infer_action = self.dependencies.require("infer_action")
        actions, diagnostics = infer_action(
            self._require_policy(),
            np.stack(observations).astype(np.float32, copy=False),
            device=self.device,
        )
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1 and len(states) == 1:
            actions_array = actions_array.reshape(1, -1)
        diagnostics_list = (
            list(diagnostics)
            if isinstance(diagnostics, (list, tuple))
            else [diagnostics]
        )
        action_list = [actions_array[i] for i in range(len(states))]
        if len(action_list) != len(states) or len(diagnostics_list) != len(states):
            raise RuntimeError(
                "policy result count does not match robot count: "
                f"actions={len(action_list)} diagnostics={len(diagnostics_list)} "
                f"robots={len(states)}"
            )
        return action_list, diagnostics_list

    def _require_policy(self) -> Any:
        if self._policy is None:
            raise RuntimeError("Go2 MoE policy runtime has not loaded a policy")
        return self._policy


class Go2MoeControlLoop:
    def __init__(
        self,
        args: argparse.Namespace,
        target_sources: Sequence[tuple[WebPolicyTarget, Any]],
        limit_mode: Any,
        dependencies: Go2MoeDependencies,
        *,
        metrics: ControlLoopMetrics | None = None,
        metrics_log_interval_s: float | None = None,
        post_step_hook: Callable[[Sequence[Any]], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        log: logging.Logger = logger,
    ) -> None:
        if not target_sources:
            raise SystemExit("no web policy targets configured")
        self.args = args
        self.target_sources = list(target_sources)
        self.limit_mode = limit_mode
        self.dependencies = dependencies
        self.metrics = metrics or ControlLoopMetrics(freq_hz=float(args.freq))
        self.metrics_log_interval_s = (
            float(metrics_log_interval_s)
            if metrics_log_interval_s is not None
            else float(getattr(args, "metrics_log_interval_s", 1.0))
        )
        self._clock = clock
        self._logger = log
        self._post_step_hook = post_step_hook
        self._last_metrics_log_at: float | None = None
        self._command_dt = 1.0 / float(args.freq)
        self._command_slew_config = TwistSlewConfig(
            accel_vx=float(getattr(args, "cmd_accel_vx", 2.0)),
            accel_vy=float(getattr(args, "cmd_accel_vy", 1.0)),
            accel_yaw=float(getattr(args, "cmd_accel_yaw", 3.14)),
            decel_vx=float(getattr(args, "cmd_decel_vx", 3.0)),
            decel_vy=float(getattr(args, "cmd_decel_vy", 1.5)),
            decel_yaw=float(getattr(args, "cmd_decel_yaw", 4.71)),
        )

    def run(self, client: Any) -> int:
        deps = self.dependencies
        select_articulation = deps.require("select_articulation")
        resolve_torque_limits = deps.require("resolve_torque_limits")

        prefixes: tuple[str, ...] = tuple(
            select_articulation(client, target.articulation)
            for target, _source in self.target_sources
        )
        states: list[_PolicyState] = []
        switched_prefixes: list[str] = []
        stop = False
        iters = 0

        def _on_sigint(_sig, _frame) -> None:
            nonlocal stop
            stop = True

        deps.signal_handler(signal.SIGINT, _on_sigint)

        try:
            torque_limits = resolve_torque_limits(self.args.torque_limit)
            self._logger.info(
                "multi-web MoE params: targets=%s freq=%.1fHz action_clip=%.1f "
                "target_slew=%s warmup_steps=%d push_gains=%s kp=%.3f kv=%.3f "
                "torque_limits=%s command_accel=%s command_decel=%s",
                list(prefixes),
                self.args.freq,
                self.args.max_action_abs,
                "none"
                if self.limit_mode.max_target_step is None
                else self.limit_mode.max_target_step,
                self.args.warmup_steps,
                bool(self.args.push_gains),
                self.args.kp,
                self.args.kv,
                np.round(torque_limits, 3).tolist(),
                self._command_slew_config.acceleration,
                self._command_slew_config.deceleration,
            )
            if self.limit_mode.raw_policy:
                self._logger.warning(
                    "RAW POLICY MODE: no action clipping and no target slew limiting"
                )

            for (target, command_source), prefix in zip(
                self.target_sources,
                prefixes,
                strict=True,
            ):
                art = client.articulations[prefix]
                states.append(
                    _PolicyState(
                        articulation=target.articulation,
                        prefix=prefix,
                        art=art,
                        command_source=command_source,
                        action_size=deps.action_size,
                        command_slew_config=self._command_slew_config,
                    )
                )
                self._logger.info(
                    "connected target: prefix=%s port=%d joints=%d actuators=%d "
                    "free_base=%s",
                    prefix,
                    target.port,
                    len(art.joints),
                    len(art.actuators),
                    art.has_free_base,
                )
            policy_runtime = Go2MoePolicyRuntime(
                policy_path=self.args.policy,
                device=self.args.device,
                dependencies=deps,
                log=self._logger,
            )
            policy_runtime.load(states)

            for state in states:
                client.runtime.set_control_source("ui", articulation=state.prefix)
            self._logger.info("control source set to UI for preflight: %s", list(prefixes))
            client.step(
                n_steps=1,
                observations="standard",
                control_articulations=prefixes,
            )

            for state in states:
                self._preflight_state(state)

            for state in states:
                state.art.set_ctrl(state.current_pose)
            client.step(
                n_steps=1,
                observations="standard",
                control_articulations=prefixes,
            )

            format_pose_sample = deps.require("format_pose_sample")
            for state in states:
                client.runtime.set_control_source("zmq", articulation=state.prefix)
                switched_prefixes.append(state.prefix)
                self._logger.info("control source set to ZMQ for %s", state.prefix)
                self._logger.info(
                    "staged %s current pose sample: %s",
                    state.prefix,
                    format_pose_sample(state.current_pose),
                )

            self._prime_policy_histories(client, states, policy_runtime)
            iters = self._run_control_loop(
                client,
                states,
                prefixes,
                policy_runtime,
                lambda: stop,
            )
        finally:
            self._teardown(client, states, switched_prefixes)

        self._logger.info("multi-web MoE run done after %d shared iterations", iters)
        return 0

    def _preflight_state(self, state: _PolicyState) -> None:
        deps = self.dependencies
        build_compatibility_report = deps.require("build_compatibility_report")
        format_compatibility_report = deps.require("format_compatibility_report")
        safety_abort_reason = deps.require("safety_abort_reason")
        push_gains = deps.require("push_gains")
        capture_actuated_joint_pose = deps.require("capture_actuated_joint_pose")
        validate_target_pose = deps.require("validate_target_pose")

        report = build_compatibility_report(
            state.art,
            max_stand_error=self.args.max_stand_error,
        )
        for line in format_compatibility_report(report).splitlines():
            self._logger.info("%s: %s", state.prefix, line)
        if not report.model_ok:
            raise SystemExit(f"{state.prefix} preflight abort: {report.summary()}")
        if not report.stand_ready:
            if self.args.strict_stand or not self.args.allow_stand_mismatch:
                raise SystemExit(
                    f"{state.prefix} stand preflight abort: {report.summary()}"
                )
            self._logger.warning(
                "%s continuing despite stand mismatch: %s",
                state.prefix,
                report.summary(),
            )

        reason = safety_abort_reason(state.art, min_base_z=self.args.min_base_z)
        if reason is not None:
            raise SystemExit(f"{state.prefix} preflight safety abort: {reason}")

        if self.args.push_gains:
            pushed = push_gains(
                state.art,
                kp=self.args.kp,
                kv=self.args.kv,
                torque_limit=self.args.torque_limit,
            )
            self._logger.info(
                "%s pushed Unitree RL Gym PD gains for %d joints",
                state.prefix,
                pushed,
            )
        else:
            self._logger.info("%s leaving existing UE PD gains unchanged", state.prefix)

        current_pose = capture_actuated_joint_pose(state.art)
        reason = validate_target_pose(current_pose)
        if reason is not None:
            raise SystemExit(f"{state.prefix} captured pose abort: {reason}")
        state.current_pose = {
            str(name): float(value) for name, value in current_pose.items()
        }
        state.previous_target = dict(state.current_pose)

    def _prime_policy_histories(
        self,
        client: Any,
        states: list[_PolicyState],
        policy_runtime: Go2MoePolicyRuntime,
    ) -> None:
        deps = self.dependencies
        sync_runtime_ui = deps.require("sync_command_source_runtime_ui")
        build_observation = deps.require("build_observation")

        observations: list[Any] = []
        commands: list[Any] = []
        for state in states:
            sync_runtime_ui(state.command_source, client, state.prefix)
            state.desired_command = tuple(state.command_source.poll())
            command = state.command_limiter.reset()
            state.applied_command = command
            obs = build_observation(
                state.art,
                command=command,
                last_action=state.last_action,
            )
            observations.append(obs)
            commands.append(command)

        policy_runtime.reset_histories(states, observations)
        for state, command in zip(states, commands, strict=True):
            self._logger.info(
                "primed MoE history for %s; desired_command=%s applied_command=%s",
                state.prefix,
                np.round(state.desired_command, 4).tolist(),
                np.round(command, 4).tolist(),
            )

    def _run_control_loop(
        self,
        client: Any,
        states: list[_PolicyState],
        prefixes: tuple[str, ...],
        policy_runtime: Go2MoePolicyRuntime,
        should_stop: Callable[[], bool],
    ) -> int:
        iters = 0
        for warmup_idx in range(max(0, int(self.args.warmup_steps))):
            if should_stop() or _any_quit_requested(states):
                break
            for state in states:
                state.art.set_ctrl(state.current_pose)
            client.step(
                n_steps=1,
                observations="standard",
                target_hz=self.args.freq,
                control_articulations=prefixes,
            )
            if warmup_idx == 0:
                self._logger.info(
                    "holding captured poses for %d warmup steps",
                    self.args.warmup_steps,
                )

        deadline = (
            None
            if float(self.args.duration) <= 0.0
            else self._clock() + float(self.args.duration)
        )
        while not should_stop() and not _any_quit_requested(states):
            if deadline is not None and self._clock() >= deadline:
                break

            tick_start = self._clock()
            policy_start = self._clock()
            commands: list[Any] = []
            observations: list[Any] = []
            for state in states:
                command, obs = self._prepare_state_observation(client, state)
                commands.append(command)
                observations.append(obs)
            actions, diagnostics = policy_runtime.infer(states, observations)
            for state, command, obs, action, state_diagnostics in zip(
                states,
                commands,
                observations,
                actions,
                diagnostics,
                strict=True,
            ):
                self._stage_state_action(
                    state,
                    command=command,
                    obs=obs,
                    action=action,
                    diagnostics=state_diagnostics,
                )
            policy_duration_s = max(0.0, self._clock() - policy_start)

            step_start = self._clock()
            client.step(
                n_steps=1,
                observations="standard",
                target_hz=self.args.freq,
                control_articulations=prefixes,
            )
            step_duration_s = max(0.0, self._clock() - step_start)
            tick_duration_s = max(0.0, self._clock() - tick_start)
            iters += 1

            for state in states:
                if state.pending_previous_target is not None:
                    state.previous_target = state.pending_previous_target
                if state.pending_last_action is not None:
                    state.last_action = state.pending_last_action
                state.pending_previous_target = None
                state.pending_last_action = None
                state.iters += 1

            if self._post_step_hook is not None:
                self._post_step_hook(states)

            command_statuses = _command_statuses_by_prefix(states)
            self.metrics.record_tick(
                tick_duration_s=tick_duration_s,
                policy_duration_s=policy_duration_s,
                step_duration_s=step_duration_s,
                active_robot_count=sum(
                    1 for status in command_statuses.values() if status.get("active")
                ),
                command_statuses=command_statuses,
            )
            self._maybe_log_metrics()

            if iters == 1 or iters % max(1, int(self.args.freq)) == 0:
                self._logger.info(
                    "step=%d sim=%.2fs targets=%s",
                    iters,
                    client.sim_time,
                    {
                        state.prefix: np.round(state.art.root_pos_w, 3).tolist()
                        for state in states
                    },
                )
        return iters

    def _maybe_log_metrics(self) -> None:
        if self.metrics_log_interval_s <= 0.0:
            return
        now = self._clock()
        if (
            self._last_metrics_log_at is not None
            and (now - self._last_metrics_log_at) < self.metrics_log_interval_s
        ):
            return
        self._last_metrics_log_at = now
        snapshot = self.metrics.snapshot()
        ages = {
            name: robot.get("last_command_age_s")
            for name, robot in snapshot.get("robots", {}).items()
        }
        self._logger.info(
            "metrics: tick=%d missed=%d tick_ms=%.2f policy_ms=%.2f "
            "step_ms=%.2f active=%d command_age_s=%s",
            snapshot["tick_count"],
            snapshot["missed_deadlines"],
            snapshot["last_tick_duration_s"] * 1000.0,
            snapshot["last_policy_duration_s"] * 1000.0,
            snapshot["last_step_duration_s"] * 1000.0,
            snapshot["active_robot_count"],
            ages,
        )

    def _prepare_state_observation(
        self,
        client: Any,
        state: _PolicyState,
    ) -> tuple[Any, Any]:
        deps = self.dependencies
        safety_abort_reason = deps.require("safety_abort_reason")
        sync_runtime_ui = deps.require("sync_command_source_runtime_ui")
        build_observation = deps.require("build_observation")

        reason = safety_abort_reason(state.art, min_base_z=self.args.min_base_z)
        if reason is not None:
            raise SystemExit(f"{state.prefix} safety abort: {reason}")

        stop_if_stale = getattr(state.command_source, "stop_if_stale", None)
        if callable(stop_if_stale):
            stop_if_stale()
        sync_runtime_ui(state.command_source, client, state.prefix)
        state.desired_command = tuple(state.command_source.poll())
        command = state.command_limiter.update(
            state.desired_command,
            dt=self._command_dt,
            brake=_command_source_brake_requested(state.command_source),
        )
        state.applied_command = command
        obs = build_observation(
            state.art,
            command=command,
            last_action=state.last_action,
        )
        return command, obs

    def _stage_state_action(
        self,
        state: _PolicyState,
        *,
        command: Any,
        obs: Any,
        action: Any,
        diagnostics: Any,
    ) -> None:
        deps = self.dependencies
        action_abort_reason = deps.require("action_abort_reason")
        apply_action_limit = deps.require("apply_action_limit")
        action_to_target_pose = deps.require("action_to_target_pose")
        validate_target_pose = deps.require("validate_target_pose")
        maybe_rate_limit_target_pose = deps.require("maybe_rate_limit_target_pose")
        target_pose_to_action = deps.require("target_pose_to_action")
        target_delta_abs_max = deps.require("target_delta_abs_max")

        reason = action_abort_reason(action, max_abs=self.args.max_action_abs)
        if reason is not None:
            if self.limit_mode.action_limit_mode == "abort":
                raise SystemExit(f"{state.prefix} safety abort: {reason}")
            if self.limit_mode.action_limit_mode == "clip":
                self._logger.warning("%s clipping policy action: %s", state.prefix, reason)
            action, _ = apply_action_limit(
                action,
                max_abs=self.args.max_action_abs,
                mode=self.limit_mode.action_limit_mode,
            )

        desired_target = action_to_target_pose(state.art, action)
        reason = validate_target_pose(desired_target)
        if reason is not None:
            raise SystemExit(f"{state.prefix} safety abort: {reason}")
        applied_target = maybe_rate_limit_target_pose(
            state.previous_target,
            desired_target,
            max_step=self.limit_mode.max_target_step,
        )

        state.art.set_ctrl(applied_target)
        state.pending_previous_target = applied_target
        state.pending_last_action = (
            action
            if self.limit_mode.raw_policy
            else target_pose_to_action(applied_target)
        )

        if state.iters == 0:
            self._logger.info(
                "%s desired_cmd=%s applied_cmd=%s obs_norm=%.3f "
                "action[min,max]=[%.3f, %.3f] "
                "raw_delta=%.3f applied_delta=%.3f weights=%s",
                state.prefix,
                np.round(state.desired_command, 3).tolist(),
                np.round(command, 3).tolist(),
                float(np.linalg.norm(obs)),
                float(np.min(action)),
                float(np.max(action)),
                target_delta_abs_max(state.previous_target, desired_target),
                target_delta_abs_max(state.previous_target, applied_target),
                np.round(diagnostics.expert_weights, 3).tolist(),
            )

    def _teardown(
        self,
        client: Any,
        states: list[_PolicyState],
        switched_prefixes: list[str],
    ) -> None:
        sync_runtime_ui = self.dependencies.require("sync_command_source_runtime_ui")
        for state in states:
            release = getattr(state.command_source, "release", None)
            if callable(release):
                try:
                    release()
                    sync_runtime_ui(state.command_source, client, state.prefix)
                except Exception as exc:  # pragma: no cover - teardown best effort
                    self._logger.warning(
                        "failed to clear command-source UI state for %s: %s",
                        state.prefix,
                        exc,
                    )
        if not self.args.leave_zmq:
            for prefix in switched_prefixes:
                try:
                    client.runtime.set_control_source("ui", articulation=prefix)
                    self._logger.info("control source restored to UI for %s", prefix)
                except Exception as exc:  # pragma: no cover - teardown best effort
                    self._logger.warning(
                        "failed to restore UI control source for %s: %s",
                        prefix,
                        exc,
                    )


def _any_quit_requested(states: list[_PolicyState]) -> bool:
    return any(
        bool(getattr(state.command_source, "quit_requested", False))
        for state in states
    )


def _command_source_brake_requested(command_source: Any) -> bool:
    brake = getattr(command_source, "brake_requested", None)
    if callable(brake):
        return bool(brake())
    if brake is not None:
        return bool(brake)
    status_fn = getattr(command_source, "status", None)
    return bool(status_fn().get("brake", False)) if callable(status_fn) else False


def _command_statuses_by_prefix(states: list[_PolicyState]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for state in states:
        status_fn = getattr(state.command_source, "status", None)
        if callable(status_fn):
            status = dict(status_fn())
        else:
            status = {
                "articulation": state.prefix,
                "active": False,
                "stale": False,
                "last_command_age_s": None,
                "twist": [0.0, 0.0, 0.0],
            }
        if not status.get("articulation"):
            status["articulation"] = state.prefix
        status["desired_twist"] = list(state.desired_command)
        status["applied_twist"] = list(state.applied_command)
        status["active"] = bool(status.get("active")) or any(
            abs(axis) > 1e-6 for axis in state.applied_command
        )
        statuses[state.prefix] = status
    return statuses
