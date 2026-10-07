"""Custom hooks for DepthGate training."""

from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from .depth_gate import DepthGate


@HOOKS.register_module()
class LogDepthGateHook(Hook):
    """After each train iter, push the per-level gate mean into the log buffer.

    Appears in the training log as `gate/L<idx>` and as `gate/mean` (avg over
    all DepthGate modules). Lets you see whether the model is leaning on RGB
    (g→1) or depth (g→0) at each FPN level.
    """

    priority = "NORMAL"
    _debug_emitted = False

    def __init__(self, interval: int = 1):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (batch_idx + 1) % self.interval != 0:
            return
        model = runner.model
        if hasattr(model, "module"):
            model = model.module

        if not LogDepthGateHook._debug_emitted:
            found = [(n, m.last_gate_mean)
                     for n, m in model.named_modules() if isinstance(m, DepthGate)]
            runner.logger.info(
                f"[LogDepthGateHook] DEBUG found {len(found)} DepthGate modules: {found}"
            )
            LogDepthGateHook._debug_emitted = True

        scalars: dict[str, float] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, DepthGate):
                continue
            short = name.split(".")[-1]
            if mod.last_gate_mean is not None:
                scalars[f"gate/L{short}"] = mod.last_gate_mean
            if mod.last_alpha_mean is not None:
                scalars[f"alpha/L{short}"] = mod.last_alpha_mean

        if not scalars:
            return

        g_vals = [v for k, v in scalars.items() if k.startswith("gate/")]
        a_vals = [v for k, v in scalars.items() if k.startswith("alpha/")]
        if g_vals:
            scalars["gate/mean"]  = sum(g_vals) / len(g_vals)
        if a_vals:
            scalars["alpha/mean"] = sum(a_vals) / len(a_vals)

        msg = " | ".join(f"{k}={v:.3f}" for k, v in scalars.items())
        runner.logger.info(f"[LogDepthGateHook] iter={runner.iter} {msg}")

        for k, v in scalars.items():
            runner.message_hub.update_scalar(k, v)
