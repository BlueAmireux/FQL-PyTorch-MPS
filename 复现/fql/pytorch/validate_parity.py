"""CPU-only, fixed-input comparison with the unchanged official JAX implementation.

The JAX export and PyTorch check run in separate interpreters. The production
PyTorch environment does not import JAX. Fixtures are diagnostic inputs, not a
converted training checkpoint or a promise of identical cross-framework RNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from collections.abc import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BATCH_PATH = ROOT / "experiment/fql_state_setup/inspection/real_batch.npz"
CODE_PATH = ROOT / "复现/fql/code"
LOSS_KEYS = (
    "critic/critic_loss",
    "actor/bc_flow_loss",
    "actor/distill_loss",
    "actor/q_loss",
)
GROUPS = ("critic", "target_critic", "actor_bc_flow", "actor_onestep_flow")
TOLERANCES = {
    "forward": {"atol": 2e-5, "rtol": 2e-5},
    "loss": {"atol": 2e-5, "rtol": 2e-5},
    "gradient": {"atol": 5e-5, "rtol": 5e-4},
    "adam": {"atol": 1e-4, "rtol": 1e-3},
    "moment": {"atol": 5e-5, "rtol": 5e-4},
}


def source_hashes():
    """Hash both reference and candidate source; never mutate legacy evidence."""
    paths = (
        BATCH_PATH, CODE_PATH / "agents/fql.py", CODE_PATH / "utils/networks.py",
        CODE_PATH / "utils/flax_utils.py", Path(__file__),
        Path(__file__).parent / "fql_torch/agent.py",
        Path(__file__).parent / "fql_torch/networks.py",
        Path(__file__).parent / "fql_torch/__init__.py",
    )
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.exists()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(tree, prefix=()):
    if isinstance(tree, Mapping):
        result = {}
        for name, value in tree.items():
            result.update(flatten(value, (*prefix, str(name))))
        return result
    return {"/".join(prefix): np.asarray(tree)}


def unflatten(flat):
    root = {}
    for path, value in flat.items():
        parts = path.split("/")
        current = root
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return root


def add_tree(arrays, prefix, tree):
    arrays.update({f"{prefix}::{path}": value for path, value in flatten(tree).items()})


def select_tree(arrays, prefix):
    marker = f"{prefix}::"
    return unflatten({key[len(marker):]: arrays[key] for key in arrays if key.startswith(marker)})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export_jax(output):
    # Set before importing JAX: numerical fixtures must never initialize Metal.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "false"
    sys.path.insert(0, str(CODE_PATH))
    import jax
    import jax.numpy as jnp
    from agents.fql import FQLAgent, get_config

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "fixture.npz").exists():
        raise FileExistsError("Refusing to replace an existing fixture; choose a new attempt directory")
    write_json(output / "protocol.json", {
        "purpose": "CPU float32 fixed author weights, real batch, and explicit randomness parity",
        "tolerances": TOLERANCES,
        "seed": 0,
        "batch": str(BATCH_PATH.relative_to(ROOT)),
        "expected_gradient_owners": {
            LOSS_KEYS[0]: ["critic"],
            LOSS_KEYS[1]: ["actor_bc_flow"],
            LOSS_KEYS[2]: ["actor_onestep_flow"],
            LOSS_KEYS[3]: ["actor_onestep_flow"],
        },
        "target_order": "Polyak update from pre-Adam critic and target, as in the original implementation",
        "limitations": "one initial float32 CPU batch; no long-run or cross-framework bitwise claim",
    })
    started = time.perf_counter()
    raw = np.load(BATCH_PATH, allow_pickle=False)
    batch = {k: jnp.asarray(raw[k], dtype=jnp.float32) for k in (
        "observations", "next_observations", "actions", "rewards", "masks", "terminals"
    )}
    config = get_config()
    config.alpha = 300.0
    config.encoder = None
    agent = FQLAgent.create(0, batch["observations"], batch["actions"], config)
    config_dict = dict(agent.config)
    write_json(output / "config.json", config_dict)
    arrays = {}
    for name, value in batch.items():
        arrays[f"batch::{name}"] = np.asarray(value)
    add_tree(arrays, "init", agent.network.params)

    # Follow the exact splits in original update -> total_loss -> critic/actor.
    _, update_rng = jax.random.split(agent.rng)
    _, actor_rng, critic_rng = jax.random.split(update_rng, 3)
    _, sample_rng = jax.random.split(critic_rng)
    next_action_rng, _ = jax.random.split(sample_rng)
    actor_rng, x_rng, t_rng = jax.random.split(actor_rng, 3)
    actor_rng, distill_rng = jax.random.split(actor_rng)
    metric_rng, _ = jax.random.split(actor_rng)
    shape = batch["actions"].shape
    randomness = {
        "next_noises": jax.random.normal(next_action_rng, shape),
        "x0": jax.random.normal(x_rng, shape),
        "t": jax.random.uniform(t_rng, (shape[0], 1)),
        "distill_noises": jax.random.normal(distill_rng, shape),
        "metric_noises": jax.random.normal(metric_rng, shape),
    }
    for name, value in randomness.items():
        arrays[f"random::{name}"] = np.asarray(value)
    n = agent.network
    obs, actions = batch["observations"], batch["actions"]
    next_actions = jnp.clip(n.select("actor_onestep_flow")(
        batch["next_observations"], randomness["next_noises"]), -1, 1)
    x_t = (1 - randomness["t"]) * randomness["x0"] + randomness["t"] * actions
    student_raw = n.select("actor_onestep_flow")(obs, randomness["distill_noises"])
    forwards = {
        "critic_q": n.select("critic")(obs, actions),
        "next_actions": next_actions,
        "target_qs": n.select("target_critic")(batch["next_observations"], next_actions),
        "bc_prediction": n.select("actor_bc_flow")(obs, x_t, randomness["t"]),
        "teacher_actions": agent.compute_flow_actions(obs, randomness["distill_noises"]),
        "student_raw": student_raw,
        "student_qs": n.select("critic")(obs, jnp.clip(student_raw, -1, 1)),
        "metric_actions": jnp.clip(n.select("actor_onestep_flow")(
            obs, randomness["metric_noises"]), -1, 1),
    }
    add_tree(arrays, "forward", forwards)
    total, metrics = agent.total_loss(batch, agent.network.params, rng=update_rng)
    arrays["loss::total"] = np.asarray(total)
    add_tree(arrays, "loss", metrics)
    for loss_key in LOSS_KEYS:
        fn = jax.jit(jax.grad(lambda params: agent.total_loss(
            batch, params, rng=update_rng)[1][loss_key]))
        grads = fn(agent.network.params)
        jax.block_until_ready(grads)
        add_tree(arrays, f"gradient::{loss_key}", grads)
        print(f"Exported gradient {loss_key}", flush=True)
    total_grad = jax.jit(jax.grad(lambda params: agent.total_loss(
        batch, params, rng=update_rng)[0]))(agent.network.params)
    add_tree(arrays, "gradient::total", total_grad)
    updated, update_metrics = agent.update(batch)
    jax.block_until_ready(updated)
    add_tree(arrays, "updated", updated.network.params)
    add_tree(arrays, "adam_mu", updated.network.opt_state[0].mu)
    add_tree(arrays, "adam_nu", updated.network.opt_state[0].nu)
    add_tree(arrays, "update_metric", update_metrics)
    np.savez_compressed(output / "fixture.npz", **arrays)
    write_json(output / "fixture_manifest.json", {
        "status": "exported", "device": str(jax.devices()[0]), "jax": jax.__version__,
        "dtype": "float32", "seed": 0, "batch_size": int(shape[0]),
        "initial_parameter_leaves": len(flatten(agent.network.params)),
        "updated_step": int(updated.network.step), "elapsed_seconds": time.perf_counter() - started,
        "fixture_sha256": sha256(output / "fixture.npz"),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            BATCH_PATH, CODE_PATH / "agents/fql.py", CODE_PATH / "utils/networks.py",
            CODE_PATH / "utils/flax_utils.py")},
        "source_sha256_at_export": source_hashes(),
        "tolerances": TOLERANCES,
    })
    print(json.dumps({"status": "exported", "output": str(output)}, ensure_ascii=False), flush=True)


def compare_arrays(reference, candidate, tolerances):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {"passed": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    bound = tolerances["atol"] + tolerances["rtol"] * np.abs(reference.astype(np.float64))
    ratio = np.divide(difference, bound, out=np.zeros_like(difference), where=bound > 0)
    ratio[(bound == 0) & (difference > 0)] = np.inf
    return {
        "passed": bool(np.isfinite(reference).all() and np.isfinite(candidate).all() and (difference <= bound).all()),
        "shape": list(reference.shape), "max_absolute_difference": float(difference.max(initial=0)),
        "maximum_tolerance_ratio": float(ratio.max(initial=0)),
        "elements_outside_tolerance": int((difference > bound).sum()),
        **tolerances,
    }


def compare_trees(reference, candidate, tolerances):
    ref, cand = flatten(reference), flatten(candidate)
    leaves = {path: compare_arrays(ref[path], cand[path], tolerances) for path in sorted(ref.keys() & cand.keys())}
    missing, extra = sorted(ref.keys() - cand.keys()), sorted(cand.keys() - ref.keys())
    return {
        "passed": not missing and not extra and all(result["passed"] for result in leaves.values()),
        "missing": missing, "extra": extra, "leaf_count": len(leaves),
        "max_absolute_difference": max((x.get("max_absolute_difference", float("inf")) for x in leaves.values()), default=0),
        "maximum_tolerance_ratio": max((x.get("maximum_tolerance_ratio", float("inf")) for x in leaves.values()), default=0),
        "leaves": leaves,
    }


def flax_tensor_tree(agent, kind="grad"):
    """Map gradients/Adam buffers to the exact Flax leaf layout without JAX."""
    result = {}
    for group in GROUPS:
        module = getattr(agent, group)
        mlp = module.value_net if group in ("critic", "target_critic") else module.mlp
        subtree = {}
        for prefix, layers in (("Dense", mlp.layers), ("LayerNorm", mlp.norms)):
            for index, layer in enumerate(layers):
                leaf = {}
                for attribute, flax_name in (("weight", "kernel" if prefix == "Dense" else "scale"), ("bias", "bias")):
                    parameter = getattr(layer, attribute)
                    if kind == "grad":
                        tensor = parameter.grad
                    else:
                        tensor = agent.optimizer.state.get(parameter, {}).get(kind)
                    value = np.zeros(tuple(parameter.shape), dtype=np.float32) if tensor is None else tensor.detach().cpu().numpy()
                    if prefix == "Dense" and attribute == "weight":
                        value = np.swapaxes(value, -1, -2)
                    leaf[flax_name] = value
                subtree[f"{prefix}_{index}"] = leaf
        result[f"modules_{group}"] = {"value_net" if group in ("critic", "target_critic") else "mlp": subtree}
    return result


def gradient_owners(tree):
    result = {}
    for group in GROUPS:
        values = list(flatten(tree[f"modules_{group}"]).values())
        result[group] = {
            "nonzero_elements": sum(int(np.count_nonzero(value)) for value in values),
            "l2_norm": float(np.sqrt(sum(float(np.sum(value.astype(np.float64) ** 2)) for value in values))),
        }
    return result


def validate_torch(fixture_dir, report_path, device="cpu", optimizer_impl="foreach"):
    import torch
    from fql_torch.agent import FQLAgent

    fixture_dir, report_path = Path(fixture_dir), Path(report_path)
    if report_path.exists():
        raise FileExistsError("Refusing to replace an existing parity report")
    protocol_path = report_path.with_suffix(".protocol.json")
    if protocol_path.exists():
        raise FileExistsError("Refusing to replace an existing parity protocol; choose a new report path")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    hashes_before = source_hashes()
    fixture_manifest = json.loads((fixture_dir / "fixture_manifest.json").read_text())
    fixture_hash = sha256(fixture_dir / "fixture.npz")
    # MPS starts with the exact same thresholds as CPU. Failures remain failures.
    write_json(protocol_path, {
        "purpose": "fixed CPU JAX reference compared with selected PyTorch device/optimizer",
        "device": device, "optimizer_impl": optimizer_impl,
        "tolerances": TOLERANCES,
        "target_order_tolerance": {"atol": 1e-7, "rtol": 0.},
        "fixture_sha256": fixture_hash,
        "fixture_manifest_sha256": sha256(fixture_dir / "fixture_manifest.json"),
        "source_sha256_before": hashes_before,
        "tolerance_policy": "CPU and MPS use the previously declared CPU thresholds; no automatic relaxation",
        "limitations": "one float32 batch, explicit randomness; no long-run/bitwise claim",
    })
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; CPU fallback is forbidden")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("MPS parity requires PYTORCH_ENABLE_MPS_FALLBACK disabled")
    torch.set_num_threads(8)
    started = time.perf_counter()
    config = json.loads((fixture_dir / "config.json").read_text())
    config["optimizer_impl"] = optimizer_impl
    arrays = np.load(fixture_dir / "fixture.npz", allow_pickle=False)
    initial_params = select_tree(arrays, "init")
    agent = FQLAgent(config, device=device, seed=0)
    agent.load_flax_params(initial_params)
    batch = {key: torch.from_numpy(value.copy()).to(device) for key, value in select_tree(arrays, "batch").items()}
    randomness = {key: torch.from_numpy(value.copy()).to(device) for key, value in select_tree(arrays, "random").items()}
    report = {
        "purpose": "one fixed real batch numerical migration validation",
        "device": device, "optimizer_impl": optimizer_impl, "torch": torch.__version__,
        "tolerances": TOLERANCES,
        "fixture_sha256": fixture_hash, "checks": {},
        "fixture_manifest_sha256": sha256(fixture_dir / "fixture_manifest.json"),
        "fixture_input_sha256": fixture_manifest["input_sha256"],
        "source_sha256_before": hashes_before,
        "limitations": "float32 approximate parity against CPU JAX; no cross-framework bitwise/long-run score claim",
    }
    report["checks"]["fixture_and_reference_integrity"] = {
        "fixture_matches_manifest": fixture_hash == fixture_manifest["fixture_sha256"],
        "reference_inputs_match_manifest": all(hashes_before.get(key) == value for key, value in fixture_manifest["input_sha256"].items()),
    }
    report["checks"]["fixture_and_reference_integrity"]["passed"] = all(report["checks"]["fixture_and_reference_integrity"].values())
    report["checks"]["requested_device_and_optimizer"] = {
        "actual_device": str(agent.device),
        "optimizer_foreach": agent.optimizer.defaults.get("foreach"),
        "optimizer_fused": agent.optimizer.defaults.get("fused"),
        "passed": agent.device.type == device and (
            agent.optimizer.defaults.get("fused") is True if optimizer_impl == "fused"
            else agent.optimizer.defaults.get("foreach") is True),
    }
    report["checks"]["parameter_mapping"] = compare_trees(initial_params, agent.export_flax_params(), {"atol": 0., "rtol": 0.})
    obs, actions = batch["observations"], batch["actions"]
    with torch.no_grad():
        next_actions = agent.actor_onestep_flow(batch["next_observations"], randomness["next_noises"]).clamp(-1, 1)
        x_t = (1 - randomness["t"]) * randomness["x0"] + randomness["t"] * actions
        student = agent.actor_onestep_flow(obs, randomness["distill_noises"])
        forward = {
            "critic_q": agent.critic(obs, actions), "next_actions": next_actions,
            "target_qs": agent.target_critic(batch["next_observations"], next_actions),
            "bc_prediction": agent.actor_bc_flow(obs, x_t, randomness["t"]),
            "teacher_actions": agent.compute_flow_actions(obs, randomness["distill_noises"]),
            "student_raw": student, "student_qs": agent.critic(obs, student.clamp(-1, 1)),
            "metric_actions": agent.actor_onestep_flow(obs, randomness["metric_noises"]).clamp(-1, 1),
        }
    forward = {key: value.detach().cpu().numpy() for key, value in forward.items()}
    report["checks"]["forward"] = compare_trees(select_tree(arrays, "forward"), forward, TOLERANCES["forward"])
    total, metrics = agent.compute_losses(batch, randomness)
    actual_losses = {key: value.detach().cpu().numpy() for key, value in metrics.items() if f"loss::{key}" in arrays}
    actual_losses["total"] = total.detach().cpu().numpy()
    report["checks"]["losses"] = compare_trees(select_tree(arrays, "loss"), actual_losses, TOLERANCES["loss"])
    expected_owners = {
        LOSS_KEYS[0]: {"critic"}, LOSS_KEYS[1]: {"actor_bc_flow"},
        LOSS_KEYS[2]: {"actor_onestep_flow"}, LOSS_KEYS[3]: {"actor_onestep_flow"},
    }
    gradient_checks = {}
    for loss_key in LOSS_KEYS:
        agent.optimizer.zero_grad(set_to_none=True)
        total, metrics = agent.compute_losses(batch, randomness)
        metrics[loss_key].backward()
        gradient_tree = flax_tensor_tree(agent)
        result = compare_trees(select_tree(arrays, f"gradient::{loss_key}"), gradient_tree, TOLERANCES["gradient"])
        result["owners"] = gradient_owners(gradient_tree)
        actual_owners = {name for name, stats in result["owners"].items() if stats["nonzero_elements"] > 0}
        result["owner_check_passed"] = actual_owners == expected_owners[loss_key]
        result["passed"] = result["passed"] and result["owner_check_passed"]
        gradient_checks[loss_key] = result
    report["checks"]["individual_gradients"] = {
        "passed": all(x["passed"] for x in gradient_checks.values()), "losses": gradient_checks,
    }
    agent.optimizer.zero_grad(set_to_none=True)
    total, _ = agent.compute_losses(batch, randomness)
    total.backward()
    report["checks"]["total_gradient"] = compare_trees(select_tree(arrays, "gradient::total"), flax_tensor_tree(agent), TOLERANCES["gradient"])

    old_critic = {key: tensor.detach().clone() for key, tensor in agent.critic.state_dict().items()}
    old_target = {key: tensor.detach().clone() for key, tensor in agent.target_critic.state_dict().items()}
    agent.optimizer.zero_grad(set_to_none=True)
    agent.update(batch, randomness=randomness)
    report["checks"]["adam_once"] = compare_trees(select_tree(arrays, "updated"), agent.export_flax_params(), TOLERANCES["adam"])
    report["checks"]["adam_first_moment"] = compare_trees(select_tree(arrays, "adam_mu"), flax_tensor_tree(agent, "exp_avg"), TOLERANCES["moment"])
    report["checks"]["adam_second_moment"] = compare_trees(select_tree(arrays, "adam_nu"), flax_tensor_tree(agent, "exp_avg_sq"), TOLERANCES["moment"])
    tau = config["tau"]
    target_candidates = {key: (1 - tau) * old_target[key] + tau * old_critic[key] for key in old_target}
    target_check = compare_trees(
        {key: tensor.cpu().numpy() for key, tensor in target_candidates.items()},
        {key: tensor.detach().cpu().numpy() for key, tensor in agent.target_critic.state_dict().items()},
        {"atol": 1e-7, "rtol": 0.})
    target_check["uses_pre_adam_critic"] = True
    target_check["one_optimizer"] = isinstance(agent.optimizer, torch.optim.Adam)
    target_check["passed"] = target_check["passed"] and target_check["one_optimizer"]
    report["checks"]["target_order_and_single_adam"] = target_check
    report["source_sha256_after"] = source_hashes()
    report["checks"]["source_unchanged_during_validation"] = {
        "passed": report["source_sha256_before"] == report["source_sha256_after"],
    }
    report["passed"] = all(result["passed"] for result in report["checks"].values())
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(json.dumps({"passed": report["passed"], "report": str(report_path),
                      "checks": {key: value["passed"] for key, value in report["checks"].items()}}, ensure_ascii=False), flush=True)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    export = subparsers.add_parser("export-jax", help="run with the preserved CPU JAX interpreter")
    export.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("validate-torch", aliases=["check"], help="run with the independent PyTorch interpreter")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    check.add_argument("--optimizer-impl", choices=("foreach", "fused"), default="foreach")
    args = parser.parse_args()
    if args.mode == "export-jax":
        export_jax(args.output)
    else:
        try:
            passed = validate_torch(args.fixture, args.report, args.device, args.optimizer_impl)
        except Exception as error:
            if not args.report.exists():
                args.report.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.report, {
                    "passed": False, "status": "error", "device": args.device,
                    "optimizer_impl": args.optimizer_impl, "tolerances": TOLERANCES,
                    "error_type": type(error).__name__, "error": str(error),
                    "source_sha256_after_error": source_hashes(),
                })
            traceback.print_exc()
            raise SystemExit(1)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
