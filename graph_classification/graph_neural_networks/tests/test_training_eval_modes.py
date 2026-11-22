import ast
import math
import os
import runpy
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
# The package initializer also imports HGP-SL and its optional extensions.
GIN = runpy.run_path(str(ROOT / "models/gconv.py"))["GIN"]
RGIN = runpy.run_path(str(ROOT / "models/rgconv.py"))["RGIN"]


def model_args(dropout):
    return SimpleNamespace(
        num_features=8, hidden_dim=16, nhid=16, num_classes=2,
        num_relations=2, dropout_ratio=dropout, additional={}, epochs=3,
    )


def graphs(count, seed):
    random = torch.Generator().manual_seed(seed)
    output = []
    for index in range(count):
        n = 5 + index % 7
        edges = torch.randint(n, (2, 2 * n + index), generator=random)
        relation = F.one_hot(torch.arange(edges.shape[1]) % 2, 2).float()
        output.append(Data(
            x=torch.randn(n, 8, generator=random), edge_index=edges,
            edge_attr=relation, y=torch.tensor([index % 2]),
        ))
    return output


def train_on_toy_graphs():
    model = GIN(model_args(0.5))
    bn = next(layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d))
    train_loader = DataLoader(graphs(16, 5), batch_size=4)
    valid_loader = DataLoader(graphs(8, 6), batch_size=4)
    start_params = [param.detach().clone() for param in model.parameters()]
    recorded = SimpleNamespace(
        forward=[], updates=[], bn_steps=[], bn_valid=[], losses={}, predictions=[],
    )
    phase = "train"

    def record_forward(layer, _):
        recorded.forward.append((phase, layer.training, bn.training))

    class RecordingAdam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            gradients = [param.grad for param in model.parameters() if param.grad is not None]
            finite = bool(gradients) and all(torch.isfinite(grad).all() for grad in gradients)
            recorded.updates.append((model.training, finite))
            return super().step(*args, **kwargs)

    def record_scalar(name, value, _):
        recorded.losses.setdefault(name, []).append(float(value))

    with tempfile.TemporaryDirectory() as directory:
        args = SimpleNamespace(epochs=3, device="cpu", save_model_dir=directory, patience=100)
        # Execute both real functions without importing dataset loading or logging extensions.
        path = ROOT / "main.py"
        functions = [node for node in ast.parse(path.read_bytes()).body
                     if isinstance(node, ast.FunctionDef) and node.name in {"train", "eval_test"}]
        scope = {
            "model": model, "args": args, "torch": torch, "F": F, "os": os, "shutil": shutil,
            "accuracy_score": accuracy_score, "f1_score": f1_score,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
        eval_test = scope["eval_test"]

        def evaluate(loader):
            nonlocal phase
            phase = "eval"
            mean = bn.running_mean.clone()
            count = bn.num_batches_tracked.item()
            result = eval_test(loader)
            recorded.bn_steps.append(count)
            recorded.bn_valid.append(
                torch.equal(mean, bn.running_mean) and bn.num_batches_tracked.item() == count
            )
            phase = "train"
            return result

        scope["eval_test"] = evaluate
        handle = model.register_forward_pre_hook(record_forward)
        start_time = time.perf_counter()
        try:
            scope["train"](
                args, model, RecordingAdam(model.parameters(), lr=1e-3),
                train_loader, valid_loader, writer=SimpleNamespace(add_scalar=record_scalar),
            )
        finally:
            handle.remove()
        recorded.seconds = time.perf_counter() - start_time
        restored = GIN(model_args(0.5))
        restored.load_state_dict(torch.load(Path(directory) / "best.pt", map_location="cpu"))
        phase = "test"
        handle = model.register_forward_pre_hook(record_forward)
        try:
            with torch.no_grad():
                for _ in range(2):
                    recorded.predictions.append(eval_test(valid_loader)["prediction"]["pred"])
        finally:
            handle.remove()
    recorded.updated = any(not torch.equal(p, old) for p, old in zip(model.parameters(), start_params))
    return recorded


class TrainingEvaluationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)

    def test_training_mode_and_batchnorm_across_three_epochs(self):
        result = train_on_toy_graphs()
        self.assertTrue(result.updated)
        self.assertTrue(all(math.isfinite(loss) for losses in result.losses.values() for loss in losses))
        self.assertEqual(result.forward, ([("train", True, True)] * 4 + [("eval", False, False)] * 2) * 3
                         + [("test", False, False)] * 4)
        self.assertEqual(result.updates, [(True, True)] * 12)
        self.assertEqual(result.bn_steps, [4, 8, 12])
        self.assertEqual(result.bn_valid, [True, True, True])

    def check_eval(self, model_type):
        batch = Batch.from_data_list(graphs(3, 11))
        model = model_type(model_args(0.5)).eval()
        oracle = model_type(model_args(0.0)).eval()
        oracle.load_state_dict(model.state_dict())
        with torch.no_grad():
            expected = oracle(batch)
            for _ in range(4):
                actual = model(batch)
                self.assertEqual(actual.shape, (3, 2))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_gin_evaluation_equals_zero_dropout(self):
        self.check_eval(GIN)

    def test_rgin_evaluation_equals_zero_dropout(self):
        self.check_eval(RGIN)

    def test_training_dropout_remains_active(self):
        batch = Batch.from_data_list(graphs(3, 12))
        for model_type in (GIN, RGIN):
            with self.subTest(model=model_type.__name__):
                model = model_type(model_args(0.5)).train()
                with torch.no_grad(), patch.object(F, "dropout", wraps=F.dropout) as dropout:
                    model(batch)
                self.assertEqual(dropout.call_count, model.no_layers)
                for call in dropout.call_args_list:
                    self.assertEqual(call.kwargs["p"], 0.5)
                    self.assertIs(call.kwargs.get("training", True), True)


if __name__ == "__main__":
    unittest.main()
