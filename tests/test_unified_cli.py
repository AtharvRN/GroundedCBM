import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_cbm_module():
    spec = importlib.util.spec_from_file_location("cbm_cli", ROOT / "scripts" / "cbm.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestUnifiedCli(unittest.TestCase):
    def setUp(self) -> None:
        self.cbm = load_cbm_module()

    def test_flat_yaml_config_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "dataset: cub",
                        "model_name: savlg_cbm",
                        "save_dir: artifacts/cub",
                        "seed: 6885",
                        "train_glm_after_cbl: false",
                        "saga_lam: 0.0002",
                    ]
                ),
                encoding="utf-8",
            )

            config = self.cbm._load_flat_config(str(path))

        self.assertEqual(config["dataset"], "cub")
        self.assertEqual(config["model_name"], "savlg_cbm")
        self.assertEqual(config["seed"], 6885)
        self.assertEqual(config["train_glm_after_cbl"], False)
        self.assertAlmostEqual(config["saga_lam"], 0.0002)

    def test_cub_yaml_config_json_is_forwarded_to_train_cbm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "cub.yaml"
            json_path = Path(tmpdir) / "cub.json"
            yaml_path.write_text(
                f"dataset: cub\nmodel_name: savlg_cbm\nconfig_json: {json_path}\n",
                encoding="utf-8",
            )
            json_path.write_text(json.dumps({"dataset": "cub", "model_name": "savlg_cbm"}), encoding="utf-8")

            with mock.patch.object(self.cbm, "_run_script") as run_script:
                self.cbm.cmd_train(["--config", str(yaml_path), "--model", "gcbm", "--seed", "7"])

        run_script.assert_called_once()
        script, forwarded = run_script.call_args.args
        self.assertEqual(script, "train_cbm.py")
        self.assertIn("--config", forwarded)
        self.assertEqual(forwarded[forwarded.index("--config") + 1], str(json_path))
        self.assertIn("--model_name", forwarded)
        self.assertEqual(forwarded[forwarded.index("--model_name") + 1], "savlg_cbm")
        self.assertIn("--seed", forwarded)
        self.assertEqual(forwarded[forwarded.index("--seed") + 1], "7")

    def test_imagenet_train_rejects_non_gcbm_model(self) -> None:
        with self.assertRaises(SystemExit):
            self.cbm.cmd_train(["--dataset", "imagenet", "--model", "salf"])

    def test_test_command_forwards_sparse_eval_options(self) -> None:
        with mock.patch.object(self.cbm, "_run_script") as run_script:
            self.cbm.cmd_test(
                [
                    "--load_path",
                    "artifacts/run",
                    "--lam",
                    "0.1",
                    "--max_images",
                    "8",
                    "--lf_cbm",
                ]
            )

        script, forwarded = run_script.call_args.args
        self.assertEqual(script, "sparse_evaluation.py")
        self.assertEqual(forwarded[:2], ["--load_path", "artifacts/run"])
        self.assertIn("--lam", forwarded)
        self.assertEqual(forwarded[forwarded.index("--lam") + 1], "0.1")
        self.assertIn("--max_images", forwarded)
        self.assertEqual(forwarded[forwarded.index("--max_images") + 1], "8")
        self.assertIn("--lf-cbm", forwarded)


if __name__ == "__main__":
    unittest.main()
