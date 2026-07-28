from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import run_manifest


class TestRunManifest(unittest.TestCase):
    def test_environment_and_text_redaction(self) -> None:
        environment = {
            "LLM_API_KEY": "top-secret-key",
            "SOLBENCH_TOKEN": "top-secret-token",
            "SOL58_NUM_ISLANDS": "8",
            "ATREX_LITELLM_DROP_PARAMS": "1",
            "SOL_EXECBENCH": "/opt/sol-execbench",
            "UNRELATED": "ignored",
        }

        selected = run_manifest.selected_environment(environment)
        text = run_manifest.redact_text(
            'api_key: "top-secret-key"\n'
            '"access_token": "hardcoded-token"\n'
            "client_secret=hardcoded-secret\n"
            "note: top-secret-token\n",
            environment,
        )

        self.assertEqual(selected["LLM_API_KEY"], "<redacted>")
        self.assertEqual(selected["SOLBENCH_TOKEN"], "<redacted>")
        self.assertEqual(selected["SOL58_NUM_ISLANDS"], "8")
        self.assertEqual(selected["ATREX_LITELLM_DROP_PARAMS"], "1")
        self.assertEqual(selected["SOL_EXECBENCH"], "/opt/sol-execbench")
        self.assertNotIn("UNRELATED", selected)
        self.assertNotIn("top-secret", text)
        self.assertNotIn("hardcoded", text)
        self.assertIn('api_key: "<redacted>"', text)
        self.assertIn('"access_token": "<redacted>"', text)
        self.assertIn('client_secret="<redacted>"', text)

    def test_argument_redaction(self) -> None:
        self.assertEqual(
            run_manifest.redact_arguments(
                ["--api-key", "secret", "--token=value", "--max-iterations", "20"]
            ),
            [
                "--api-key",
                "<redacted>",
                "--token=<redacted>",
                "--max-iterations",
                "20",
            ],
        )

    def test_manifest_records_resolved_sources_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rendered_config = root / "rendered.yaml"
            rendered_task = root / "task.txt"
            initial_environment = root / "initial.json"
            config_copy = root / "record" / "resolved.yaml"
            task_copy = root / "record" / "task.txt"
            rendered_config.write_text(
                'api_key: "secret"\nmax_iterations: 40\n', encoding="utf-8"
            )
            rendered_task.write_text("optimize kernel", encoding="utf-8")
            initial_environment.write_text(
                json.dumps(
                    {
                        "environment": {
                            "LLM_API_KEY": "<redacted>",
                            "SOL58_NUM_ISLANDS": "4",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                run_manifest.os.environ,
                {
                    "LLM_API_KEY": "secret",
                    "SOL58_NUM_ISLANDS": "8",
                    "SOL58_MAX_ITERATIONS": "40",
                },
                clear=True,
            ):
                manifest = run_manifest.build_manifest(
                    task="sol58",
                    run_dir=str(root),
                    workspace=str(root / "output"),
                    arguments=["--num-islands", "8"],
                    cli_keys=["SOL58_NUM_ISLANDS"],
                    initial_environment=run_manifest._load_initial_environment(
                        initial_environment
                    ),
                    repositories={"atrex": Path(__file__).parents[1]},
                    source_files={"config": rendered_config},
                    rendered_config=str(rendered_config),
                    rendered_task=str(rendered_task),
                    config_copy=str(config_copy),
                    task_copy=str(task_copy),
                )

            resolved = manifest["configuration"]["resolved_environment"]
            self.assertEqual(resolved["SOL58_NUM_ISLANDS"]["source"], "cli")
            self.assertEqual(
                resolved["SOL58_MAX_ITERATIONS"]["source"],
                "runner_default_or_derived",
            )
            self.assertEqual(resolved["LLM_API_KEY"]["value"], "<redacted>")
            self.assertEqual(len(manifest["configuration"]["fingerprint_sha256"]), 64)
            self.assertNotIn("secret", config_copy.read_text(encoding="utf-8"))
            self.assertIn("commit", manifest["repositories"]["atrex"])


if __name__ == "__main__":
    unittest.main()
