from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.loongflow_compat.upstream_contract import (
    UpstreamContractError,
    validate_upstream_checkout,
)


class TestUpstreamContract(unittest.TestCase):
    def _repository(self, root: Path) -> tuple[Path, str]:
        project = root / "project"
        project.mkdir()
        (project / "required.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Test"], check=True
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-qm", "initial"], check=True
        )
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        return project, commit

    def test_accepts_pinned_clean_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, commit = self._repository(root)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "expected_commit": commit,
                        "required_paths": ["required.py"],
                    }
                ),
                encoding="utf-8",
            )

            result = validate_upstream_checkout(project, contract)

            self.assertTrue(result["commit_matches"])
            self.assertFalse(result["tracked_dirty"])

    def test_rejects_commit_mismatch_and_tracked_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, commit = self._repository(root)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "expected_commit": "0" * 40,
                        "required_paths": ["required.py"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(UpstreamContractError, "commit mismatch"):
                validate_upstream_checkout(project, contract)

            contract.write_text(
                json.dumps(
                    {
                        "expected_commit": commit,
                        "required_paths": ["required.py"],
                    }
                ),
                encoding="utf-8",
            )
            (project / "required.py").write_text("value = 2\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    UpstreamContractError, "tracked modifications"
                ):
                    validate_upstream_checkout(project, contract)


if __name__ == "__main__":
    unittest.main()
