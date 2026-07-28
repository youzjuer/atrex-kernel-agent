from __future__ import annotations

import json
import hashlib
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
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        return project, commit

    def _contract(
        self,
        root: Path,
        project: Path,
        commit: str,
        *,
        expected_hash: str | None = None,
        parameters: list[str] | None = None,
    ) -> Path:
        target = project / "required.py"
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "expected_commit": commit,
                    "targets": {
                        "required.py": {
                            "sha256": expected_hash
                            or hashlib.sha256(target.read_bytes()).hexdigest(),
                            "callables": {
                                "target": parameters or ["value"],
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return contract

    def test_accepts_pinned_clean_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, commit = self._repository(root)
            (project / "required.py").write_text(
                "def target(value=1):\n    return value\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "add callable"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            contract = self._contract(root, project, commit)

            result = validate_upstream_checkout(project, contract)

            self.assertTrue(result["commit_matches"])
            self.assertFalse(result["tracked_dirty"])
            self.assertTrue(result["targets"]["required.py"]["hash_matches"])
            self.assertEqual(
                result["targets"]["required.py"]["callables"]["target"],
                ["value"],
            )

    def test_rejects_commit_mismatch_and_tracked_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, commit = self._repository(root)
            (project / "required.py").write_text(
                "def target(value=1):\n    return value\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "add callable"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            contract = self._contract(root, project, "0" * 40)
            with self.assertRaisesRegex(UpstreamContractError, "commit mismatch"):
                validate_upstream_checkout(project, contract)

            contract = self._contract(root, project, commit)
            (project / "required.py").write_text("value = 2\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    UpstreamContractError, "tracked modifications"
                ):
                    validate_upstream_checkout(project, contract)

    def test_rejects_hash_and_signature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _ = self._repository(root)
            (project / "required.py").write_text(
                "def target(value=1):\n    return value\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "add callable"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()

            bad_hash = self._contract(root, project, commit, expected_hash="0" * 64)
            with self.assertRaisesRegex(UpstreamContractError, "hash mismatch"):
                validate_upstream_checkout(project, bad_hash)

            bad_signature = self._contract(
                root, project, commit, parameters=["value", "extra"]
            )
            with self.assertRaisesRegex(UpstreamContractError, "signature mismatch"):
                validate_upstream_checkout(project, bad_signature)

    def test_overrides_require_reason_and_are_forbidden_in_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _ = self._repository(root)
            (project / "required.py").write_text(
                "def target(value=1):\n    return value\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "add callable"],
                check=True,
            )
            contract = self._contract(root, project, "0" * 40)

            with mock.patch.dict(
                "os.environ", {"ATREX_LOONGFLOW_ALLOW_UNPINNED": "1"}, clear=True
            ):
                with self.assertRaisesRegex(UpstreamContractError, "require.*REASON"):
                    validate_upstream_checkout(project, contract)

            override_environment = {
                "ATREX_LOONGFLOW_ALLOW_UNPINNED": "1",
                "ATREX_LOONGFLOW_OVERRIDE_REASON": "testing a reviewed upstream commit",
            }
            with mock.patch.dict("os.environ", override_environment, clear=True):
                result = validate_upstream_checkout(project, contract)
            self.assertTrue(result["overrides"]["active"])
            self.assertEqual(
                result["overrides"]["reason"],
                "testing a reviewed upstream commit",
            )

            with mock.patch.dict(
                "os.environ", {**override_environment, "CI": "true"}, clear=True
            ):
                with self.assertRaisesRegex(UpstreamContractError, "forbidden in CI"):
                    validate_upstream_checkout(project, contract)


if __name__ == "__main__":
    unittest.main()
