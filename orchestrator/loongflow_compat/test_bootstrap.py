from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.loongflow_compat import bootstrap, compat_adapter


class TestExplicitBootstrap(unittest.TestCase):
    def test_adapter_import_has_no_patch_side_effect(self) -> None:
        self.assertEqual(compat_adapter.PATCH_MANIFEST, {})

    def test_install_applies_then_validates(self) -> None:
        expected = {"database": {"required": True, "applied": True}}
        with (
            mock.patch.object(compat_adapter, "apply_compat_patches") as apply,
            mock.patch.object(
                compat_adapter,
                "validate_patch_manifest",
                return_value=expected,
            ) as validate,
        ):
            actual = bootstrap.install_and_validate()

        apply.assert_called_once_with()
        validate.assert_called_once_with()
        self.assertEqual(actual, expected)

    def test_run_entrypoint_uses_same_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entrypoint = Path(tmp) / "entry.py"
            entrypoint.write_text("RESULT = 'ran'\n", encoding="utf-8")
            with mock.patch.object(bootstrap.runpy, "run_path") as run_path:
                bootstrap.run_entrypoint(entrypoint, ["--flag", "value"])

        run_path.assert_called_once_with(str(entrypoint.resolve()), run_name="__main__")
        self.assertEqual(
            bootstrap.sys.argv,
            [str(entrypoint.resolve()), "--flag", "value"],
        )


if __name__ == "__main__":
    unittest.main()
