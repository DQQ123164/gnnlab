import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.preprocessing.config import env_path, load_env_file, optional_env_path


class EnvironmentConfigTest(unittest.TestCase):
    def test_loads_and_expands_paths_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "GNNLAB_ROOT=/work/gnnlab\n"
                "GNNLAB_DSG_DIR=${GNNLAB_ROOT}/dsg\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_file)
                self.assertEqual(optional_env_path("GNNLAB_ROOT"), Path("/work/gnnlab"))
                self.assertEqual(
                    optional_env_path("GNNLAB_DSG_DIR"),
                    Path("/work/gnnlab/dsg"),
                )

    def test_shell_environment_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text("GNNLAB_ROOT=/from/file\n", encoding="utf-8")
            with patch.dict(os.environ, {"GNNLAB_ROOT": "/from/shell"}, clear=True):
                load_env_file(env_file)
                self.assertEqual(
                    env_path("GNNLAB_ROOT", Path("/default")),
                    Path("/from/shell"),
                )

    def test_missing_file_uses_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            load_env_file(Path("/missing/.env"))
            self.assertEqual(env_path("GNNLAB_ROOT", Path("/default")), Path("/default"))


if __name__ == "__main__":
    unittest.main()
