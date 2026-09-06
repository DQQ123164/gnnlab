import tempfile
import unittest
from pathlib import Path

from tools.preprocessing.dsg import (
    DSG_ACTION_LABELS,
    build_official_protocols,
    discover_materialized_protocols,
    parse_dsg_filename,
)


class DsgProtocolTest(unittest.TestCase):
    def test_original_five_class_label_order(self) -> None:
        self.assertEqual(
            DSG_ACTION_LABELS,
            {"A059": 0, "A030": 1, "A016": 2, "A005": 3, "A027": 4},
        )

    def test_official_protocol_assignment(self) -> None:
        training_subject_camera_one = parse_dsg_filename(
            Path("S001C001P001R001A059.skeleton")
        )
        test_subject_camera_two = parse_dsg_filename(
            Path("S001C002P003R001A030.skeleton")
        )
        self.assertIsNotNone(training_subject_camera_one)
        self.assertIsNotNone(test_subject_camera_two)

        protocols = build_official_protocols(
            [training_subject_camera_one, test_subject_camera_two]  # type: ignore[list-item]
        )
        self.assertEqual(protocols["xsub"]["train"], [training_subject_camera_one])
        self.assertEqual(protocols["xsub"]["test"], [test_subject_camera_two])
        self.assertEqual(protocols["xview"]["train"], [test_subject_camera_two])
        self.assertEqual(protocols["xview"]["test"], [training_subject_camera_one])

    def test_wrong_materialized_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for protocol in ("xsub", "xview"):
                for split in ("train", "test"):
                    directory = root / protocol / split
                    directory.mkdir(parents=True)
                    (directory / "S001C001P001R001A033.skeleton").touch()
            with self.assertRaisesRegex(ValueError, "Unexpected DSG action"):
                discover_materialized_protocols(root)


if __name__ == "__main__":
    unittest.main()
