import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gcbm.eval_cub70_localization import find_cub70_images, load_concept_part_mapping


class TestCubLocalizationUtils(unittest.TestCase):
    def test_cub70_images_are_matched_by_split_class_and_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cub_root = root / "CUB_200_2011"
            cub70_root = root / "CUB70-PartSegmentationDataset"
            (cub_root / "images").mkdir(parents=True)
            (cub70_root / "AnnotationMasksPerclass" / "1").mkdir(parents=True)

            (cub_root / "images.txt").write_text(
                "\n".join(
                    [
                        "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg",
                        "2 002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg",
                    ]
                ),
                encoding="utf-8",
            )
            (cub_root / "train_test_split.txt").write_text("1 0\n2 1\n", encoding="utf-8")
            Image.new("L", (4, 4), color=255).save(
                cub70_root
                / "AnnotationMasksPerclass"
                / "1"
                / "Black_Footed_Albatross_0001_796111_beak.png"
            )

            matched = find_cub70_images(str(cub70_root), str(cub_root), split="test")

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["img_id"], 1)
        self.assertIn("beak", matched[0]["masks"])

    def test_concept_part_mapping_canonicalizes_concept_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_path = Path(tmpdir) / "mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "mappings": [
                            {"concept": "A red-bill.", "part_group": "beak", "keep": True},
                            {"concept": "unused", "part_group": "wing", "keep": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            mapping = load_concept_part_mapping(str(mapping_path), ["red bill", "black wing"])

        self.assertEqual(mapping, {0: "beak"})


if __name__ == "__main__":
    unittest.main()
