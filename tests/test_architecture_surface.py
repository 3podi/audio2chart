import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# SHA-1 over the file with line endings normalised to LF. Hashing the raw bytes made this
# test fail on every Windows checkout -- Git's core.autocrlf=true writes CRLF, so the digest
# never matched and the pins could not be told apart from a real edit.
FROZEN_HASHES = {
    "inference/engine.py": "2df4766615f13eb0056bce211ce3c25dbf440735",
    "inference/model_inference.py": "cd526cfbb27acd9921fb51a3c240f489bdad80c6",
    "inference/layers.py": "e44bfddff391fe20f482c7152eef11abd9dc88bd",
    "generate.py": "f9983ce2cf245dff81af10e613cdd0446775b637",
    "notebooks/audio2chart_charting.ipynb": "784fe5ffbbc2aa8efdbf20104ae09591e5ab6bf0",
}


def frozen_digest(path):
    return hashlib.sha1(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def class_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


class ArchitectureSurfaceTests(unittest.TestCase):
    def test_only_supported_lightning_modules_remain(self):
        self.assertEqual(
            class_names(ROOT / "modules" / "trainer.py"),
            {"NotesTransformer", "WaveformTransformerDiscrete"},
        )

    def test_only_supported_training_models_remain(self):
        self.assertEqual(
            class_names(ROOT / "modules" / "models.py"),
            {"Encodec", "ResnetBlock2d", "SEANetEncoder2d",
             "TransformerEncoder", "TransformerDecoderOnly"},
        )

    def test_inference_files_are_still_tracked_and_present(self):
        self.assertTrue((ROOT / "inference" / "engine.py").is_file())
        self.assertTrue((ROOT / "inference" / "model_inference.py").is_file())
        self.assertTrue((ROOT / "inference" / "layers.py").is_file())

    def test_inference_model_is_self_contained(self):
        contents = (ROOT / "inference" / "model_inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from inference.layers import", contents)
        self.assertNotIn("from modules.", contents)
        self.assertFalse((ROOT / "modules" / "transformer2.py").exists())

    def test_frozen_inference_surface_is_byte_identical(self):
        # Report every drifted file, not just the first: asserting inside the loop meant one
        # stale pin hid whether the other four had changed at all.
        drifted = {
            relative_path: frozen_digest(ROOT / relative_path)
            for relative_path, expected in FROZEN_HASHES.items()
            if frozen_digest(ROOT / relative_path) != expected
        }
        self.assertEqual({}, drifted, f"frozen inference surface changed: {drifted}")

    def test_training_configs_use_separate_transformer(self):
        for name in ("audio_codec.yaml", "audio_discrete.yaml"):
            contents = (ROOT / "configs" / "model" / name).read_text(encoding="utf-8")
            self.assertIn("modules.training_transformer.TransformerDecoderAudioConditioned", contents)


if __name__ == "__main__":
    unittest.main()
