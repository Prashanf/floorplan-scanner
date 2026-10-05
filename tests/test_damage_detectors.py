"""Damage detector selection (heuristic default, MobileSAM + OWL), the model path with fake models, and the comparison."""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from src.damage import compare
from src.damage import detection as det
from src.damage import model_detection as md
from src.damage.detection import DamageDetection, detect_damage


def _det(box, cls="crack", image="a.jpg", conf=0.5):
    return DamageDetection(image_path=image, bbox=box, damage_class=cls, confidence=conf, pixel_area=100,
                           image_size=(1000, 800))


def _stain_image(path):
    img = np.full((960, 1280, 3), 235, np.uint8)
    img[100:300, 100:400] = (110, 150, 185)  # light brown: a water stain for the OpenCV rules
    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------- selection

def test_heuristic_is_the_default_and_unchanged(tmp_path):
    path = _stain_image(tmp_path / "wall.png")
    default, explicit = detect_damage([path]), detect_damage([path], detector="heuristic")
    assert default and [(d.bbox, d.damage_class, d.confidence) for d in default] == \
        [(d.bbox, d.damage_class, d.confidence) for d in explicit]
    assert any(d.damage_class == "water_stain" for d in default)


def test_opencv_is_the_cli_default_and_an_alias_of_the_heuristic(tmp_path):
    import run
    option = next(p for p in run.main.params if p.name == "damage_detector")
    assert option.default == "opencv" and "--damage-model" in option.opts
    path = _stain_image(tmp_path / "wall.png")
    assert [(d.bbox, d.damage_class) for d in detect_damage([path], detector="opencv")] == \
        [(d.bbox, d.damage_class) for d in detect_damage([path])]


def test_unknown_detector_and_stray_options_are_errors(tmp_path):
    path = _stain_image(tmp_path / "wall.png")
    with pytest.raises(ValueError, match="unknown damage detector"):
        detect_damage([path], detector="yolo")
    with pytest.raises(TypeError):
        detect_damage([path], detector="heuristic", confidence_threshold=0.5)


@pytest.mark.parametrize("name", ["mobilesam", "model", "nanoowl", "MobileSAM"])
def test_model_names_route_to_the_model_detector(monkeypatch, name):
    seen = {}
    monkeypatch.setattr(md, "detect_damage_model", lambda images, persistent_frames=0, **kw: seen.update(kw) or [])
    assert detect_damage(["x.jpg"], detector=name, confidence_threshold=0.4) == []
    assert seen == {"confidence_threshold": 0.4}


# ---------------------------------------------------------------- model path with fake models

class _FakeSam:
    def __init__(self, area_side=30):
        self.side, self.image = area_side, None

    def set_image(self, image):
        self.image = image

    def predict(self, box, multimask_output):
        mask = np.zeros(self.image.shape[:2], bool)
        x1, y1 = int(box[0]), int(box[1])
        mask[y1:y1 + self.side, x1:x1 + 4 * self.side] = True  # a wide, flat blob: reads as a horizontal crack
        return np.array([mask]), np.array([0.9]), None


def _model_run(monkeypatch, tmp_path, boxes, sam=True):
    path = str(tmp_path / "wall.jpg")
    cv2.imwrite(path, np.full((600, 800, 3), 200, np.uint8))
    monkeypatch.setattr(md, "_load_owl", lambda name, device: (None, None))
    monkeypatch.setattr(md, "_load_sam", lambda weights, device: _FakeSam() if sam else None)
    monkeypatch.setattr(md, "_owl_boxes", lambda *a, **k: boxes)
    return path, md.detect_damage_model([path, str(tmp_path / "missing.jpg")], device="cpu")


def test_model_detections_use_mask_area_and_class_of_the_prompt(monkeypatch, tmp_path):
    prompt = md.DEFAULT_PROMPTS["crack"][0]
    path, found = _model_run(monkeypatch, tmp_path, [(100, 100, 300, 160, 0.7, prompt)])
    assert len(found) == 1
    d = found[0]
    assert d.damage_class == "crack" and d.image_path == path and d.image_size == (800, 600)
    assert d.pixel_area == 30 * 120  # the mask, not the 200 x 60 box
    assert d.thickness_px == pytest.approx(30, abs=1) and d.angle_deg in (pytest.approx(0, abs=2), pytest.approx(180, abs=2))
    assert d.confidence == pytest.approx(0.7)


def test_without_sam_the_box_area_is_used(monkeypatch, tmp_path):
    prompt = md.DEFAULT_PROMPTS["water_stain"][0]
    _, found = _model_run(monkeypatch, tmp_path, [(100, 100, 300, 160, 0.6, prompt)], sam=False)
    assert found[0].damage_class == "water_stain" and found[0].pixel_area == 200 * 60


def test_model_boxes_covering_the_frame_are_dropped_and_overlaps_merged(monkeypatch, tmp_path):
    crack = md.DEFAULT_PROMPTS["crack"]
    boxes = [(0, 0, 800, 400, 0.9, crack[0]),                 # 67% of the frame: a wall, not damage
             (100, 100, 300, 160, 0.7, crack[0]),             # kept
             (105, 102, 305, 162, 0.5, crack[1])]             # same crack, other prompt: merged
    _, found = _model_run(monkeypatch, tmp_path, boxes, sam=False)
    assert len(found) == 1 and found[0].bbox[0] <= 105


def test_at_most_five_model_detections_per_image(monkeypatch, tmp_path):
    prompt = md.DEFAULT_PROMPTS["mold"][0]
    boxes = [(10 + 120 * i, 10, 60 + 120 * i, 50, 0.5 + i / 100, prompt) for i in range(7)]
    _, found = _model_run(monkeypatch, tmp_path, boxes, sam=False)
    assert len(found) == det.MAX_PER_IMAGE
    assert [d.confidence for d in found] == sorted((d.confidence for d in found), reverse=True)


def test_missing_dependencies_give_a_clear_error(monkeypatch):
    monkeypatch.setattr(md, "_CACHE", {})
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)  # makes the import fail
    with pytest.raises(ImportError, match="requirements-damage-model.txt"):
        md._load_owl("some/model", "cpu")


# ---------------------------------------------------------------- comparison

def test_match_boxes_is_greedy_one_to_one_and_class_aware():
    a = [_det((0, 0, 100, 100), "crack"), _det((200, 200, 300, 300), "mold")]
    b = [_det((5, 5, 105, 105), "crack"), _det((205, 205, 305, 305), "water_stain")]
    assert [(i, j) for i, j, _ in compare.match_boxes(a, b)] == [(0, 0)]
    assert sorted((i, j) for i, j, _ in compare.match_boxes(a, b, same_class=False)) == [(0, 0), (1, 1)]
    assert compare.match_boxes(a, [_det((500, 500, 600, 600))]) == []


def test_agreement_counts_per_image():
    first = [_det((0, 0, 100, 100), "crack", "a.jpg"), _det((0, 0, 50, 50), "mold", "b.jpg"),
             _det((300, 300, 400, 400), "hole", "a.jpg")]
    second = [_det((3, 3, 103, 103), "crack", "a.jpg"), _det((0, 0, 50, 50), "water_stain", "b.jpg"),
              _det((0, 0, 100, 100), "crack", "c.jpg")]
    assert compare.agreement(first, second) == {"same_class": 1, "same_place_other_class": 1,
                                                "only_first": 1, "only_second": 1}


def test_scores_against_labels():
    labels = {"a.jpg": [{"class": "crack", "bbox": [0, 0, 100, 100]}, {"class": "mold", "bbox": [400, 400, 500, 500]}],
              "clean.jpg": []}
    found = [_det((2, 2, 102, 102), "crack", "/x/a.jpg"),      # right (matched by file name)
             _det((600, 0, 700, 100), "crack", "a.jpg"),       # wrong place
             _det((0, 0, 50, 50), "hole", "clean.jpg"),        # false alarm on a clean image
             _det((0, 0, 50, 50), "hole", "unlabelled.jpg")]   # not scored
    score = compare.score_against_labels(found, labels)
    assert score["overall"]["tp"] == 1 and score["overall"]["fp"] == 2 and score["overall"]["fn"] == 1
    assert score["overall"]["precision"] == pytest.approx(1 / 3) and score["overall"]["recall"] == pytest.approx(1 / 2)
    assert score["per_class"]["mold"]["fn"] == 1 and score["per_class"]["hole"]["fp"] == 1


def test_compare_detectors_runs_both_and_scores(monkeypatch):
    outputs = {"heuristic": [_det((0, 0, 100, 100), "crack", "a.jpg")],
               "mobilesam": [_det((0, 0, 100, 100), "crack", "a.jpg"), _det((300, 0, 400, 100), "mold", "a.jpg")]}
    calls = []

    def fake(images, detector, **kw):
        calls.append((detector, kw))
        return outputs[detector]

    result = compare.compare_detectors(["a.jpg"], {"a.jpg": [{"class": "crack", "bbox": [0, 0, 100, 100]}]},
                                       run=fake, confidence_threshold=0.2)
    assert calls == [("heuristic", {}), ("mobilesam", {"confidence_threshold": 0.2})]  # options go to the model only
    assert result["summary"]["heuristic"]["detections"] == 1 and result["summary"]["mobilesam"]["per_class"] == {"crack": 1, "mold": 1}
    assert result["agreement"]["same_class"] == 1 and result["agreement"]["only_second"] == 1
    assert result["scores"]["heuristic"]["overall"]["precision"] == 1.0
    assert result["scores"]["mobilesam"]["overall"]["fp"] == 1


# ---------------------------------------------------------------- CLI

def test_cli_rejects_model_options_with_the_heuristic_detector(tmp_path):
    import run
    result = CliRunner().invoke(run.main, [str(tmp_path), "--tier", "photo", "--damage-threshold", "0.2"])
    assert result.exit_code != 0 and "--damage-model mobilesam" in result.output
