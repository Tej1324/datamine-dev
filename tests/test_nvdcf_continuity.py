import json
from types import SimpleNamespace

from deepstream.nvdcf_continuity import NvdcfContinuityDiagnostic


def obj(local_id, left=100, top=100, confidence=.9, tracker=.9, cls=0):
    return SimpleNamespace(
        class_id=cls,
        object_id=local_id,
        confidence=confidence,
        tracker_confidence=tracker,
        rect_params=SimpleNamespace(left=left, top=top, width=50, height=100),
    )


def frame(number, *objects, timestamp=None, source="2"):
    return SimpleNamespace(
        source_id=source,
        frame_number=number,
        ntp_timestamp=number if timestamp is None else timestamp,
        buffer_pts=0,
        object_items=list(objects),
    )


def batch(*frames):
    return SimpleNamespace(frame_items=list(frames), n_frames=len(frames))


def diagnostic(monkeypatch, tmp_path, **values):
    monkeypatch.setenv("NVCDF_CONTINUITY_OUTPUT", str(tmp_path / "continuity.jsonl"))
    monkeypatch.setenv("NVCDF_CONTINUITY_STATE_TTL_SECONDS", "10")
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    return NvdcfContinuityDiagnostic()


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_consecutive_same_id_has_no_switch(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj(7)), frame(2, obj(7))))
    d.close()
    assert not [e for e in events(tmp_path / "continuity.jsonl") if "SWITCH" in e["type"]]


def test_high_iou_one_frame_gap_is_continuous_switch_candidate(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj(7)), frame(2, obj(8, left=102))))
    d.close()
    switch = next(e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "LOCAL_ID_CONTINUITY_BREAK")
    assert switch["frame_gap"] == 1
    assert switch["bbox_iou"] > .3
    assert switch["classification"] == "DETECTOR_CONTINUOUS_TRACKER_SWITCH"
    assert switch["previous_detector_confidence"] == .9
    assert switch["new_detector_confidence"] == .9
    assert switch["previous_bbox"]["width"] == 50.0
    assert switch["new_bbox"]["height"] == 100.0


def test_detector_gap_classification(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj(7)), frame(3, obj(8))))
    d.close()
    switch = next(e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "LOCAL_ID_CONTINUITY_BREAK")
    assert switch["classification"] == "DETECTOR_GAP_NEW_TRACK"


def test_far_objects_do_not_create_continuity_break(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj(7, tracker=.2)), frame(2, obj(8, left=1000, tracker=.2))))
    d.close()
    assert not [e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "LOCAL_ID_CONTINUITY_BREAK"]


def test_small_center_distance_is_continuity_break(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path, NVCDF_CONTINUITY_IOU_THRESHOLD=.99,
                   NVCDF_CONTINUITY_CENTER_DISTANCE_THRESHOLD=20)
    d.handle_metadata(batch(frame(1, obj(7, tracker=.2)), frame(2, obj(8, left=115, tracker=.2))))
    d.close()
    switch = next(e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "LOCAL_ID_CONTINUITY_BREAK")
    assert switch["classification"] == "LOW_TRACKER_CONFIDENCE_TRANSITION"
    assert "DETECTOR_CONTINUOUS_TRACKER_SWITCH" in switch["classifications"]


def test_crossing_uses_one_to_one_best_spatial_successor(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(
        frame(1, obj(7, left=100), obj(8, left=300)),
        frame(2, obj(17, left=102), obj(18, left=302)),
    ))
    d.close()
    breaks = [e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "LOCAL_ID_CONTINUITY_BREAK"]
    assert len(breaks) == 2
    assert {(e["previous_local_object_id"], e["new_local_object_id"]) for e in breaks} == {(7, 17), (8, 18)}


def test_track_summary_and_ttl_cleanup(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj(7)), frame(2, obj(7))))
    assert d.cleanup(12) == 1
    d.close()
    summary = next(e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "TRACK_SUMMARY")
    assert summary["duration_frames"] == 2
    assert summary["observation_count"] == 2


def test_malformed_metadata_does_not_crash(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    malformed = SimpleNamespace(class_id=0, object_id=1, rect_params=None)
    d.handle_metadata(batch(frame(1, malformed), frame(2, obj(2, cls=1))))
    d.close()
    assert (tmp_path / "continuity.jsonl").exists()


def test_untracked_uint64_sentinel_is_ignored(monkeypatch, tmp_path):
    d = diagnostic(monkeypatch, tmp_path)
    d.handle_metadata(batch(frame(1, obj((1 << 64) - 1))))
    d.close()
    assert not [e for e in events(tmp_path / "continuity.jsonl") if e["type"] == "TRACK_OBSERVATION"]
