import numpy as np

from deepstream.global_identity import GlobalIdentityManager, TrackObservation


def observation(source, local_id, timestamp, vector, det=.9, tracker=.9, height=120):
    return TrackObservation(
        source, local_id, timestamp, float(timestamp), (0, 0, 40, height),
        det, tracker, np.asarray(vector, dtype=np.float32), None,
    )


def manager(monkeypatch, tmp_path, identity="CH10", observer="CH11", gap="10"):
    monkeypatch.setenv("GLOBAL_REID_OUTPUT", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("GLOBAL_REID_MATCH_THRESHOLD", "0.8")
    monkeypatch.setenv("GLOBAL_REID_MIN_MARGIN", "0.05")
    monkeypatch.setenv("GLOBAL_REID_MAX_GAP_SECONDS", gap)
    monkeypatch.setenv("GLOBAL_REID_MIN_GOOD_OBSERVATIONS", "3")
    monkeypatch.setenv("GLOBAL_REID_IDENTITY_CAMERAS", identity)
    monkeypatch.setenv("GLOBAL_REID_OBSERVER_CAMERAS", observer)
    return GlobalIdentityManager()


def three_observations(manager, source, local_id, vector, start=1):
    return [
        manager.observe(observation(source, local_id, start + offset, vector))
        for offset in range(3)
    ]


def test_global_id_waits_for_three_good_embeddings(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        results = three_observations(m, "CH10", 17, vector)
        assert results[0]["assignment_reason"] == "PENDING_GOOD_EMBEDDINGS"
        assert results[1]["assignment_reason"] == "PENDING_GOOD_EMBEDDINGS"
        assert results[2]["assignment_reason"] == "NEW_GLOBAL_PERSON_AFTER_CONFIRMATION"
        assert results[2]["assigned_global_id"] is not None
    finally:
        m.close()


def test_same_local_track_keeps_global_id_and_updates_gallery(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        first = three_observations(m, "CH10", 17, vector)[-1]
        second = m.observe(observation("CH10", 17, 4, vector))
        assert second["assigned_global_id"] == first["assigned_global_id"]
        assert second["assignment_reason"] in {"SAME_LOCAL_TRACK", "SAME_LOCAL_TRACK_GALLERY_UPDATED"}
    finally:
        m.close()


def test_observer_camera_recovers_existing_id(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        first = three_observations(m, "CH10", 17, vector)[-1]
        results = three_observations(m, "CH11", 4, vector, start=4)
        assert results[-1]["assigned_global_id"] == first["assigned_global_id"]
        assert results[-1]["assignment_reason"] == "CROSS_CAMERA_REID_MATCH"
    finally:
        m.close()


def test_observer_camera_never_creates_global_id(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        results = three_observations(m, "CH11", 4, vector)
        assert all(result["assigned_global_id"] is None for result in results)
        assert results[-1]["assignment_reason"] == "UNASSIGNED_NO_MATCH"
        assert m.summary()["new_global_persons"] == 0
    finally:
        m.close()


def test_ambiguous_candidate_remains_unassigned(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        first_vector = np.zeros(256, dtype=np.float32)
        first_vector[0] = 1
        second_vector = np.zeros(256, dtype=np.float32)
        second_vector[0] = 1
        second_vector[1] = .1
        candidate = np.zeros(256, dtype=np.float32)
        candidate[0] = 1
        three_observations(m, "CH10", 1, first_vector)
        three_observations(m, "CH10", 2, second_vector, start=4)
        results = three_observations(m, "CH11", 7, candidate, start=7)
        assert results[-1]["assigned_global_id"] is None
        assert results[-1]["assignment_reason"] == "UNASSIGNED_LOW_MARGIN"
    finally:
        m.close()


def test_same_camera_active_track_is_rejected(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        first = three_observations(m, "CH10", 1, vector)[-1]
        result = three_observations(m, "CH10", 2, vector, start=4)[-1]
        assert result["assigned_global_id"] != first["assigned_global_id"]
        assert m.summary()["same_camera_active_track_rejections"] >= 1
    finally:
        m.close()


def test_expiry_and_invalid_embedding(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path, gap="10")
    try:
        vector = np.ones(256, dtype=np.float32)
        first = three_observations(m, "CH10", 1, vector)[-1]
        expired = three_observations(m, "CH11", 2, vector, start=20)[-1]
        assert expired["assigned_global_id"] is None
        assert expired["assignment_reason"] == "UNASSIGNED_NO_MATCH"
        assert first["assigned_global_id"] is not None
        invalid = m.observe(observation("CH10", 3, 21, np.zeros(256, dtype=np.float32)))
        assert invalid["assigned_global_id"] is None
        assert invalid["assignment_reason"] == "INVALID_EMBEDDING"
    finally:
        m.close()


def test_low_quality_observation_does_not_enter_gallery(monkeypatch, tmp_path):
    m = manager(monkeypatch, tmp_path)
    try:
        vector = np.ones(256, dtype=np.float32)
        result = m.observe(observation("CH10", 1, 1, vector, det=.1))
        assert result["assigned_global_id"] is None
        assert result["assignment_reason"] == "LOW_DETECTOR_CONFIDENCE"
        assert m.summary()["gallery_vectors"] == 0
    finally:
        m.close()
