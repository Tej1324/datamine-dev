import json

import numpy as np

from deepstream.tracklet_embeddings import LocalTrackletEmbeddingStore, TrackletEmbeddingObservation


def obs(source="CH10", local_id=1, timestamp=1.0, vector=None):
    if vector is None:
        vector = np.arange(1, 257, dtype=np.float32)
    return TrackletEmbeddingObservation(source, local_id, int(timestamp), timestamp, (1, 2, 30, 80), .9, .8, vector)


def store(monkeypatch, tmp_path, **values):
    monkeypatch.setenv("TRACKLET_EMBED_OUTPUT", str(tmp_path / "tracklets.jsonl"))
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    return LocalTrackletEmbeddingStore()


def test_accepts_and_normalizes(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path)
    s.observe(obs())
    assert len(s._tracks[("CH10", 1)].embeddings) == 1
    assert np.isclose(np.linalg.norm(s._tracks[("CH10", 1)].embeddings[0]), 1.0)
    s.close()


def test_invalid_dimension_nan_and_inf_rejected(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path, TRACKLET_EMBED_SAMPLE_EVERY=1, TRACKLET_EMBED_MIN_GAP_MS=0)
    for vector in (np.ones(3), np.full(256, np.nan), np.full(256, np.inf)):
        s.observe(obs(timestamp=1.0 + len(s._tracks), vector=vector))
    assert s._tracks[("CH10", 1)].rejected_embedding_count == 3
    s.close()


def test_sampling_and_minimum_gap(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path, TRACKLET_EMBED_SAMPLE_EVERY=2, TRACKLET_EMBED_MIN_GAP_MS=250)
    s.observe(obs(timestamp=1.0))
    s.observe(obs(timestamp=1.1))
    s.observe(obs(timestamp=1.2))
    assert len(s._tracks[("CH10", 1)].embeddings) == 1
    assert s._tracks[("CH10", 1)].observation_count == 3
    s.close()


def test_redundant_and_maximum(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path, TRACKLET_EMBED_SAMPLE_EVERY=1, TRACKLET_EMBED_MIN_GAP_MS=0, TRACKLET_EMBED_MAX_PER_TRACK=2)
    s.observe(obs(timestamp=1.0))
    s.observe(obs(timestamp=2.0))
    different = np.zeros(256, dtype=np.float32); different[0] = 1
    s.observe(obs(timestamp=3.0, vector=different))
    s.observe(obs(timestamp=4.0, vector=different))
    track = s._tracks[("CH10", 1)]
    assert len(track.embeddings) == 2
    assert track.rejected_embedding_count >= 2
    s.close()


def test_summary_prototype_and_ttl(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path, TRACKLET_EMBED_SAMPLE_EVERY=1, TRACKLET_EMBED_MIN_GAP_MS=0, TRACKLET_MEMORY_TTL_SECONDS=2)
    s.observe(obs(timestamp=1.0))
    s.observe(obs(timestamp=2.0, vector=np.arange(256, 0, -1, dtype=np.float32)))
    assert s.cleanup(4.1) == 1
    summary = s.summaries()[0]
    assert summary["stored_embedding_count"] == 2
    assert summary["mean_prototype_similarity"] is not None
    s.cleanup(7.0)
    assert not s.summaries()
    s.close()


def test_track_keys_are_independent(monkeypatch, tmp_path):
    s = store(monkeypatch, tmp_path, TRACKLET_EMBED_SAMPLE_EVERY=1, TRACKLET_EMBED_MIN_GAP_MS=0)
    s.observe(obs("CH10", 1, 1.0))
    s.observe(obs("CH10", 2, 1.0))
    s.observe(obs("CH11", 1, 1.0))
    assert len(s._tracks) == 3
    s.close()


def test_jsonl_is_async_and_no_vector_by_default(monkeypatch, tmp_path):
    path = tmp_path / "tracklets.jsonl"
    monkeypatch.setenv("TRACKLET_EMBED_OUTPUT", str(path))
    s = LocalTrackletEmbeddingStore()
    s.observe(obs())
    s.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    accepted = next(event for event in events if event["type"] == "EMBED_ACCEPTED")
    assert "embedding" not in accepted
