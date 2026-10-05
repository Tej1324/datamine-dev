import json
import os

import numpy as np

from deepstream.tracklet_embeddings import LocalTrackletEmbeddingStore, TrackletEmbeddingObservation
from deepstream.tracklet_representation import TrackletRepresentation


def unit(index, dimension=256):
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_normalized_prototype_and_statistics():
    result = TrackletRepresentation.build_representation([unit(0), unit(0), unit(1)])
    assert result["embedding_count"] == 3
    assert np.isclose(np.linalg.norm(result["prototype"]), 1.0)
    assert result["prototype_similarity_max"] > result["prototype_similarity_min"]
    assert result["pairwise_similarity_mean"] == 1 / 3


def test_deterministic_bounded_exemplars():
    vectors = [unit(0), unit(1), unit(2), unit(3), unit(4)]
    first = TrackletRepresentation.build_representation(vectors, exemplar_count=3, min_diversity=0.03)
    second = TrackletRepresentation.build_representation(vectors, exemplar_count=3, min_diversity=0.03)
    assert first["exemplar_count"] == 3
    assert [vector.tolist() for vector in first["exemplars"]] == [vector.tolist() for vector in second["exemplars"]]


def test_redundant_embeddings_are_not_unnecessarily_selected():
    result = TrackletRepresentation.build_representation([unit(0), unit(0), unit(0)], exemplar_count=4)
    assert result["exemplar_count"] == 1


def test_diverse_embeddings_become_exemplars():
    result = TrackletRepresentation.build_representation([unit(0), unit(1), unit(2)], exemplar_count=3)
    assert result["exemplar_count"] == 3
    assert result["pairwise_similarity_min"] == 0.0


def test_exemplar_top_k_metrics():
    result = TrackletRepresentation.build_representation([unit(0), unit(1), unit(2)], exemplar_count=3)
    assert np.isclose(result["best_exemplar_similarity"], 1.0)
    assert np.isclose(result["mean_top2_exemplar_similarity"], 0.5)
    assert np.isclose(result["mean_top3_exemplar_similarity"], 1 / 3)


def test_one_and_empty_tracklets_are_safe():
    one = TrackletRepresentation.build_representation([unit(0)])
    assert one["embedding_count"] == 1
    assert one["exemplar_count"] == 1
    assert np.isclose(one["mean_top2_exemplar_similarity"], 1.0)
    empty = TrackletRepresentation.build_representation([])
    assert empty["embedding_count"] == 0
    assert empty["prototype"] is None


def test_invalid_vectors_are_ignored():
    result = TrackletRepresentation.build_representation([np.ones(3), np.full(256, np.nan), unit(0)])
    assert result["embedding_count"] == 1


def test_diagnostic_disabled_does_not_create_representation_output(monkeypatch, tmp_path):
    output = tmp_path / "representations.jsonl"
    monkeypatch.delenv("TRACKLET_REPRESENTATION_DIAGNOSTIC", raising=False)
    monkeypatch.setenv("TRACKLET_EMBED_OUTPUT", str(tmp_path / "embeddings.jsonl"))
    monkeypatch.setenv("TRACKLET_REPRESENTATION_OUTPUT", str(output))
    assert os.getenv("TRACKLET_REPRESENTATION_DIAGNOSTIC", "0") == "0"
    store = LocalTrackletEmbeddingStore()
    store.observe(TrackletEmbeddingObservation("CH10", 1, 1, 1.0, (0, 0, 10, 20), .9, .8, unit(0)))
    store.cleanup(100.0)
    store.close()
    assert not output.exists()


def test_enabled_output_contains_summary_without_vectors(monkeypatch, tmp_path):
    output = tmp_path / "representations.jsonl"
    monkeypatch.setenv("TRACKLET_REPRESENTATION_OUTPUT", str(output))
    monkeypatch.setenv("TRACKLET_EMBED_OUTPUT", str(tmp_path / "embeddings.jsonl"))
    monkeypatch.setenv("TRACKLET_REPRESENTATION_INCLUDE_VECTORS", "0")
    representation = TrackletRepresentation()
    store = LocalTrackletEmbeddingStore(representation)
    store.observe(TrackletEmbeddingObservation("CH10", 1, 1, 1.0, (0, 0, 10, 20), .9, .8, unit(0)))
    store.cleanup(100.0)
    store.close()
    events = [json.loads(line) for line in output.read_text().splitlines()]
    summary = next(event for event in events if event["type"] == "REPRESENTATION_SUMMARY")
    assert summary["source_id"] == "CH10"
    assert "prototype" not in summary
    assert "exemplars" not in summary
