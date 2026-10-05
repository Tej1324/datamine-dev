# SPDX-FileCopyrightText: 2026 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from tools.evaluate_ch10_ch11 import evaluate, load_annotations


def row(source, local, global_id, timestamp=100, frame=1):
    return {
        "type": "world_identity_assignment",
        "source_id": source,
        "local_object_id": local,
        "global_id": global_id,
        "timestamp": timestamp,
        "frame": frame,
    }


def test_global_pairs_report_misses_and_false_merges():
    annotations = [
        {"reference_id": "p1", "same_person": True, "source_a": 0, "id_a": "G7", "source_b": 1, "id_b": "G8"},
        {"reference_id": "p2", "same_person": True, "source_a": 0, "id_a": "G9", "source_b": 1, "id_b": "G9"},
        {"reference_id": "n1", "same_person": False, "source_a": 0, "id_a": "G7", "source_b": 1, "id_b": "G7"},
    ]
    report = evaluate([], annotations)
    assert report["counts"]["missed_match"] == 1
    assert report["counts"]["positive"] == 1
    assert report["counts"]["false_merge"] == 1
    assert report["positive_recall"] == 0.5


def test_local_ids_resolve_to_global_ids():
    rows = [row(0, 11, 41), row(1, 22, 41)]
    annotations = [{
        "reference_id": "p1", "same_person": True,
        "source_a": 0, "id_a": 11, "source_b": 1, "id_b": 22,
        "id_type": "local", "timestamp_ns": 100,
    }]
    report = evaluate(rows, annotations)
    assert report["counts"]["positive"] == 1
    assert report["results"][0]["resolved_global_a"] == 41
    assert report["results"][0]["resolved_global_b"] == 41


def test_simple_csv_annotations(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "second,reference_id,left_id,right_id,same\n"
        "0,person_001,G7,G8,yes\n"
        "1,person_001,G9,G9,yes\n"
        "2,negative_001,G7,G7,no\n",
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    report = evaluate([], annotations)
    assert len(annotations) == 3
    assert annotations[0]["sample_second"] == 0.0
    assert report["counts"]["positive"] == 1
    assert report["counts"]["missed_match"] == 1
    assert report["counts"]["false_merge"] == 1
