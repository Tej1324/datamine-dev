from src.reid_experiment.gallery import ExperimentalGallery


def test_partial_crop_preserves_identity_without_gallery_update(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl")
    first = gallery.observe(source=0, local_track=1, timestamp=1,
                            embedding=[1, 0], world=None)
    partial = gallery.observe(source=0, local_track=1, timestamp=2,
                              embedding=[0, 1], world=None, quality=0.2)
    assert partial["experimental_global_id"] == first["experimental_global_id"]
    assert partial["decision"] == "STICKY_LOW_QUALITY"
    assert len(gallery.identities[1].embeddings) == 1
    assert gallery.tracks[(0, 1)].last_seen == 2


def test_late_merge_requires_repeated_evidence_and_keeps_old_id(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=2,
                                  minimum_margin=0, gallery_size=1)
    first = gallery.observe(source=0, local_track=1, timestamp=1,
                            embedding=[1, 0], world=None)
    second = gallery.observe(source=1, local_track=2, timestamp=1.1,
                             embedding=[0, 1], world=None)
    pending = gallery.observe(source=1, local_track=2, timestamp=1.2,
                              embedding=[1, 0], world=None)
    assert pending["experimental_global_id"] == second["experimental_global_id"]
    assert pending["decision"] == "STICKY_PENDING_MERGE"
    merged = gallery.observe(source=1, local_track=2, timestamp=1.3,
                             embedding=[1, 0], world=None)
    assert merged["decision"] == "RETROACTIVE_MERGE"
    assert merged["experimental_global_id"] == first["experimental_global_id"]


def test_nonfinite_embedding_cannot_create_identity(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl")
    result = gallery.observe(source=0, local_track=1, timestamp=1,
                             embedding=[float("nan"), 0], world=None)
    assert result["decision"] == "REJECT_QUALITY"
    assert not gallery.identities


def test_local_track_is_sticky(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=1)
    first = gallery.observe(source=0, local_track=11, timestamp=1,
                            embedding=[1.0, 0.0], world=(0.0, 0.0))
    second = gallery.observe(source=0, local_track=11, timestamp=2,
                             embedding=[0.0, 1.0], world=(8.0, 8.0))
    assert first["experimental_global_id"] == second["experimental_global_id"]
    assert second["decision"] == "STICKY_LOCAL_TRACK"


def test_cross_camera_requires_confirmation(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=2)
    first = gallery.observe(source=0, local_track=1, timestamp=1,
                            embedding=[1.0, 0.0], world=(0.0, 0.0))
    # Cross-camera matching requires a small stable gallery, not one lucky
    # embedding from a newly-created identity.
    gallery.observe(source=0, local_track=1, timestamp=1.01,
                    embedding=[0.999, 0.01], world=(0.0, 0.0))
    gallery.observe(source=0, local_track=1, timestamp=1.02,
                    embedding=[0.998, 0.02], world=(0.0, 0.0))
    pending = gallery.observe(source=1, local_track=2, timestamp=1.1,
                              embedding=[0.99, 0.01], world=(0.1, 0.1))
    accepted = gallery.observe(source=1, local_track=2, timestamp=1.2,
                               embedding=[0.99, 0.01], world=(0.1, 0.1))
    assert pending["decision"] == "PENDING_CONFIRMATION"
    assert accepted["experimental_global_id"] == first["experimental_global_id"]
    assert accepted["decision"] == "CROSS_CAMERA_MATCH"


def test_bidirectional_match_allows_either_camera_to_birth_identity(tmp_path):
    for first_source, second_source in ((0, 1), (1, 0)):
        gallery = ExperimentalGallery(tmp_path / f"events-{first_source}.jsonl",
                                      confirmations=2, minimum_margin=0.0)
        first = gallery.observe(source=first_source, local_track=10, timestamp=10.0,
                                embedding=[1.0, 0.0], world=None)
        pending = gallery.observe(source=second_source, local_track=20, timestamp=10.1,
                                  embedding=[0.99, 0.01], world=None)
        accepted = gallery.observe(source=second_source, local_track=20, timestamp=10.2,
                                   embedding=[0.98, 0.02], world=None)
        assert first["decision"] == "NEW_ID"
        assert pending["decision"] == "PENDING_CONFIRMATION"
        assert accepted["decision"] == "CROSS_CAMERA_MATCH"
        assert accepted["experimental_global_id"] == first["experimental_global_id"]
        assert accepted["cross_camera_confirmed"] is True


def test_retroactive_merge_keeps_late_tracklet_evidence(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=1,
                                  minimum_margin=0.0)
    first = gallery.observe(source=0, local_track=1, timestamp=1.0,
                            embedding=[1.0, 0.0], world=None)
    second = gallery.observe(source=1, local_track=2, timestamp=1.1,
                             embedding=[0.0, 1.0], world=None)
    merged = None
    for index in range(3):
        merged = gallery.observe(source=1, local_track=2, timestamp=1.2 + index * 0.1,
                                 embedding=[1.0, 0.0], world=None)
    assert merged["decision"] == "RETROACTIVE_MERGE"

    # The two provisional identities collapse to one journey and the
    # confirmed local track is remapped to that winner.
    assert gallery.stats["retroactive_merges"] == 1
    assert len(gallery.identities) == 1
    winner_id = merged["experimental_global_id"]
    journey = gallery.journey_snapshot()[0]
    assert journey["global_id"] == winner_id
    assert {item["local_track_id"] for item in journey["tracklets"]} == {1, 2}


def test_journey_snapshot_contains_both_camera_tracklets(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=1,
                                  minimum_margin=0.0)
    first = gallery.observe(source=1, local_track=8, timestamp=2.0,
                            embedding=[1.0, 0.0], world=None)
    gallery.observe(source=0, local_track=17, timestamp=2.1,
                    embedding=[1.0, 0.0], world=None)
    journey = gallery.journey_snapshot()[0]
    assert journey["global_id"] == first["experimental_global_id"]
    assert journey["source_ids"] == [0, 1]
    assert journey["cross_camera_confirmed"] is True
    assert {(item["source_id"], item["local_track_id"]) for item in journey["tracklets"]} == {(0, 17), (1, 8)}


def test_same_camera_active_tracks_cannot_share_identity(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=1,
                                  minimum_margin=0.0, local_track_gap_seconds=3.0)
    first = gallery.observe(source=0, local_track=1, timestamp=1.0,
                            embedding=[1.0, 0.0], world=None)
    simultaneous = gallery.observe(source=0, local_track=2, timestamp=1.1,
                                   embedding=[1.0, 0.0], world=None)
    assert simultaneous["decision"] == "NEW_ID"
    assert simultaneous["experimental_global_id"] != first["experimental_global_id"]


def test_cross_camera_simultaneous_tracks_are_allowed(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", confirmations=1,
                                  minimum_margin=0.0, local_track_gap_seconds=3.0)
    first = gallery.observe(source=0, local_track=1, timestamp=1.0,
                            embedding=[1.0, 0.0], world=None)
    matched = gallery.observe(source=1, local_track=2, timestamp=1.1,
                              embedding=[1.0, 0.0], world=None)
    assert matched["decision"] == "CROSS_CAMERA_MATCH"
    assert matched["experimental_global_id"] == first["experimental_global_id"]


def test_ch11_anchored_mode_never_births_from_ch10(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", similarity_threshold=0.50,
                                  confirmations=1, minimum_margin=0.0,
                                  ch11_only=True, anchor_source=1)
    unmatched = gallery.observe(source=0, local_track=45, timestamp=1.0,
                                embedding=[1.0, 0.0], world=None)
    assert unmatched["experimental_global_id"] is None
    assert unmatched["decision"] == "UNMATCHED"
    assert gallery.identities == {}

    anchor = gallery.observe(source=1, local_track=101, timestamp=1.1,
                             embedding=[1.0, 0.0], world=None)
    assert anchor["decision"] == "CH11_NEW_GLOBAL_ID"
    matched = gallery.observe(source=0, local_track=45, timestamp=1.2,
                              embedding=[0.99, 0.01], world=None)
    assert matched["experimental_global_id"] == anchor["experimental_global_id"]
    assert matched["decision"] == "CROSS_CAMERA_MATCH"
    assert matched["pairwise_comparisons"] == 2


def test_ch11_anchored_gallery_keeps_all_embeddings(tmp_path):
    gallery = ExperimentalGallery(tmp_path / "events.jsonl", ch11_only=True,
                                  anchor_source=1, gallery_size=1)
    first = gallery.observe(source=1, local_track=7, timestamp=1.0,
                            embedding=[1.0, 0.0], world=None)
    for index in range(20):
        gallery.observe(source=1, local_track=7, timestamp=2.0 + index,
                        embedding=[1.0, 0.0], world=None)
    assert len(gallery.identities[first["experimental_global_id"]].embeddings) == 21
