#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <nvds_tracker_meta.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef PACKAGE
#define PACKAGE "datamine-cv"
#endif

namespace {

struct Key {
  guint source = 0;
  guint64 local_id = 0;
  bool operator==(const Key &other) const { return source == other.source && local_id == other.local_id; }
};

struct KeyHash {
  std::size_t operator()(const Key &key) const {
    return (static_cast<std::size_t>(key.source) << 32) ^ static_cast<std::size_t>(key.local_id);
  }
};

struct PairKey {
  Key first;
  Key second;
  bool operator==(const PairKey &other) const { return first == other.first && second == other.second; }
};

struct PairKeyHash {
  std::size_t operator()(const PairKey &key) const {
    const std::size_t first_hash = KeyHash{}(key.first);
    const std::size_t second_hash = KeyHash{}(key.second);
    return first_hash ^ (second_hash + static_cast<std::size_t>(0x9e3779b9) +
                         (first_hash << 6) + (first_hash >> 2));
  }
};

static PairKey canonical_pair_key(const Key &left, const Key &right) {
  const bool left_first = left.source < right.source ||
                          (left.source == right.source && left.local_id <= right.local_id);
  return left_first ? PairKey{left, right} : PairKey{right, left};
}

struct Observation {
  NvDsObjectMeta *object = nullptr;
  Key key;
  guint64 frame = 0;
  guint64 timestamp = 0;
  double seconds = 0.0;
  double x = 0.0;
  double y = 0.0;
  std::vector<float> embedding;
  bool fresh_embedding = false;
  std::size_t verification_samples = 0;
  std::size_t good_embedding_samples = 0;
  double quality_score = 0.0;
  int quality_level = 0;
  double bbox_width = 0.0;
  double bbox_height = 0.0;
  double bbox_area = 0.0;
  double detector_confidence = -1.0;
  double crop_completeness = 0.0;
  double blur_score = -1.0;
};

struct Track {
  int global_id = 0;
  double last_seen = 0.0;
  double x = 0.0;
  double y = 0.0;
  guint source = 0;
  std::vector<float> embedding;
  std::deque<std::vector<float>> gallery;
  std::size_t verified_samples = 0;
  std::size_t good_embedding_samples = 0;
  double first_seen = 0.0;
  std::size_t observed_samples = 0;
};

struct GlobalIdentity {
  int global_id = 0;
  Key last_key;
  guint source = 0;
  double last_seen = 0.0;
  double x = 0.0;
  double y = 0.0;
  std::vector<float> embedding;
  std::deque<std::vector<float>> gallery;
  std::size_t good_embedding_samples = 0;
};

struct PendingMatch {
  int global_id = 0;
  std::size_t confirmations = 0;
  double last_seen = 0.0;
};

struct PendingPair {
  int global_id = 0;
  std::size_t confirmations = 0;
  double last_seen = 0.0;
};

// A merge is deliberately more demanding than assigning a new local track:
// both IDs are already visible to an operator, so a false merge is much more
// costly than delaying a correct merge for a few frames.
struct PendingIdentityMerge {
  std::size_t confirmations = 0;
  double last_seen = 0.0;
  double appearance_sum = 0.0;
  double score_sum = 0.0;
};

static bool normalize(std::vector<float> &value);

static void update_embedding(std::vector<float> &stored, const std::vector<float> &current) {
  if (current.empty()) return;
  if (stored.size() != current.size()) {
    stored = current;
    return;
  }
  // Keep a stable gallery vector instead of replacing it with a single
  // viewpoint. This is important when the two cameras see different sides of
  // the same person and the tracker emits Re-ID metadata intermittently.
  constexpr double kNewSampleWeight = 0.25;
  for (std::size_t index = 0; index < stored.size(); ++index) {
    stored[index] = static_cast<float>((1.0 - kNewSampleWeight) * stored[index] +
                                       kNewSampleWeight * current[index]);
  }
  normalize(stored);
}

struct Candidate {
  std::size_t observation = 0;
  Key track_key;
  int global_id = 0;
  double distance = 0.0;
  double time_gap = 0.0;
  double appearance = -1.0;
  double score = 0.0;
  double spatial_score = 0.0;
  double temporal_score = 0.0;
  std::size_t paired_observation = std::numeric_limits<std::size_t>::max();
};

struct MergeCandidate {
  std::size_t left_observation = 0;
  std::size_t right_observation = 0;
  int left_global_id = 0;
  int right_global_id = 0;
  double distance = 0.0;
  double appearance = -1.0;
  double score = 0.0;
  bool identity_gallery_target = false;
};

static std::uint64_t identity_pair_key(int first, int second) {
  const std::uint32_t left = static_cast<std::uint32_t>(std::min(first, second));
  const std::uint32_t right = static_cast<std::uint32_t>(std::max(first, second));
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

struct QualityMetrics {
  double score = 0.0;
  int level = 0;
  double width = 0.0;
  double height = 0.0;
  double area = 0.0;
  double confidence = -1.0;
  double completeness = 0.0;
  double blur = -1.0;
};

static double env_double(const char *name, double fallback) {
  const char *value = std::getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  return end && *end == '\0' && std::isfinite(parsed) ? parsed : fallback;
}

static int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end && *end == '\0' ? static_cast<int>(parsed) : fallback;
}

static bool env_bool(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  if (!value) return fallback;
  return std::string(value) == "1" || std::string(value) == "true";
}

static std::string env_string(const char *name, const char *fallback) {
  const char *value = std::getenv(name);
  return value ? std::string(value) : std::string(fallback);
}

static double timestamp_seconds(guint64 timestamp) {
  return timestamp > 1000000000ULL ? static_cast<double>(timestamp) / 1e9
                                   : static_cast<double>(timestamp);
}

static QualityMetrics evaluate_quality(const NvDsObjectMeta *object,
                                       const NvDsFrameMeta *frame) {
  QualityMetrics result;
  if (!object) return result;
  const NvOSD_RectParams &rect = object->rect_params;
  result.width = std::max(0.0, static_cast<double>(rect.width));
  result.height = std::max(0.0, static_cast<double>(rect.height));
  result.area = result.width * result.height;
  result.confidence = object->confidence >= 0.0f ? object->confidence : object->tracker_confidence;
  if (!std::isfinite(result.confidence) || result.confidence < 0.0) result.confidence = 0.5;
  result.confidence = std::clamp(result.confidence, 0.0, 1.0);

  const double frame_width = frame && frame->source_frame_width > 0
                                 ? static_cast<double>(frame->source_frame_width) : 1280.0;
  const double frame_height = frame && frame->source_frame_height > 0
                                  ? static_cast<double>(frame->source_frame_height) : 720.0;
  const double margin = std::max(2.0, env_double("MV3DT_WORLD_IDENTITY_EDGE_MARGIN_PIXELS", 4.0));
  const bool clipped = rect.left <= margin || rect.top <= margin ||
                       rect.left + rect.width >= frame_width - margin ||
                       rect.top + rect.height >= frame_height - margin;
  result.completeness = clipped ? 0.35 : 1.0;
  result.blur = -1.0;  // Pixel blur requires surface mapping; unavailable at this metadata probe.

  const double height_score = std::clamp(result.height / 220.0, 0.0, 1.0);
  const double area_score = std::clamp(result.area / (frame_width * frame_height * 0.08), 0.0, 1.0);
  result.score = 0.50 * height_score + 0.25 * area_score +
                 0.15 * result.confidence + 0.10 * result.completeness;
  const double high_height = env_double("MV3DT_WORLD_IDENTITY_HIGH_MIN_HEIGHT", 160.0);
  const double medium_height = env_double("MV3DT_WORLD_IDENTITY_MEDIUM_MIN_HEIGHT", 80.0);
  const double high_confidence = env_double("MV3DT_WORLD_IDENTITY_HIGH_MIN_CONFIDENCE", 0.55);
  const double medium_confidence = env_double("MV3DT_WORLD_IDENTITY_MEDIUM_MIN_CONFIDENCE", 0.35);
  if (result.height >= high_height && result.confidence >= high_confidence && !clipped) {
    result.level = 2;
  } else if (result.height >= medium_height && result.confidence >= medium_confidence) {
    result.level = 1;
  }
  return result;
}

static bool read_world_feet(NvDsObjectMeta *object, double &x, double &y) {
  for (NvDsMetaList *item = object->obj_user_meta_list; item; item = item->next) {
    auto *meta = static_cast<NvDsUserMeta *>(item->data);
    if (!meta || meta->base_meta.meta_type != NVDS_OBJ_WORLD_FOOT_LOCATION || !meta->user_meta_data)
      continue;
    const auto *point = static_cast<const float *>(meta->user_meta_data);
    x = point[0];
    y = point[1];
    return std::isfinite(x) && std::isfinite(y);
  }
  return false;
}

static bool normalize(std::vector<float> &value) {
  if (value.empty()) return false;
  double norm = 0.0;
  for (float item : value) {
    if (!std::isfinite(item)) return false;
    norm += static_cast<double>(item) * item;
  }
  norm = std::sqrt(norm);
  if (!std::isfinite(norm) || norm <= 0.0) return false;
  for (float &item : value) item = static_cast<float>(item / norm);
  return true;
}

static double cosine(const std::vector<float> &left, const std::vector<float> &right) {
  if (left.size() != right.size() || left.empty()) return -1.0;
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) value += left[index] * right[index];
  return value;
}

static double gallery_similarity(const std::vector<float> &embedding,
                                 const std::vector<float> &prototype,
                                 const std::deque<std::vector<float>> &gallery) {
  if (embedding.empty()) return -1.0;
  std::vector<double> scores;
  if (!prototype.empty()) scores.push_back(cosine(embedding, prototype));
  for (const auto &sample : gallery) scores.push_back(cosine(embedding, sample));
  if (scores.empty()) return -1.0;
  std::sort(scores.begin(), scores.end(), std::greater<double>());
  const std::size_t count = std::min<std::size_t>(3, scores.size());
  double total = 0.0;
  for (std::size_t index = 0; index < count; ++index) total += scores[index];
  return total / static_cast<double>(count);
}

static double gallery_similarity(const std::vector<float> &embedding, const Track &track) {
  return gallery_similarity(embedding, track.embedding, track.gallery);
}

static double gallery_similarity(const std::vector<float> &embedding,
                                 const GlobalIdentity &identity) {
  return gallery_similarity(embedding, identity.embedding, identity.gallery);
}

static bool diverse_gallery_sample(const std::vector<float> &embedding,
                                   const std::vector<float> &prototype,
                                   const std::deque<std::vector<float>> &gallery,
                                   double minimum_distance) {
  if (embedding.empty()) return false;
  if (prototype.empty() && gallery.empty()) return true;
  if (!prototype.empty() && 1.0 - cosine(embedding, prototype) >= minimum_distance) return true;
  for (const auto &sample : gallery) {
    if (1.0 - cosine(embedding, sample) >= minimum_distance) return true;
  }
  return false;
}

static bool read_reid_embedding(NvDsObjectMeta *object, std::vector<float> &embedding) {
  for (NvDsMetaList *item = object->obj_user_meta_list; item; item = item->next) {
    auto *meta = static_cast<NvDsUserMeta *>(item->data);
    if (!meta || meta->base_meta.meta_type != NVDS_TRACKER_OBJ_REID_META || !meta->user_meta_data)
      continue;
    auto *reid = static_cast<NvDsObjReid *>(meta->user_meta_data);
    if (!reid->ptr_host || reid->featureSize == 0) continue;
    embedding.assign(reid->ptr_host, reid->ptr_host + reid->featureSize);
    if (normalize(embedding)) return true;
    embedding.clear();
  }
  return false;
}

static void set_label(NvDsObjectMeta *object, int global_id) {
  g_free(object->text_params.display_text);
  if (global_id > 0) {
    const std::string label = "G" + std::to_string(global_id);
    object->text_params.display_text = g_strdup(label.c_str());
  } else {
    object->text_params.display_text = g_strdup("");
  }
}

class WorldIdentity {
 public:
  WorldIdentity()
      // The VGGT projection currently has several metres of residual error
      // between the two views. Use this as a birth-match search window only;
      // established local tracks are never reassigned by this geometry gate.
      : max_distance_(env_double("MV3DT_WORLD_IDENTITY_MAX_DISTANCE", 8.00)),
        max_gap_(env_double("MV3DT_WORLD_IDENTITY_MAX_GAP_SECONDS", 2.00)),
        sync_window_(env_double("MV3DT_WORLD_IDENTITY_SYNC_WINDOW_SECONDS", 0.75)),
        global_memory_(env_double("MV3DT_WORLD_IDENTITY_GLOBAL_MEMORY_SECONDS", 60.0)),
        reid_weight_(env_double("MV3DT_WORLD_IDENTITY_REID_WEIGHT", 0.35)),
        min_reid_similarity_(env_double("MV3DT_WORLD_IDENTITY_MIN_REID_SIMILARITY", 0.70)),
        merge_reid_similarity_(env_double("MV3DT_WORLD_IDENTITY_MERGE_REID_SIMILARITY", 0.75)),
        candidate_margin_(env_double("MV3DT_WORLD_IDENTITY_CANDIDATE_MARGIN", 0.05)),
        merge_candidate_margin_(env_double("MV3DT_WORLD_IDENTITY_MERGE_CANDIDATE_MARGIN", 0.08)),
        strong_geometry_distance_(env_double("MV3DT_WORLD_IDENTITY_STRONG_GEOMETRY_DISTANCE", 0.35)),
        min_quality_level_(std::max(0, env_int("MV3DT_WORLD_IDENTITY_MIN_QUALITY_LEVEL", 1))),
        gallery_max_size_(static_cast<std::size_t>(std::max(
            4, env_int("MV3DT_WORLD_IDENTITY_GALLERY_MAX_SIZE", 24)))),
        gallery_min_diversity_(env_double("MV3DT_WORLD_IDENTITY_GALLERY_MIN_DIVERSITY", 0.08)),
        match_confirmations_(static_cast<std::size_t>(std::max(
            1.0, env_double("MV3DT_WORLD_IDENTITY_MATCH_CONFIRMATIONS", 3.0)))),
        merge_confirmations_(static_cast<std::size_t>(std::max(
            2.0, env_double("MV3DT_WORLD_IDENTITY_MERGE_CONFIRMATIONS", 5.0)))),
        birth_confirmations_(static_cast<std::size_t>(std::max(
            1.0, env_double("MV3DT_WORLD_IDENTITY_BIRTH_CONFIRMATIONS", 3.0)))),
        birth_timeout_(env_double("MV3DT_WORLD_IDENTITY_BIRTH_TIMEOUT_SECONDS", 1.5)),
        output_path_(env_string("MV3DT_WORLD_IDENTITY_OUTPUT", "/workspace/runs/mv3dt_world_identity.jsonl")),
        next_id_(1) {
    if (!output_path_.empty()) output_.open(output_path_, std::ios::out | std::ios::app);
    g_message("worldidentity: enabled max-distance=%.2f m max-gap=%.2f s sync-window=%.2f s global-memory=%.2f s "
              "reid-weight=%.2f "
              "min-reid=%.2f merge-reid=%.2f candidate-margin=%.2f merge-margin=%.2f "
              "confirmations=%zu merge-confirmations=%zu birth-confirmations=%zu "
              "birth-timeout=%.2f min-quality=%d gallery-size=%zu",
              max_distance_, max_gap_, sync_window_, global_memory_, reid_weight_, min_reid_similarity_,
              merge_reid_similarity_, candidate_margin_, merge_candidate_margin_, match_confirmations_,
              merge_confirmations_, birth_confirmations_, birth_timeout_, min_quality_level_, gallery_max_size_);
  }

  ~WorldIdentity() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (output_.is_open()) output_.close();
  }

  void process(NvDsBatchMeta *batch, GstClockTime buffer_pts) {
    if (!batch) return;
    std::lock_guard<std::mutex> lock(mutex_);
    // Do not compare camera-provided NTP values for association. The NVR
    // feeds can carry different clock offsets (the observed offset is tens of
    // seconds), which makes simultaneous people look temporally unrelated.
    // The pipeline arrival clock is common to both sources and is used only
    // for matching; the source timestamp is retained in the JSON diagnostics.
    const double processing_seconds = static_cast<double>(g_get_monotonic_time()) / 1000000.0;
    std::vector<Observation> observations;
    for (NvDsMetaList *frame_item = batch->frame_meta_list; frame_item; frame_item = frame_item->next) {
      auto *frame = static_cast<NvDsFrameMeta *>(frame_item->data);
      if (!frame) continue;
      const guint64 timestamp = frame->ntp_timestamp ? frame->ntp_timestamp :
                                (frame->buf_pts ? frame->buf_pts : buffer_pts);
      for (NvDsMetaList *object_item = frame->obj_meta_list; object_item; object_item = object_item->next) {
        auto *object = static_cast<NvDsObjectMeta *>(object_item->data);
        if (!object || object->class_id != 0 || object->object_id == UNTRACKED_OBJECT_ID) continue;
        Observation observation;
        observation.object = object;
        observation.key = {frame->source_id, object->object_id};
        observation.frame = frame->frame_num;
        observation.timestamp = timestamp;
        observation.seconds = processing_seconds;
        const QualityMetrics quality = evaluate_quality(object, frame);
        observation.quality_score = quality.score;
        observation.quality_level = quality.level;
        observation.bbox_width = quality.width;
        observation.bbox_height = quality.height;
        observation.bbox_area = quality.area;
        observation.detector_confidence = quality.confidence;
        observation.crop_completeness = quality.completeness;
        observation.blur_score = quality.blur;
        if (!read_world_feet(object, observation.x, observation.y)) {
          set_label(object, 0);
          continue;
        }
        observation.fresh_embedding = read_reid_embedding(object, observation.embedding);
        if (observation.embedding.empty()) {
          // NvTracker emits Re-ID metadata at the extraction interval rather
          // than necessarily on every buffer. Reuse the local track gallery
          // instead of treating the missing sample as a missing identity.
          auto previous = tracks_.find(observation.key);
          if (previous != tracks_.end()) observation.embedding = previous->second.embedding;
        }
        auto previous = tracks_.find(observation.key);
        observation.verification_samples = previous == tracks_.end()
                                               ? (observation.fresh_embedding ? 1 : 0)
                                               : previous->second.verified_samples +
                                                     (observation.fresh_embedding ? 1 : 0);
        observation.good_embedding_samples = previous == tracks_.end()
                                                 ? (observation.fresh_embedding &&
                                                    observation.quality_level >= min_quality_level_ ? 1 : 0)
                                                 : previous->second.good_embedding_samples +
                                                       (observation.fresh_embedding &&
                                                        observation.quality_level >= min_quality_level_ ? 1 : 0);
        observations.push_back(observation);
      }
    }
    if (observations.empty()) return;

    purge(observations.front().seconds);
    std::vector<Candidate> candidates;
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const auto &observation = observations[index];
      const auto existing = tracks_.find(observation.key);
      const bool existing_track = existing != tracks_.end();
      const bool established_track = existing_track && existing->second.global_id > 0;
      // A local track that already owns an identity is sticky. Cross-camera
      // association is only allowed to initialize a new/pending local track;
      // otherwise nearby people can repeatedly steal the identity every frame.
      if (established_track) continue;
      for (const auto &item : global_identities_) {
        const GlobalIdentity &identity = item.second;
        if (identity.global_id <= 0) continue;
        if (identity.source == observation.key.source) continue;
        const double time_gap = std::abs(observation.seconds - identity.last_seen);
        // The two cameras overlap and are intended to identify people who
        // are visible at the same moment.  Retaining a gallery for 60 seconds
        // is useful, but using a 60-second candidate window creates stale
        // cross-camera matches when RTSP feeds drift or people move away.
        if (time_gap > sync_window_) continue;
        const double distance = std::hypot(observation.x - identity.x, observation.y - identity.y);
        if (distance > max_distance_) continue;
        const double appearance = gallery_similarity(observation.embedding, identity);
        const double spatial_score = 1.0 - std::min(1.0, distance / max_distance_);
        const double temporal_score = std::exp(-time_gap / std::max(0.05, sync_window_));
        const double appearance_score = appearance >= 0.0 ? std::clamp(appearance, 0.0, 1.0) : 0.0;
        const double weight = std::clamp(reid_weight_, 0.0, 1.0);
        const double remaining_weight = 1.0 - weight;
        const double score = weight * appearance_score +
                             remaining_weight * (0.75 * spatial_score + 0.25 * temporal_score);
        // This candidate path is only for a new or pending local track.
        // Established local tracks are handled as sticky identities above;
        // geometry-only evidence can therefore never swap a nearby person
        // into an already-running track.
        // Require multiple independent Re-ID samples before an identity can
        // be transferred between cameras. A single crop can be occluded,
        // blurred, or belong to the neighbouring person.
        // A new local track has no history yet; requiring two samples here
        // would make cross-camera birth matching impossible. The pending
        // state below supplies the temporal confirmation instead. Every
        // confirmation must still contain a fresh tracker Re-ID tensor.
        if (!observation.fresh_embedding || observation.quality_level < min_quality_level_ ||
            appearance < min_reid_similarity_ || identity.good_embedding_samples < 2)
          continue;
        candidates.push_back({index, identity.last_key, canonical_global_id(identity.global_id), distance,
                              time_gap, appearance, score, spatial_score, temporal_score});
      }
    }

    // Bootstrap a shared identity when both cameras create new local tracks
    // in the same batch. Neither track has a Global ID yet, so the gallery
    // path above cannot see this pair. Require usable crops, both signals,
    // and the same multi-observation confirmation rule as normal matching.
    for (std::size_t left = 0; left < observations.size(); ++left) {
      const auto &left_observation = observations[left];
      const auto left_track = tracks_.find(left_observation.key);
      if (left_track != tracks_.end() && left_track->second.global_id > 0) continue;
      if (!left_observation.fresh_embedding || left_observation.embedding.empty() ||
          left_observation.quality_level < min_quality_level_)
        continue;
      for (std::size_t right = left + 1; right < observations.size(); ++right) {
        const auto &right_observation = observations[right];
        if (left_observation.key.source == right_observation.key.source) continue;
        const auto right_track = tracks_.find(right_observation.key);
        if (right_track != tracks_.end() && right_track->second.global_id > 0) continue;
        if (!right_observation.fresh_embedding || right_observation.embedding.empty() ||
            right_observation.quality_level < min_quality_level_)
          continue;
        const double distance = std::hypot(left_observation.x - right_observation.x,
                                           left_observation.y - right_observation.y);
        if (distance > max_distance_) continue;
        const double appearance = cosine(left_observation.embedding, right_observation.embedding);
        if (appearance < min_reid_similarity_) continue;
        const double spatial_score = 1.0 - std::min(1.0, distance / max_distance_);
        const double temporal_score = 1.0;
        const double weight = std::clamp(reid_weight_, 0.0, 1.0);
        const double score = weight * std::clamp(appearance, 0.0, 1.0) +
                             (1.0 - weight) * spatial_score;
        candidates.push_back({left, right_observation.key, 0, distance, 0.0, appearance,
                              score, spatial_score, temporal_score, right});
      }
    }

    // A common overlap-camera failure mode is that both local trackers mint
    // independent IDs before the bootstrap path sees them in the same batch.
    // Do not reassign either active track here. Instead, collect strong
    // evidence that two *established* IDs represent the same person, then
    // canonically merge the identities after repeated confirmation. This
    // preserves stable local IDs and prevents a nearby person from stealing
    // an ID on a single ambiguous frame.
    std::vector<MergeCandidate> merge_candidates;
    for (std::size_t left = 0; left < observations.size(); ++left) {
      const auto &left_observation = observations[left];
      const auto left_track = tracks_.find(left_observation.key);
      if (left_track == tracks_.end() || left_track->second.global_id <= 0 ||
          !left_observation.fresh_embedding || left_observation.embedding.empty() ||
          left_observation.quality_level < min_quality_level_)
        continue;
      const int left_id = canonical_global_id(left_track->second.global_id);
      for (std::size_t right = left + 1; right < observations.size(); ++right) {
        const auto &right_observation = observations[right];
        if (left_observation.key.source == right_observation.key.source) continue;
        const auto right_track = tracks_.find(right_observation.key);
        if (right_track == tracks_.end() || right_track->second.global_id <= 0 ||
            !right_observation.fresh_embedding || right_observation.embedding.empty() ||
            right_observation.quality_level < min_quality_level_)
          continue;
        const int right_id = canonical_global_id(right_track->second.global_id);
        if (left_id <= 0 || right_id <= 0 || left_id == right_id) continue;
        const double distance = std::hypot(left_observation.x - right_observation.x,
                                           left_observation.y - right_observation.y);
        if (distance > max_distance_) continue;
        const double appearance = cosine(left_observation.embedding, right_observation.embedding);
        if (appearance < merge_reid_similarity_) continue;
        const double spatial_score = 1.0 - std::min(1.0, distance / max_distance_);
        const double score = std::clamp(reid_weight_, 0.0, 1.0) * appearance +
                             (1.0 - std::clamp(reid_weight_, 0.0, 1.0)) * spatial_score;
        merge_candidates.push_back({left, right, left_id, right_id, distance, appearance, score, false});
      }
    }

    // Tracklet-level reconciliation also works when the two source frames
    // arrive in adjacent batches. Compare a fresh sample from an established
    // local track with the other camera's maintained identity gallery. The
    // strict synchronization window still prevents an old gallery state from
    // being used as a live same-time match.
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const auto &observation = observations[index];
      const auto local_track = tracks_.find(observation.key);
      if (local_track == tracks_.end() || local_track->second.global_id <= 0 ||
          !observation.fresh_embedding || observation.embedding.empty() ||
          observation.quality_level < min_quality_level_)
        continue;
      const int local_id = canonical_global_id(local_track->second.global_id);
      for (const auto &item : global_identities_) {
        const GlobalIdentity &identity = item.second;
        const int gallery_id = canonical_global_id(identity.global_id);
        if (gallery_id <= 0 || gallery_id == local_id || identity.source == observation.key.source ||
            identity.last_key == observation.key)
          continue;
        const double time_gap = std::abs(observation.seconds - identity.last_seen);
        if (time_gap > sync_window_) continue;
        const double distance = std::hypot(observation.x - identity.x, observation.y - identity.y);
        if (distance > max_distance_) continue;
        const double appearance = gallery_similarity(observation.embedding, identity);
        if (appearance < merge_reid_similarity_) continue;
        const double spatial_score = 1.0 - std::min(1.0, distance / max_distance_);
        const double temporal_score = std::exp(-time_gap / std::max(0.05, sync_window_));
        const double score = std::clamp(reid_weight_, 0.0, 1.0) * appearance +
                             (1.0 - std::clamp(reid_weight_, 0.0, 1.0)) *
                                 (0.75 * spatial_score + 0.25 * temporal_score);
        merge_candidates.push_back({index, std::numeric_limits<std::size_t>::max(), local_id,
                                    gallery_id, distance, appearance, score, true});
      }
    }

    std::sort(merge_candidates.begin(), merge_candidates.end(),
              [](const MergeCandidate &left, const MergeCandidate &right) {
                if (left.score != right.score) return left.score > right.score;
                if (left.appearance != right.appearance) return left.appearance > right.appearance;
                return left.distance < right.distance;
              });
    std::unordered_map<Key, std::vector<const MergeCandidate *>, KeyHash> merge_options;
    for (const auto &candidate : merge_candidates) {
      merge_options[observations[candidate.left_observation].key].push_back(&candidate);
      if (candidate.identity_gallery_target) {
        const auto target = global_identities_.find(candidate.right_global_id);
        if (target != global_identities_.end()) merge_options[target->second.last_key].push_back(&candidate);
      } else {
        merge_options[observations[candidate.right_observation].key].push_back(&candidate);
      }
    }
    std::unordered_map<Key, bool, KeyHash> ambiguous_merge_tracks;
    for (const auto &item : merge_options) {
      const auto &options = item.second;
      if (options.size() >= 2 && options[0]->score - options[1]->score < merge_candidate_margin_)
        ambiguous_merge_tracks[item.first] = true;
    }
    std::unordered_map<Key, bool, KeyHash> claimed_merge_tracks;
    for (const auto &candidate : merge_candidates) {
      const Key &left_key = observations[candidate.left_observation].key;
      Key right_key;
      if (candidate.identity_gallery_target) {
        const auto target = global_identities_.find(candidate.right_global_id);
        if (target == global_identities_.end()) continue;
        right_key = target->second.last_key;
      } else {
        right_key = observations[candidate.right_observation].key;
      }
      if (claimed_merge_tracks[left_key] || claimed_merge_tracks[right_key] ||
          ambiguous_merge_tracks[left_key] || ambiguous_merge_tracks[right_key])
        continue;
      claimed_merge_tracks[left_key] = true;
      claimed_merge_tracks[right_key] = true;
      const std::uint64_t key = identity_pair_key(candidate.left_global_id, candidate.right_global_id);
      PendingIdentityMerge &pending = pending_identity_merges_[key];
      if (pending.confirmations > 0 &&
          observations[candidate.left_observation].seconds - pending.last_seen <= max_gap_) {
        ++pending.confirmations;
        pending.appearance_sum += candidate.appearance;
        pending.score_sum += candidate.score;
      } else {
        pending = PendingIdentityMerge{1, observations[candidate.left_observation].seconds,
                                       candidate.appearance, candidate.score};
      }
      pending.last_seen = observations[candidate.left_observation].seconds;
      if (pending.confirmations < merge_confirmations_) continue;

      const double average_appearance = pending.appearance_sum / pending.confirmations;
      const double average_score = pending.score_sum / pending.confirmations;
      const int merged_id = merge_global_identities(candidate.left_global_id, candidate.right_global_id);
      pending_identity_merges_.erase(key);
      write_assignment(observations[candidate.left_observation], merged_id, "IDENTITY_MERGE",
                       candidate.distance, 0.0, average_appearance, average_score);
      if (!candidate.identity_gallery_target) {
        write_assignment(observations[candidate.right_observation], merged_id, "IDENTITY_MERGE",
                         candidate.distance, 0.0, average_appearance, average_score);
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate &left, const Candidate &right) {
      if (left.score != right.score) return left.score > right.score;
      if (left.distance != right.distance) return left.distance < right.distance;
      return left.time_gap < right.time_gap;
    });

    // If two different identities are nearly tied for one new local track,
    // do not let greedy ordering turn an ambiguous observation into a wrong
    // cross-camera identity. The birth hold below will keep it unassigned
    // until later Re-ID samples make the decision clearer.
    std::unordered_map<std::size_t, std::vector<const Candidate *>> by_observation;
    for (const Candidate &candidate : candidates) {
      by_observation[candidate.observation].push_back(&candidate);
    }
    std::unordered_map<std::size_t, bool> ambiguous_observations;
    for (const auto &item : by_observation) {
      const auto &options = item.second;
      const bool different_targets = options.size() >= 2 &&
          (options[0]->global_id != options[1]->global_id ||
           !(options[0]->track_key == options[1]->track_key));
      if (options.size() >= 2 && different_targets &&
          options[0]->score - options[1]->score < candidate_margin_) {
        ambiguous_observations[item.first] = true;
      }
    }

    std::unordered_map<std::size_t, int> assignments;
    std::unordered_map<std::size_t, Candidate> selected_candidates;
    std::unordered_map<Key, bool, KeyHash> claimed_tracks;
    std::unordered_map<std::size_t, bool> claimed_observations;
    std::unordered_map<guint, std::unordered_map<int, bool>> claimed_ids_by_source;
    for (const Candidate &candidate : candidates) {
      if (assignments.count(candidate.observation) || claimed_observations[candidate.observation] ||
          claimed_tracks.count(candidate.track_key))
        continue;
      if (ambiguous_observations[candidate.observation]) continue;
      const auto &observation = observations[candidate.observation];
      if (candidate.paired_observation != std::numeric_limits<std::size_t>::max() &&
          (candidate.paired_observation >= observations.size() ||
           claimed_observations[candidate.paired_observation]))
        continue;
      if (candidate.global_id > 0 &&
          (claimed_ids_by_source[observation.key.source][candidate.global_id] ||
           global_id_active_on_source(candidate.global_id, observation.key.source,
                                      observation.seconds, candidate.track_key)))
        continue;
      selected_candidates[candidate.observation] = candidate;
      claimed_observations[candidate.observation] = true;
      if (candidate.paired_observation != std::numeric_limits<std::size_t>::max())
        claimed_observations[candidate.paired_observation] = true;
      claimed_tracks[candidate.track_key] = true;
      if (candidate.global_id > 0)
        claimed_ids_by_source[observation.key.source][candidate.global_id] = true;
    }

    // Confirm a candidate across consecutive observations. A candidate is
    // deliberately not assigned a new ID while it is being verified.
    for (const auto &item : selected_candidates) {
      const std::size_t index = item.first;
      const Candidate &candidate = item.second;
      const Observation &observation = observations[index];
      if (candidate.paired_observation != std::numeric_limits<std::size_t>::max()) {
        const std::size_t paired_index = candidate.paired_observation;
        const Observation &paired_observation = observations[paired_index];
        const PairKey pair_key = canonical_pair_key(observation.key, paired_observation.key);
        PendingPair &pending = pending_pairs_[pair_key];
        if (pending.global_id > 0 && observation.seconds - pending.last_seen <= max_gap_) {
          ++pending.confirmations;
        } else {
          pending.global_id = next_id_++;
          pending.confirmations = 1;
        }
        pending.last_seen = observation.seconds;
        pending_matches_[observation.key] =
            PendingMatch{pending.global_id, pending.confirmations, pending.last_seen};
        pending_matches_[paired_observation.key] =
            PendingMatch{pending.global_id, pending.confirmations, pending.last_seen};
        if (pending.confirmations >= match_confirmations_) {
          assignments[index] = pending.global_id;
          assignments[paired_index] = pending.global_id;
          pending_matches_.erase(observation.key);
          pending_matches_.erase(paired_observation.key);
          pending_pairs_.erase(pair_key);
        }
        continue;
      }
      PendingMatch &pending = pending_matches_[observation.key];
      if (pending.global_id == candidate.global_id &&
          observation.seconds - pending.last_seen <= max_gap_) {
        ++pending.confirmations;
      } else {
        pending.global_id = candidate.global_id;
        pending.confirmations = 1;
      }
      pending.last_seen = observation.seconds;
      if (pending.confirmations >= match_confirmations_) {
        assignments[index] = candidate.global_id;
        pending_matches_.erase(observation.key);
      }
    }

    // Prefer established local tracks when resolving a same-camera collision.
    // Never manufacture a replacement global ID: the losing observation stays
    // pending and can recover from a later verified cross-camera match.
    std::vector<std::size_t> processing_order(observations.size());
    std::iota(processing_order.begin(), processing_order.end(), 0);
    std::sort(processing_order.begin(), processing_order.end(), [&](std::size_t left, std::size_t right) {
      const auto left_track = tracks_.find(observations[left].key);
      const auto right_track = tracks_.find(observations[right].key);
      const bool left_established = left_track != tracks_.end() && left_track->second.global_id > 0;
      const bool right_established = right_track != tracks_.end() && right_track->second.global_id > 0;
      if (left_established != right_established) return left_established > right_established;
      const std::size_t left_samples = left_track == tracks_.end() ? 0 : left_track->second.verified_samples;
      const std::size_t right_samples = right_track == tracks_.end() ? 0 : right_track->second.verified_samples;
      return left_samples > right_samples;
    });

    std::unordered_map<guint, std::unordered_map<int, bool>> displayed_ids_by_source;
    for (const std::size_t index : processing_order) {
      const auto &observation = observations[index];
      int global_id = 0;
      const char *pending_reason = nullptr;
      const auto assigned = assignments.find(index);
      auto existing = tracks_.find(observation.key);
      if (assigned != assignments.end()) {
        global_id = canonical_global_id(assigned->second);
      } else if (selected_candidates.count(index)) {
        // A candidate is still being confirmed. Keep this local track
        // unassigned instead of minting a speculative global ID.
        global_id = 0;
        pending_reason = "PENDING_CROSS_CAMERA_MATCH";
      } else if (existing != tracks_.end()) {
        global_id = canonical_global_id(existing->second.global_id);
        if (global_id <= 0 && pending_matches_.find(observation.key) == pending_matches_.end() &&
            birth_ready(existing->second, observation)) {
          global_id = next_id_++;
          write_assignment(observation, global_id, "NEW_WORLD_TRACK", 0.0, 0.0, -1.0, 0.0);
        } else if (global_id <= 0) {
          pending_reason = "BIRTH_PENDING";
        }
      } else {
        // Hold the first samples without minting an identity. This gives the
        // Re-ID gallery and the cross-camera matcher time to attach a new
        // local track to an existing person instead of creating a duplicate.
        global_id = 0;
        pending_reason = "BIRTH_PENDING";
      }
      if (global_id > 0 && displayed_ids_by_source[observation.key.source][global_id]) {
        // This is a conflict, not a new person. Do not increment next_id_.
        // Keep the internal track pending so it can be resolved later.
        global_id = 0;
        pending_reason = "ID_CONFLICT_PENDING";
      }
      if (global_id > 0) displayed_ids_by_source[observation.key.source][global_id] = true;
      if (pending_reason) {
        const auto selected = selected_candidates.find(index);
        if (selected != selected_candidates.end()) {
          const auto &candidate = selected->second;
          write_assignment(observation, 0, pending_reason, candidate.distance,
                           candidate.time_gap, candidate.appearance, candidate.score,
                           candidate.spatial_score, candidate.temporal_score);
        }
      } else if (assigned != assignments.end()) {
        const auto selected = selected_candidates.find(index);
        if (selected != selected_candidates.end()) {
          const auto &candidate = selected->second;
          const char *reason = candidate.paired_observation != std::numeric_limits<std::size_t>::max()
                                   ? "BOOTSTRAP_PAIR_MATCH"
                                   : "WORLD_DISTANCE_MATCH";
          write_assignment(observation, global_id, reason, candidate.distance,
                           candidate.time_gap, candidate.appearance, candidate.score,
                           candidate.spatial_score, candidate.temporal_score);
        }
      }
      auto previous = tracks_.find(observation.key);
      std::vector<float> gallery;
      std::size_t verified_samples = 0;
      std::size_t good_embedding_samples = 0;
      if (previous != tracks_.end()) gallery = previous->second.embedding;
      if (previous != tracks_.end()) {
        verified_samples = previous->second.verified_samples;
        good_embedding_samples = previous->second.good_embedding_samples;
      }
      std::deque<std::vector<float>> gallery_samples;
      if (previous != tracks_.end()) gallery_samples = previous->second.gallery;
      if (observation.fresh_embedding && !observation.embedding.empty() &&
          observation.quality_level >= min_quality_level_) {
        if (diverse_gallery_sample(observation.embedding, gallery, gallery_samples,
                                   gallery_min_diversity_)) {
          if (gallery_samples.size() >= gallery_max_size_) gallery_samples.pop_front();
          gallery_samples.push_back(observation.embedding);
        }
        ++verified_samples;
        ++good_embedding_samples;
        if (observation.quality_level >= 2 || gallery.empty())
          update_embedding(gallery, observation.embedding);
      } else if (!observation.fresh_embedding && gallery.empty() && !observation.embedding.empty()) {
        gallery = observation.embedding;
      }
      const double first_seen = previous != tracks_.end() && previous->second.first_seen > 0.0
                                    ? previous->second.first_seen : observation.seconds;
      const std::size_t observed_samples = previous != tracks_.end()
                                                ? previous->second.observed_samples + 1 : 1;
      tracks_[observation.key] = Track{global_id, observation.seconds, observation.x, observation.y,
                                       observation.key.source, std::move(gallery),
                                       std::move(gallery_samples), verified_samples,
                                       good_embedding_samples,
                                       first_seen, observed_samples};
      update_global_identity(global_id, observation);
      set_label(observation.object, global_id);
    }
  }

 private:
  int merge_global_identities(int first_id, int second_id) {
    const int first = canonical_global_id(first_id);
    const int second = canonical_global_id(second_id);
    if (first <= 0) return second;
    if (second <= 0 || first == second) return first;

    // Keep the lower ID as the visible canonical ID. This makes a merge
    // deterministic and avoids oscillating labels when the cameras swap
    // arrival order from batch to batch.
    const int winner = std::min(first, second);
    const int loser = std::max(first, second);
    canonical_ids_[loser] = winner;

    auto loser_identity = global_identities_.find(loser);
    if (loser_identity != global_identities_.end()) {
      GlobalIdentity &target = global_identities_[winner];
      target.global_id = winner;
      if (target.embedding.empty()) {
        target.embedding = loser_identity->second.embedding;
      } else if (!loser_identity->second.embedding.empty()) {
        update_embedding(target.embedding, loser_identity->second.embedding);
      }
      for (const auto &sample : loser_identity->second.gallery) {
        if (!diverse_gallery_sample(sample, target.embedding, target.gallery, gallery_min_diversity_))
          continue;
        if (target.gallery.size() >= gallery_max_size_) target.gallery.pop_front();
        target.gallery.push_back(sample);
      }
      target.good_embedding_samples += loser_identity->second.good_embedding_samples;
      if (loser_identity->second.last_seen > target.last_seen) {
        target.last_key = loser_identity->second.last_key;
        target.source = loser_identity->second.source;
        target.last_seen = loser_identity->second.last_seen;
        target.x = loser_identity->second.x;
        target.y = loser_identity->second.y;
      }
      global_identities_.erase(loser_identity);
    }

    // Persist the canonical ID in active local tracks as well as in the alias
    // table. This keeps same-camera collision checks strict after a merge.
    for (auto &item : tracks_) {
      if (canonical_global_id(item.second.global_id) == winner) item.second.global_id = winner;
    }
    return winner;
  }

  void update_global_identity(int global_id, const Observation &observation) {
    if (global_id <= 0) return;
    GlobalIdentity &identity = global_identities_[global_id];
    identity.global_id = global_id;
    identity.last_key = observation.key;
    identity.source = observation.key.source;
    identity.last_seen = observation.seconds;
    identity.x = observation.x;
    identity.y = observation.y;
    if (!observation.fresh_embedding || observation.embedding.empty() ||
        observation.quality_level < min_quality_level_)
      return;
    if (diverse_gallery_sample(observation.embedding, identity.embedding, identity.gallery,
                               gallery_min_diversity_)) {
      if (identity.gallery.size() >= gallery_max_size_) identity.gallery.pop_front();
      identity.gallery.push_back(observation.embedding);
    }
    ++identity.good_embedding_samples;
    if (observation.quality_level >= 2 || identity.embedding.empty())
      update_embedding(identity.embedding, observation.embedding);
  }

  void purge(double now) {
    for (auto iterator = tracks_.begin(); iterator != tracks_.end();) {
      if (now - iterator->second.last_seen > max_gap_ * 2.0) iterator = tracks_.erase(iterator);
      else ++iterator;
    }
    for (auto iterator = pending_matches_.begin(); iterator != pending_matches_.end();) {
      if (now - iterator->second.last_seen > max_gap_ * 2.0) iterator = pending_matches_.erase(iterator);
      else ++iterator;
    }
    for (auto iterator = global_identities_.begin(); iterator != global_identities_.end();) {
      if (now - iterator->second.last_seen > global_memory_) iterator = global_identities_.erase(iterator);
      else ++iterator;
    }
    for (auto iterator = pending_pairs_.begin(); iterator != pending_pairs_.end();) {
      if (now - iterator->second.last_seen > max_gap_ * 2.0) iterator = pending_pairs_.erase(iterator);
      else ++iterator;
    }
    for (auto iterator = pending_identity_merges_.begin(); iterator != pending_identity_merges_.end();) {
      if (now - iterator->second.last_seen > max_gap_ * 2.0)
        iterator = pending_identity_merges_.erase(iterator);
      else
        ++iterator;
    }
  }

  bool birth_ready(const Track &track, const Observation &observation) const {
    if (observation.quality_level < min_quality_level_) return false;
    const double age = std::max(0.0, observation.seconds - track.first_seen);
    const std::size_t samples = track.observed_samples + 1;
    return observation.good_embedding_samples >= birth_confirmations_ ||
           (samples >= birth_confirmations_ && age >= birth_timeout_);
  }

  int canonical_global_id(int id) {
    if (id <= 0) return 0;
    int root = id;
    auto found = canonical_ids_.find(root);
    while (found != canonical_ids_.end() && found->second != root) {
      root = found->second;
      found = canonical_ids_.find(root);
    }
    canonical_ids_[id] = root;
    return root;
  }

  bool global_id_active_on_source(int global_id, guint source, double now,
                                  const Key &except_key) const {
    for (const auto &item : tracks_) {
      if (item.first == except_key) continue;
      const Track &track = item.second;
      if (track.source == source && track.global_id == global_id &&
          std::abs(now - track.last_seen) <= max_gap_ * 2.0)
        return true;
    }
    return false;
  }

  void write_assignment(const Observation &observation, int global_id, const char *reason,
                        double distance, double time_gap, double appearance = -1.0,
                        double score = 0.0, double spatial_score = 0.0,
                        double temporal_score = 0.0) {
    if (!output_.is_open()) return;
    output_ << "{\"type\":\"world_identity_assignment\",\"timestamp\":" << observation.timestamp
            << ",\"frame\":" << observation.frame
            << ",\"source_id\":" << observation.key.source
            << ",\"local_object_id\":" << observation.key.local_id
            << ",\"global_id\":" << global_id
            << ",\"reason\":\"" << reason << "\""
            << ",\"world_feet\":[" << observation.x << ',' << observation.y << ']'
            << ",\"distance_m\":" << distance << ",\"time_gap_s\":" << time_gap
            << ",\"reid_cosine\":" << appearance << ",\"combined_score\":" << score
            << ",\"spatial_score\":" << spatial_score
            << ",\"temporal_score\":" << temporal_score
            << ",\"quality_score\":" << observation.quality_score
            << ",\"quality_level\":" << observation.quality_level
            << ",\"bbox_width\":" << observation.bbox_width
            << ",\"bbox_height\":" << observation.bbox_height
            << ",\"bbox_area\":" << observation.bbox_area
            << ",\"detector_confidence\":" << observation.detector_confidence
            << ",\"crop_completeness\":" << observation.crop_completeness
            << ",\"blur_score\":" << observation.blur_score
            << ",\"fresh_embedding\":" << (observation.fresh_embedding ? 1 : 0)
            << ",\"verification_samples\":" << observation.verification_samples
            << ",\"good_embedding_samples\":" << observation.good_embedding_samples << "}\n";
    output_.flush();
  }

  const double max_distance_;
  const double max_gap_;
  const double sync_window_;
  const double global_memory_;
  const double reid_weight_;
  const double min_reid_similarity_;
  const double merge_reid_similarity_;
  const double candidate_margin_;
  const double merge_candidate_margin_;
  const double strong_geometry_distance_;
  const int min_quality_level_;
  const std::size_t gallery_max_size_;
  const double gallery_min_diversity_;
  const std::size_t match_confirmations_;
  const std::size_t merge_confirmations_;
  const std::size_t birth_confirmations_;
  const double birth_timeout_;
  const std::string output_path_;
  std::ofstream output_;
  std::mutex mutex_;
  std::unordered_map<Key, Track, KeyHash> tracks_;
  std::unordered_map<Key, PendingMatch, KeyHash> pending_matches_;
  std::unordered_map<PairKey, PendingPair, PairKeyHash> pending_pairs_;
  std::unordered_map<std::uint64_t, PendingIdentityMerge> pending_identity_merges_;
  std::unordered_map<int, GlobalIdentity> global_identities_;
  std::unordered_map<int, int> canonical_ids_;
  int next_id_;
};

typedef struct _GstMv3dtWorldIdentity {
  GstBaseTransform parent;
  WorldIdentity *identity;
} GstMv3dtWorldIdentity;

typedef struct _GstMv3dtWorldIdentityClass {
  GstBaseTransformClass parent_class;
} GstMv3dtWorldIdentityClass;

#define GST_TYPE_MV3DT_WORLD_IDENTITY (gst_mv3dt_world_identity_get_type())
#define GST_MV3DT_WORLD_IDENTITY(obj) ((GstMv3dtWorldIdentity *)(obj))
G_DEFINE_TYPE(GstMv3dtWorldIdentity, gst_mv3dt_world_identity, GST_TYPE_BASE_TRANSFORM)

static GstFlowReturn transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = GST_MV3DT_WORLD_IDENTITY(base);
  if (self->identity) self->identity->process(gst_buffer_get_nvds_batch_meta(buffer), GST_BUFFER_PTS(buffer));
  return GST_FLOW_OK;
}

static void finalize(GObject *object) {
  auto *self = GST_MV3DT_WORLD_IDENTITY(object);
  delete self->identity;
  self->identity = nullptr;
  G_OBJECT_CLASS(gst_mv3dt_world_identity_parent_class)->finalize(object);
}

static void gst_mv3dt_world_identity_class_init(GstMv3dtWorldIdentityClass *klass) {
  GST_BASE_TRANSFORM_CLASS(klass)->transform_ip = transform_ip;
  G_OBJECT_CLASS(klass)->finalize = finalize;
  auto *element = GST_ELEMENT_CLASS(klass);
  static GstStaticPadTemplate sink_template =
      GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  static GstStaticPadTemplate src_template =
      GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  gst_element_class_add_static_pad_template(element, &sink_template);
  gst_element_class_add_static_pad_template(element, &src_template);
  gst_element_class_set_static_metadata(element, "MV3DT World Identity", "Filter/Metadata",
                                        "Calibrated cross-camera identity assignment", PACKAGE);
}

static void gst_mv3dt_world_identity_init(GstMv3dtWorldIdentity *self) {
  self->identity = new WorldIdentity();
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "mv3dtworldidentity", GST_RANK_NONE, GST_TYPE_MV3DT_WORLD_IDENTITY);
}

}  // namespace

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mv3dtworldidentity,
                  "MV3DT calibrated world identity", plugin_init, "1.0", "Proprietary", PACKAGE,
                  "https://nvidia.com")
