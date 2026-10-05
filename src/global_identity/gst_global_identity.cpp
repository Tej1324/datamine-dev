#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <nvds_tracker_meta.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef PACKAGE
#define PACKAGE "datamine-cv"
#endif

namespace {

struct Key {
  guint source = 0;
  guint64 object = 0;
  bool operator==(const Key &other) const { return source == other.source && object == other.object; }
};

struct KeyHash {
  std::size_t operator()(const Key &key) const {
    return (static_cast<std::size_t>(key.source) << 32) ^ static_cast<std::size_t>(key.object);
  }
};

struct Observation {
  Key key;
  guint64 frame = 0;
  guint64 timestamp = 0;
  double seconds = 0.0;
  float left = 0.0f, top = 0.0f, width = 0.0f, height = 0.0f;
  float detector_confidence = 0.0f;
  float tracker_confidence = 0.0f;
  std::vector<float> embedding;
  bool world_valid = false;
  double world_x = 0.0, world_y = 0.0;
};

struct PendingEmbedding {
  std::vector<float> vector;
  guint source = 0;
  double seconds = 0.0;
  float quality = 0.0f;
  int view_bucket = 0;
};

struct GalleryEntry {
  std::vector<float> vector;
  guint source = 0;
  double seconds = 0.0;
  float quality = 0.0f;
  int view_bucket = 0;
};

// Positive-only staff gallery entries.  These are deliberately separate from
// the customer Global-ID gallery: a staff match must never create or alter a
// Global Person ID.
struct StaffGalleryEntry {
  std::vector<float> vector;
  int source = -1;
  float quality = 0.0f;
  int view_bucket = 0;
};

enum class LifecycleState {
  ENTRY_PENDING,
  ACTIVE_VERIFIED,
  EXIT_PENDING,
  CLOSED,
};

struct LocalState {
  double last_seen = 0.0;
  int assigned = 0;
  int good = 0;
  double last_gallery_update = -1.0;
  double last_revalidated = -1.0;
  guint last_source = 0;
  LifecycleState lifecycle = LifecycleState::ENTRY_PENDING;
  int candidate_id = 0;
  int candidate_hits = 0;
  double candidate_last_seen = -1.0;
  int revalidation_failures = 0;
  std::deque<int> staff_reid_votes;
  bool staff_reid_confirmed = false;
  std::deque<PendingEmbedding> pending;
};

struct Person {
  int id = 0;
  double last_seen = 0.0;
  guint last_source = 0;
  LifecycleState lifecycle = LifecycleState::ACTIVE_VERIFIED;
  std::unordered_map<guint, std::pair<guint64, double>> active_tracks;
  std::deque<GalleryEntry> gallery;
  struct CameraWorld {
    bool valid = false;
    double x = 0.0, y = 0.0, seconds = 0.0;
  };
  std::unordered_map<guint, CameraWorld> camera_world;
  bool world_valid = false;
  double world_x = 0.0, world_y = 0.0;
};

static double env_double(const char *name, double fallback);
static bool env_bool(const char *name, bool fallback);
static std::string env_string(const char *name, const char *fallback);

class FloorCalibration {
 public:
  FloorCalibration() {
    enabled_ = env_bool("GLOBAL_REID_CALIBRATION_ENABLE", true);
    directory_ = env_string("GLOBAL_REID_CALIBRATION_DIR", "");
    source_indices_ = parse_source_list(env_string("GLOBAL_REID_CALIBRATION_SOURCES", "2,5"));
    max_distance_ = env_double("GLOBAL_REID_CALIBRATION_MAX_DISTANCE", 3.0);
    pair_max_distance_ = env_double("GLOBAL_REID_CALIBRATION_PAIR_MAX_DISTANCE", 0.90);
    max_gap_ = env_double("GLOBAL_REID_CALIBRATION_MAX_GAP_SECONDS", 3.0);
    if (!enabled_ || directory_.empty()) return;
    for (std::size_t i = 0; i < source_indices_.size(); ++i) {
      Matrix inverse{};
      if (load_inverse(directory_ + "/cam_" + (i < 10 ? "0" : "") + std::to_string(i) + ".yml", inverse)) {
        inverses_[source_indices_[i]] = inverse;
      }
    }
    g_message("globalidentity: calibrated geometry %s, cameras=%zu, max-distance=%.2f m, max-gap=%.2f s",
              inverses_.empty() ? "unavailable" : "enabled", inverses_.size(), max_distance_, max_gap_);
  }

  bool project(guint source, float left, float top, float width, float height,
               double &x, double &y) const {
    auto found = inverses_.find(source);
    if (found == inverses_.end()) return false;
    const double u = left + width * 0.5;
    const double v = top + height;
    const Matrix &m = found->second;
    const double w = m[6] * u + m[7] * v + m[8];
    if (!std::isfinite(w) || std::abs(w) < 1e-9) return false;
    x = (m[0] * u + m[1] * v + m[2]) / w;
    y = (m[3] * u + m[4] * v + m[5]) / w;
    return std::isfinite(x) && std::isfinite(y);
  }

  bool has_pair_geometry(const Observation &observation, const Person &person) const {
    return enabled_ && observation.world_valid && reference_position(observation, person).first;
  }

  bool allows(const Observation &observation, const Person &person) const {
    if (!enabled_ || !observation.world_valid) return true;
    const auto reference = reference_position(observation, person);
    if (!reference.first) return true;
    const double gap = std::abs(observation.seconds - reference.second.seconds);
    if (gap > max_gap_) return false;
    const double dx = observation.world_x - reference.second.x;
    const double dy = observation.world_y - reference.second.y;
    return std::sqrt(dx * dx + dy * dy) <= pair_max_distance_;
  }

  float combined_score(const Observation &observation, const Person &person, float appearance) const {
    if (!enabled_ || !observation.world_valid) return appearance;
    const auto reference = reference_position(observation, person);
    if (!reference.first) return appearance;
    const double dx = observation.world_x - reference.second.x;
    const double dy = observation.world_y - reference.second.y;
    const double distance = std::sqrt(dx * dx + dy * dy);
    const double spatial = 1.0 - std::min(1.0, distance / pair_max_distance_);
    const double weight = env_double("GLOBAL_REID_CALIBRATION_SCORE_WEIGHT", 0.45);
    return static_cast<float>((1.0 - weight) * appearance + weight * spatial);
  }

 private:
  using Matrix = std::array<double, 9>;

  std::pair<bool, Person::CameraWorld> reference_position(const Observation &observation,
                                                           const Person &person) const {
    Person::CameraWorld best;
    bool found = false;
    for (const auto &item : person.camera_world) {
      if (item.first == observation.key.source || !item.second.valid) continue;
      if (!found || item.second.seconds > best.seconds) {
        best = item.second;
        found = true;
      }
    }
    if (found) return {true, best};
    if (person.world_valid && person.last_source != observation.key.source)
      return {true, Person::CameraWorld{true, person.world_x, person.world_y, person.last_seen}};
    return {false, Person::CameraWorld{}};
  }

  static std::vector<guint> parse_source_list(const std::string &value) {
    std::vector<guint> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
      char *end = nullptr;
      const long id = std::strtol(item.c_str(), &end, 10);
      if (end && *end == '\0' && id >= 0) result.push_back(static_cast<guint>(id));
    }
    return result;
  }

  static bool load_inverse(const std::string &path, Matrix &inverse) {
    std::ifstream input(path);
    if (!input.is_open()) return false;
    std::stringstream contents;
    contents << input.rdbuf();
    const std::string text = contents.str();
    const std::size_t start = text.find("projectionMatrix_3x4_w2p");
    if (start == std::string::npos) return false;
    const std::string values = text.substr(start);
    static const std::regex number(R"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::sregex_iterator it(values.begin(), values.end(), number), end;
    std::vector<double> p;
    for (; it != end && p.size() < 12; ++it) p.push_back(std::stod(it->str()));
    if (p.size() != 12) return false;
    const double a = p[0], b = p[1], c = p[3];
    const double d = p[4], e = p[5], f = p[7];
    const double g = p[8], h = p[9], i = p[11];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(det) || std::abs(det) < 1e-10) return false;
    inverse = {(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
               (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
               (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
    return true;
  }

  bool enabled_ = false;
  std::string directory_;
  std::vector<guint> source_indices_;
  std::unordered_map<guint, Matrix> inverses_;
  double max_distance_ = 3.0;
  double pair_max_distance_ = 0.90;
  double max_gap_ = 3.0;
};

struct DisplayId {
  int id = 0;
  double seconds = 0.0;
};

struct ActiveClaim {
  int id = 0;
  double last_seen = 0.0;
  float quality = 0.0f;
};

static double env_double(const char *name, double fallback) {
  const char *value = std::getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  const double result = std::strtod(value, &end);
  return end && *end == '\0' ? result : fallback;
}

static int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  const long result = std::strtol(value, &end, 10);
  return end && *end == '\0' ? static_cast<int>(result) : fallback;
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

static std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
  return value;
}

static std::unordered_set<guint> parse_sources(const std::string &value) {
  std::unordered_set<guint> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    while (!item.empty() && item.front() == ' ') item.erase(item.begin());
    while (!item.empty() && item.back() == ' ') item.pop_back();
    if (item == "ground_01" || item == "CH7") result.insert(0);
    else if (item == "ground_02" || item == "CH16") result.insert(1);
    else if (item == "ground_03" || item == "CH10") result.insert(2);
    else if (item == "ground_04" || item == "CH18") result.insert(3);
    else if (item == "ground_05" || item == "CH20") result.insert(4);
    else if (item == "ground_06" || item == "CH11") result.insert(5);
    else {
      char *end = nullptr;
      const long id = std::strtol(item.c_str(), &end, 10);
      if (end && *end == '\0' && id >= 0) result.insert(static_cast<guint>(id));
    }
  }
  return result;
}

static std::string camera_pair_key(guint left, guint right) {
  if (left > right) std::swap(left, right);
  return std::to_string(left) + "-" + std::to_string(right);
}

static std::unordered_set<std::string> parse_camera_pairs(const std::string &value) {
  std::unordered_set<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::size_t separator = item.find_first_of("-:");
    if (separator == std::string::npos) continue;
    char *left_end = nullptr;
    char *right_end = nullptr;
    const long left = std::strtol(item.substr(0, separator).c_str(), &left_end, 10);
    const long right = std::strtol(item.substr(separator + 1).c_str(), &right_end, 10);
    if (left_end && *left_end == '\0' && right_end && *right_end == '\0' && left >= 0 && right >= 0)
      result.insert(camera_pair_key(static_cast<guint>(left), static_cast<guint>(right)));
  }
  return result;
}

static double timestamp_seconds(guint64 timestamp) {
  if (timestamp > 1000000000ULL) return static_cast<double>(timestamp) / 1e9;
  return static_cast<double>(timestamp);
}

static float cosine(const std::vector<float> &left, const std::vector<float> &right) {
  if (left.size() != right.size() || left.empty()) return -1.0f;
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) value += left[index] * right[index];
  return static_cast<float>(value);
}

static int view_bucket(const Observation &observation) {
  // NvDCF exposes the embedding but not a reliable front/side/back label.
  // Use camera plus person scale as a stable, conservative view proxy. The
  // gallery still keeps multiple embeddings within each bucket.
  const int scale = observation.height < 110.0f ? 0 : observation.height < 220.0f ? 1 : 2;
  return static_cast<int>(observation.key.source) * 10 + scale;
}

static const char *lifecycle_name(LifecycleState state) {
  switch (state) {
    case LifecycleState::ENTRY_PENDING: return "ENTRY_PENDING";
    case LifecycleState::ACTIVE_VERIFIED: return "ACTIVE_VERIFIED";
    case LifecycleState::EXIT_PENDING: return "EXIT_PENDING";
    case LifecycleState::CLOSED: return "CLOSED";
  }
  return "UNKNOWN";
}

static bool normalize(std::vector<float> &value) {
  if (value.size() != 256) return false;
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

static std::string json_string(const std::string &value) {
  std::string result = "\"";
  for (char c : value) {
    if (c == '\\' || c == '\"') result += '\\';
    result += c;
  }
  result += '"';
  return result;
}

class IdentityBackend {
 public:
  IdentityBackend()
      : threshold_(env_double("GLOBAL_REID_MATCH_THRESHOLD", 0.62)),
        min_margin_(env_double("GLOBAL_REID_MIN_MARGIN", 0.05)),
        max_gap_(env_double("GLOBAL_REID_MAX_GAP_SECONDS", 3600.0)),
        gallery_size_(std::max(1, env_int("GLOBAL_REID_GALLERY_SIZE", 24))),
        gallery_interval_(env_double("GLOBAL_REID_GALLERY_MIN_INTERVAL_SECONDS", 0.5)),
        min_good_(std::max(1, env_int("GLOBAL_REID_MIN_GOOD_OBSERVATIONS", 3))),
        match_confirmations_(std::max(1, env_int("GLOBAL_REID_MATCH_CONFIRMATIONS", 2))),
        cross_match_threshold_(static_cast<float>(env_double("GLOBAL_REID_CROSS_CAMERA_MATCH_THRESHOLD", 0.76))),
        cross_min_margin_(static_cast<float>(env_double("GLOBAL_REID_CROSS_CAMERA_MIN_MARGIN", 0.08))),
        cross_match_confirmations_(std::max(1, env_int("GLOBAL_REID_CROSS_CAMERA_CONFIRMATIONS", 3))),
        cross_min_quality_(static_cast<float>(env_double("GLOBAL_REID_CROSS_CAMERA_MIN_QUALITY", 0.65))),
        candidate_window_(env_double("GLOBAL_REID_MATCH_CONFIRMATION_WINDOW_SECONDS", 4.0)),
        sample_every_(std::max(1, env_int("GLOBAL_REID_SAMPLE_EVERY_FRAMES", 5))),
        same_camera_gap_(env_double("GLOBAL_REID_SAME_CAMERA_ACTIVE_GAP_SECONDS", 3.0)),
        active_claim_gap_(env_double("GLOBAL_REID_ACTIVE_CLAIM_GAP_SECONDS", 5.0)),
        simultaneous_gap_(env_double("GLOBAL_REID_SIMULTANEOUS_CAMERA_GAP_SECONDS", 1.5)),
        revalidate_interval_(env_double("GLOBAL_REID_REVALIDATE_INTERVAL_SECONDS", 2.0)),
        revalidate_failures_required_(std::max(1, env_int("GLOBAL_REID_REVALIDATE_FAILURES", 2))),
        min_detector_(env_double("GLOBAL_REID_MIN_DETECTOR_CONFIDENCE", 0.30)),
        min_tracker_(env_double("GLOBAL_REID_MIN_TRACKER_CONFIDENCE", 0.30)),
        staff_reid_requested_(env_bool("STAFF_REID_FILTER_ENABLE", false)),
        staff_reid_threshold_(static_cast<float>(env_double("STAFF_REID_MATCH_THRESHOLD", 0.86))),
        staff_reid_min_quality_(static_cast<float>(env_double("STAFF_REID_MIN_QUALITY", 0.62))),
        staff_reid_min_votes_(std::max(1, env_int("STAFF_REID_MIN_VOTES", 3))),
        staff_reid_vote_window_(std::max(1, env_int("STAFF_REID_VOTE_WINDOW", 5))),
        staff_reid_gallery_path_(env_string("STAFF_REID_GALLERY_PATH", "data/staff_filter/reid_gallery.tsv")),
        enrollment_output_path_(env_string("STAFF_ENROLLMENT_OUTPUT", "runs/staff_enrollment_live.json")),
        output_path_(env_string("GLOBAL_REID_OUTPUT", "runs/global_identity.jsonl")),
        identity_sources_(parse_sources(env_string("GLOBAL_REID_IDENTITY_CAMERAS", "ground_01"))),
        simultaneous_pairs_(parse_camera_pairs(env_string("GLOBAL_REID_SIMULTANEOUS_CAMERA_PAIRS", "2-5"))),
        queue_limit_(std::max(32, env_int("GLOBAL_REID_CPP_QUEUE_SIZE", 512))),
        stop_(false), next_id_(std::max(1, env_int("GLOBAL_REID_START_ID", 1))) {
    const std::filesystem::path output_path(output_path_);
    if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
    output_.open(output_path_, std::ios::out | std::ios::app);
    load_staff_gallery();
    if (env_bool("STAFF_ENROLLMENT_EXPORT_ENABLE", false)) {
      const std::filesystem::path enrollment_path(enrollment_output_path_);
      if (!enrollment_path.parent_path().empty()) std::filesystem::create_directories(enrollment_path.parent_path());
    }
    worker_ = std::thread(&IdentityBackend::run, this);
  }

  ~IdentityBackend() { close(); }

  void close() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stop_) return;
      stop_ = true;
    }
    queue_condition_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (output_.is_open()) output_.close();
  }

  bool should_sample(const Key &key, guint64 frame) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto found = last_sampled_.find(key);
    if (found != last_sampled_.end() && frame - found->second < static_cast<guint64>(sample_every_)) return false;
    last_sampled_[key] = frame;
    return true;
  }

  bool submit(Observation observation) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (stop_ || queue_.size() >= static_cast<std::size_t>(queue_limit_)) {
      dropped_++;
      return false;
    }
    queue_.push_back(std::move(observation));
    queue_condition_.notify_one();
    return true;
  }

  bool lookup(const Key &key, double now, int &id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto found = display_ids_.find(key);
    if (found == display_ids_.end() || now - found->second.seconds > max_gap_) return false;
    id = found->second.id;
    return true;
  }

  bool is_staff(const Key &key, double now) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto found = local_tracks_.find(key);
    if (found == local_tracks_.end() || now - found->second.last_seen > max_gap_) return false;
    return found->second.staff_reid_confirmed;
  }

  void publish_enrollment_snapshot(const std::vector<Observation> &observations) {
    if (!env_bool("STAFF_ENROLLMENT_EXPORT_ENABLE", false)) return;
    std::lock_guard<std::mutex> lock(enrollment_mutex_);
    const std::filesystem::path output_path(enrollment_output_path_);
    const std::filesystem::path temporary = output_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output.is_open()) return;
    const auto now = std::chrono::system_clock::now();
    const double updated_at = std::chrono::duration<double>(now.time_since_epoch()).count();
    output << std::setprecision(17) << "{\"updated_at\":" << updated_at
           << ",\"observations\":[";
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const auto &item = observations[index];
      if (index) output << ',';
      output << "{\"source_id\":" << item.key.source
             << ",\"local_object_id\":" << item.key.object
             << ",\"frame_number\":" << item.frame
             << ",\"timestamp\":" << item.timestamp
             << ",\"bbox\":[" << item.left << ',' << item.top << ','
             << item.width << ',' << item.height << "]"
             << ",\"detector_confidence\":" << item.detector_confidence
             << ",\"tracker_confidence\":" << item.tracker_confidence
             << ",\"embedding\":[";
      for (std::size_t vector_index = 0; vector_index < item.embedding.size(); ++vector_index) {
        if (vector_index) output << ',';
        output << item.embedding[vector_index];
      }
      output << "]}";
    }
    output << "]}\n";
    output.close();
    std::error_code error;
    std::filesystem::rename(temporary, output_path, error);
    if (error) std::filesystem::remove(temporary);
  }

 private:
  void run() {
    while (true) {
      Observation observation;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_condition_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty() && stop_) break;
        observation = std::move(queue_.front());
        queue_.pop_front();
      }
      process(observation);
    }
  }

  float quality(const Observation &observation) const {
    if (observation.detector_confidence < min_detector_ || observation.tracker_confidence < min_tracker_ ||
        observation.width < 24.0f || observation.height < 64.0f) return 0.0f;
    const float size = std::min(observation.width / 48.0f, observation.height / 128.0f);
    return std::clamp(0.35f * observation.detector_confidence +
                      0.35f * observation.tracker_confidence +
                      0.20f + 0.10f * std::min(1.0f, size), 0.0f, 1.0f);
  }

  void load_staff_gallery() {
    if (!staff_reid_requested_) {
      g_message("globalidentity: staff Re-ID filter disabled (STAFF_REID_FILTER_ENABLE=0)");
      return;
    }
    std::ifstream input(staff_reid_gallery_path_);
    if (!input.is_open()) {
      g_warning("globalidentity: staff Re-ID gallery is not available: %s", staff_reid_gallery_path_.c_str());
      return;
    }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (line.empty() || line.front() == '#') continue;
      std::istringstream values(line);
      StaffGalleryEntry entry;
      std::string staff_label;
      if (!(values >> staff_label >> entry.source >> entry.view_bucket >> entry.quality)) {
        g_warning("globalidentity: ignoring malformed staff gallery line %zu", line_number);
        continue;
      }
      entry.vector.resize(256);
      bool valid = true;
      for (float &item : entry.vector) {
        if (!(values >> item)) { valid = false; break; }
      }
      if (!valid || !normalize(entry.vector) || !std::isfinite(entry.quality) || entry.quality < 0.0f) {
        g_warning("globalidentity: ignoring invalid staff gallery vector on line %zu", line_number);
        continue;
      }
      staff_reid_gallery_.push_back(std::move(entry));
    }
    staff_reid_enabled_ = !staff_reid_gallery_.empty();
    if (staff_reid_enabled_) {
      g_message("globalidentity: staff Re-ID gallery enabled entries=%zu threshold=%.3f votes=%d/%d",
                staff_reid_gallery_.size(), staff_reid_threshold_, staff_reid_min_votes_,
                staff_reid_vote_window_);
    } else {
      g_warning("globalidentity: staff Re-ID filter requested but gallery has no valid entries");
    }
  }

  float staff_reid_score(const std::vector<float> &embedding) const {
    float best = -1.0f;
    for (const auto &entry : staff_reid_gallery_) {
      best = std::max(best, cosine(embedding, entry.vector));
    }
    return best;
  }

  bool update_staff_reid_state(LocalState &state, const Observation &observation,
                               const std::vector<float> &embedding, float q) {
    if (!staff_reid_enabled_) return false;
    const float score = staff_reid_score(embedding);
    const bool positive = q >= staff_reid_min_quality_ && score >= staff_reid_threshold_;
    state.staff_reid_votes.push_back(positive ? 1 : 0);
    while (static_cast<int>(state.staff_reid_votes.size()) > staff_reid_vote_window_)
      state.staff_reid_votes.pop_front();
    int positives = 0;
    for (int vote : state.staff_reid_votes) positives += vote;
    if (!state.staff_reid_confirmed && positives >= staff_reid_min_votes_) {
      state.staff_reid_confirmed = true;
      std::ostringstream out;
      out << "{\"type\":\"STAFF_REID_CONFIRMED\",\"timestamp\":" << observation.timestamp
          << ",\"source_id\":" << observation.key.source
          << ",\"local_object_id\":" << observation.key.object
          << ",\"score\":" << score << ",\"quality_score\":" << q
          << ",\"positive_votes\":" << positives
          << ",\"vote_window\":" << state.staff_reid_votes.size() << "}";
      write(out.str());
    }
    return state.staff_reid_confirmed;
  }

  std::pair<float, float> similarity(const std::deque<PendingEmbedding> &probes,
                                     const Person &person) const {
    std::vector<std::pair<float, int>> values;
    for (const auto &probe : probes) {
      std::unordered_set<int> used_buckets;
      std::vector<std::pair<float, int>> probe_values;
      for (const auto &entry : person.gallery)
        probe_values.emplace_back(cosine(probe.vector, entry.vector), entry.view_bucket);
      std::sort(probe_values.begin(), probe_values.end(),
                [](const auto &left, const auto &right) { return left.first > right.first; });
      // Prevent many nearly identical frames from one view dominating the
      // score. Prefer the strongest embedding from distinct view buckets.
      for (const auto &item : probe_values) {
        if (used_buckets.insert(item.second).second) values.push_back(item);
        if (used_buckets.size() >= 3) break;
      }
    }
    if (values.empty()) return {0.0f, 0.0f};
    std::sort(values.begin(), values.end(),
              [](const auto &left, const auto &right) { return left.first > right.first; });
    const std::size_t count = std::min<std::size_t>(3, values.size());
    float mean = 0.0f;
    for (std::size_t index = 0; index < count; ++index) mean += values[index].first;
    mean /= static_cast<float>(count);
    return {0.70f * values[0].first + 0.30f * mean, values[0].first};
  }

  void append_gallery(Person &person, const Observation &observation, const std::vector<float> &vector,
                      float q, int bucket_override = -1) {
    const int bucket = bucket_override >= 0 ? bucket_override : view_bucket(observation);
    for (const auto &entry : person.gallery) {
      if (entry.view_bucket == bucket && cosine(vector, entry.vector) >= 0.995f) return;
    }
    GalleryEntry entry{vector, observation.key.source, observation.seconds, q, bucket};
    if (person.gallery.size() < static_cast<std::size_t>(gallery_size_)) {
      person.gallery.push_back(std::move(entry));
      return;
    }
    // Preserve a new camera/scale bucket by evicting the weakest member of an
    // over-represented bucket first. This retains front/side/back and near/far
    // views without needing a brittle front/back classifier.
    std::unordered_map<int, int> bucket_counts;
    for (const auto &item : person.gallery) ++bucket_counts[item.view_bucket];
    auto weakest = person.gallery.end();
    for (auto iterator = person.gallery.begin(); iterator != person.gallery.end(); ++iterator) {
      if (bucket_counts[iterator->view_bucket] <= 1) continue;
      if (weakest == person.gallery.end() || iterator->quality < weakest->quality) weakest = iterator;
    }
    if (weakest == person.gallery.end()) {
      weakest = std::min_element(person.gallery.begin(), person.gallery.end(),
        [](const GalleryEntry &left, const GalleryEntry &right) { return left.quality < right.quality; });
    }
    if (weakest != person.gallery.end() && (bucket_counts.count(bucket) == 0 || weakest->quality < q)) {
      *weakest = std::move(entry);
    }
  }

  void purge(double now) {
    for (auto iterator = persons_.begin(); iterator != persons_.end();) {
      if (now - iterator->second.last_seen > max_gap_) {
        emit_expired(iterator->second.id, now);
        iterator = persons_.erase(iterator);
      } else ++iterator;
    }
    for (auto iterator = local_tracks_.begin(); iterator != local_tracks_.end();) {
      if (now - iterator->second.last_seen > max_gap_) iterator = local_tracks_.erase(iterator);
      else ++iterator;
    }
    for (auto iterator = active_claims_.begin(); iterator != active_claims_.end();) {
      if (now - iterator->second.last_seen > active_claim_gap_) iterator = active_claims_.erase(iterator);
      else ++iterator;
    }
  }

  bool claim_conflict(int id, const Key &key, double now, int &conflicts) const {
    bool conflict = false;
    for (const auto &item : active_claims_) {
      const Key &other_key = item.first;
      const ActiveClaim &claim = item.second;
      if (claim.id != id || other_key == key) continue;
      const double gap = now - claim.last_seen;
      if (gap < 0.0 || gap > active_claim_gap_) continue;
      const bool same_camera = other_key.source == key.source;
      const bool allowed_simultaneous_pair =
          simultaneous_pairs_.count(camera_pair_key(other_key.source, key.source)) != 0;
      if ((same_camera && gap <= same_camera_gap_) ||
          (!same_camera && gap <= simultaneous_gap_ && !allowed_simultaneous_pair)) {
        ++conflicts;
        conflict = true;
      }
    }
    return conflict;
  }

  void update_claim(int id, const Key &key, double now, float quality) {
    active_claims_[key] = ActiveClaim{id, now, quality};
  }

  void write(const std::string &line) {
    if (output_.is_open()) {
      output_ << line << '\n';
      output_.flush();
    }
  }

  void emit_expired(int id, double now) {
    std::ostringstream out;
    out << "{\"type\":\"GLOBAL_ID_EXPIRED\",\"timestamp\":" << now << ",\"global_id\":" << id << "}";
    write(out.str());
  }

  void emit_merge(int kept_id, int removed_id, double now, float score) {
    std::ostringstream out;
    out << "{\"type\":\"GLOBAL_ID_MERGED\",\"timestamp\":" << now
        << ",\"kept_global_id\":" << kept_id
        << ",\"removed_global_id\":" << removed_id
        << ",\"score\":" << score << "}";
    write(out.str());
  }

  void merge_people(int kept_id, int removed_id, double now, float score) {
    if (kept_id == removed_id) return;
    auto kept_iterator = persons_.find(kept_id);
    auto removed_iterator = persons_.find(removed_id);
    if (kept_iterator == persons_.end() || removed_iterator == persons_.end()) return;

    Person &kept = kept_iterator->second;
    Person &removed = removed_iterator->second;
    const double removed_last_seen = removed.last_seen;

    for (const auto &track : removed.active_tracks) {
      auto current = kept.active_tracks.find(track.first);
      if (current == kept.active_tracks.end() || current->second.second < track.second.second)
        kept.active_tracks[track.first] = track.second;
    }
    for (const auto &item : removed.camera_world) {
      auto current = kept.camera_world.find(item.first);
      if (current == kept.camera_world.end() || current->second.seconds < item.second.seconds)
        kept.camera_world[item.first] = item.second;
    }
    for (const auto &entry : removed.gallery) kept.gallery.push_back(entry);
    std::sort(kept.gallery.begin(), kept.gallery.end(),
              [](const GalleryEntry &left, const GalleryEntry &right) {
                return left.quality > right.quality;
              });
    while (kept.gallery.size() > static_cast<std::size_t>(gallery_size_)) kept.gallery.pop_back();
    if (removed_last_seen > kept.last_seen) {
      kept.last_seen = removed.last_seen;
      kept.last_source = removed.last_source;
      kept.world_valid = removed.world_valid;
      kept.world_x = removed.world_x;
      kept.world_y = removed.world_y;
    }
    kept.lifecycle = LifecycleState::ACTIVE_VERIFIED;

    for (auto &item : local_tracks_)
      if (item.second.assigned == removed_id) item.second.assigned = kept_id;
    for (auto &item : display_ids_)
      if (item.second.id == removed_id) item.second.id = kept_id;
    for (auto &item : active_claims_)
      if (item.second.id == removed_id) item.second.id = kept_id;

    persons_.erase(removed_iterator);
    emit_merge(kept_id, removed_id, now, score);
  }

  void update_world(Person &person, const Observation &observation) {
    person.camera_world[observation.key.source] = Person::CameraWorld{
        observation.world_valid, observation.world_x, observation.world_y, observation.seconds};
    if (observation.world_valid) {
      person.world_valid = true;
      person.world_x = observation.world_x;
      person.world_y = observation.world_y;
    }
  }

  int merge_duplicate_for_track(const Observation &observation, const LocalState &state,
                                int assigned_id, double now) {
    if (observation.embedding.empty()) return assigned_id;
    auto assigned_iterator = persons_.find(assigned_id);
    if (assigned_iterator == persons_.end()) return assigned_id;
    Person &assigned = assigned_iterator->second;

    const double max_gap = env_double("GLOBAL_REID_ID_MERGE_MAX_GAP_SECONDS", 3.0);
    const float merge_threshold = static_cast<float>(
        env_double("GLOBAL_REID_ID_MERGE_THRESHOLD", 0.80));
    const float merge_margin = static_cast<float>(
        env_double("GLOBAL_REID_ID_MERGE_MARGIN", 0.03));
    const float assigned_score = calibration_.combined_score(
        observation, assigned, similarity(state.pending, assigned).first);
    int best_duplicate = 0;
    float best_score = merge_threshold;

    for (auto &item : persons_) {
      Person &candidate = item.second;
      if (candidate.id == assigned_id || candidate.last_source == observation.key.source) continue;
      if (std::abs(now - candidate.last_seen) > max_gap) continue;
      if (!candidate.world_valid || !observation.world_valid) continue;
      if (!calibration_.allows(observation, candidate)) continue;
      const bool candidate_has_other_camera = std::any_of(
          candidate.active_tracks.begin(), candidate.active_tracks.end(),
          [&observation](const auto &track) { return track.first != observation.key.source; });
      if (!candidate_has_other_camera) continue;
      const float score = calibration_.combined_score(
          observation, candidate, similarity(state.pending, candidate).first);
      if (score > best_score && score >= assigned_score + merge_margin) {
        best_score = score;
        best_duplicate = candidate.id;
      }
    }
    if (!best_duplicate) return assigned_id;
    const int kept_id = std::min(assigned_id, best_duplicate);
    const int removed_id = std::max(assigned_id, best_duplicate);
    merge_people(kept_id, removed_id, now, best_score);
    return kept_id;
  }

  void emit_assignment(const Observation &observation, const char *reason, const char *mode,
                       int assigned, int candidates, float best, float second, float margin, float q,
                       LifecycleState lifecycle, std::size_t gallery_size = 0,
                       int conflict_rejections = 0, int confirmation_hits = 0) {
    std::ostringstream out;
    out << "{\"type\":\"assignment\",\"timestamp\":" << observation.timestamp
        << ",\"source_id\":\"" << observation.key.source << "\",\"local_object_id\":" << observation.key.object
        << ",\"assigned_global_id\":" << (assigned ? std::to_string(assigned) : "null")
        << ",\"assignment_reason\":" << json_string(reason)
        << ",\"camera_mode\":" << json_string(mode)
        << ",\"candidate_count\":" << candidates
        << ",\"best_similarity\":" << best
        << ",\"second_best_similarity\":" << second
        << ",\"margin\":" << margin
        << ",\"quality_score\":" << q
        << ",\"view_bucket\":" << view_bucket(observation)
        << ",\"lifecycle_state\":" << json_string(lifecycle_name(lifecycle))
        << ",\"gallery_size\":" << gallery_size
        << ",\"conflict_rejections\":" << conflict_rejections
        << ",\"confirmation_hits\":" << confirmation_hits << "}";
    write(out.str());
  }

  void process(Observation observation) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    purge(observation.seconds);
    observation.world_valid = calibration_.project(
        observation.key.source, observation.left, observation.top, observation.width, observation.height,
        observation.world_x, observation.world_y);
    const Key key = observation.key;
    LocalState &state = local_tracks_[key];
    state.last_seen = observation.seconds;
    state.last_source = key.source;
    const bool birth_camera = identity_sources_.count(key.source) != 0;
    if (!state.assigned) state.lifecycle = birth_camera ? LifecycleState::ENTRY_PENDING : LifecycleState::ENTRY_PENDING;
    const float q = quality(observation);
    std::vector<float> normalized = observation.embedding;
    if (q <= 0.0f || !normalize(normalized)) {
      if (state.assigned) {
        auto person = persons_.find(state.assigned);
        if (person != persons_.end()) {
          int conflicts = 0;
          if (claim_conflict(person->second.id, key, observation.seconds, conflicts)) {
            active_claims_.erase(key);
            display_ids_.erase(key);
            state.assigned = 0;
            state.candidate_id = 0;
            state.candidate_hits = 0;
            state.candidate_last_seen = -1.0;
            state.pending.clear();
            state.good = 0;
            state.lifecycle = LifecycleState::ENTRY_PENDING;
            emit_assignment(observation, "GLOBAL_ID_CONFLICT_REJECTED", "assigned", 0, 0,
                            0.0f, 0.0f, 0.0f, q, state.lifecycle, 0, conflicts, 0);
            return;
          }
          person->second.last_seen = observation.seconds;
          update_world(person->second, observation);
          update_claim(person->second.id, key, observation.seconds, 0.0f);
        }
      }
      emit_assignment(observation, "LOW_QUALITY_OR_INVALID_EMBEDDING",
                      birth_camera ? "identity_birth" : "observer_only",
                      state.assigned, 0, 0.0f, 0.0f, 0.0f, q, state.lifecycle,
                      state.assigned ? persons_.at(state.assigned).gallery.size() : 0);
      return;
    }
    state.good++;
    state.pending.push_back(PendingEmbedding{normalized, key.source, observation.seconds, q, view_bucket(observation)});
    while (state.pending.size() > 8) state.pending.pop_front();
    if (update_staff_reid_state(state, observation, normalized, q)) {
      if (state.assigned) {
        auto person = persons_.find(state.assigned);
        if (person != persons_.end()) {
          auto active = person->second.active_tracks.find(key.source);
          if (active != person->second.active_tracks.end() && active->second.first == key.object)
            person->second.active_tracks.erase(active);
        }
      }
      state.assigned = 0;
      state.candidate_id = 0;
      state.candidate_hits = 0;
      state.candidate_last_seen = -1.0;
      state.pending.clear();
      state.good = 0;
      active_claims_.erase(key);
      display_ids_.erase(key);
      emit_assignment(observation, "STAFF_REID_CONFIRMED", "staff_filtered", 0, 0,
                      staff_reid_score(normalized), 0.0f, 0.0f, q,
                      LifecycleState::CLOSED, 0, 0, 0);
      return;
    }

    if (state.assigned) {
      auto person_iterator = persons_.find(state.assigned);
      if (person_iterator == persons_.end()) {
        state.assigned = 0;
        state.candidate_id = 0;
        state.candidate_hits = 0;
        state.pending.clear();
        state.good = 0;
      } else {
        const int assigned_id = person_iterator->second.id;
        const int merged_id = merge_duplicate_for_track(observation, state, assigned_id, observation.seconds);
        if (merged_id != assigned_id) {
          state.assigned = merged_id;
          person_iterator = persons_.find(merged_id);
          if (person_iterator == persons_.end()) return;
        }
        Person &current_person = person_iterator->second;
        int conflicts = 0;
        if (claim_conflict(current_person.id, key, observation.seconds, conflicts)) {
          active_claims_.erase(key);
          display_ids_.erase(key);
          state.assigned = 0;
          state.candidate_id = 0;
          state.candidate_hits = 0;
          state.pending.clear();
          state.good = 0;
          state.lifecycle = LifecycleState::ENTRY_PENDING;
          emit_assignment(observation, "GLOBAL_ID_CONFLICT_REJECTED", "assigned", 0, 0,
                          0.0f, 0.0f, 0.0f, q, state.lifecycle, 0, conflicts, 0);
          return;
        }
        const guint previous_source = current_person.last_source;
        if (state.last_revalidated < 0.0 ||
            observation.seconds - state.last_revalidated >= revalidate_interval_) {
          const auto validation = similarity(state.pending, current_person);
          state.last_revalidated = observation.seconds;
          if (validation.first < threshold_) {
            ++state.revalidation_failures;
          } else {
            state.revalidation_failures = 0;
          }
          if (state.revalidation_failures >= revalidate_failures_required_) {
            active_claims_.erase(key);
            display_ids_.erase(key);
            state.assigned = 0;
            state.candidate_id = 0;
            state.candidate_hits = 0;
            state.candidate_last_seen = -1.0;
            state.pending.clear();
            state.good = 0;
            state.revalidation_failures = 0;
            state.lifecycle = LifecycleState::ENTRY_PENDING;
            emit_assignment(observation, "REID_REVALIDATION_REJECTED", "assigned", 0, 1,
                            validation.first, 0.0f, validation.first, q, state.lifecycle,
                            0, 0, 0);
            return;
          }
        }
        current_person.last_seen = observation.seconds;
        current_person.last_source = key.source;
        current_person.active_tracks[key.source] = {key.object, observation.seconds};
        update_world(current_person, observation);
        update_claim(current_person.id, key, observation.seconds, q);
        if (state.last_gallery_update < 0.0 || observation.seconds - state.last_gallery_update >= gallery_interval_) {
          append_gallery(current_person, observation, normalized, q);
          state.last_gallery_update = observation.seconds;
        }
        // A return to CH7 after an interior observation is logged as an exit
        // candidate. It is deliberately not closed here: direction/gate points
        // must be configured before an identity is permanently closed.
        if (birth_camera && previous_source != key.source && previous_source != 0) {
          state.lifecycle = LifecycleState::EXIT_PENDING;
        } else {
          state.lifecycle = LifecycleState::ACTIVE_VERIFIED;
        }
        current_person.lifecycle = state.lifecycle;
        display_ids_[key] = {current_person.id, observation.seconds};
        emit_assignment(observation, "SAME_LOCAL_TRACK", "assigned", current_person.id, 0, 0.0f, 0.0f, 0.0f, q,
                        state.lifecycle, current_person.gallery.size(), 0, state.candidate_hits);
        return;
      }
    }
    if (state.good < min_good_) return;

    int best_id = 0;
    float best = -1.0f, second = -1.0f;
    int candidate_count = 0;
    int conflict_rejections = 0;
    for (auto &item : persons_) {
      Person &person = item.second;
      const double gap = observation.seconds - person.last_seen;
      if (gap < 0.0 || gap > max_gap_) continue;
      if (claim_conflict(person.id, key, observation.seconds, conflict_rejections)) continue;
      if (person.last_source != key.source && !calibration_.has_pair_geometry(observation, person)) continue;
      if (!calibration_.allows(observation, person)) continue;
      const float appearance = similarity(state.pending, person).first;
      const float score = calibration_.combined_score(observation, person, appearance);
      candidate_count++;
      if (score > best) { second = best; best = score; best_id = person.id; }
      else if (score > second) second = score;
    }
    const float margin = best - std::max(0.0f, second);
    const bool cross_camera = best_id && key.source != persons_.at(best_id).last_source;
    const float required_threshold = cross_camera ? cross_match_threshold_ : threshold_;
    const float required_margin = cross_camera ? cross_min_margin_ : min_margin_;
    const int required_confirmations = cross_camera ? cross_match_confirmations_ : match_confirmations_;
    const bool match = best_id && best >= required_threshold && margin >= required_margin &&
                       (!cross_camera || q >= cross_min_quality_);
    const char *mode = birth_camera ? "identity_birth" : "observer_only";
    if (match) {
      if (state.candidate_id == best_id && state.candidate_last_seen >= 0.0 &&
          observation.seconds - state.candidate_last_seen <= candidate_window_) {
        ++state.candidate_hits;
      } else {
        state.candidate_id = best_id;
        state.candidate_hits = 1;
      }
      state.candidate_last_seen = observation.seconds;
      if (state.candidate_hits < required_confirmations) {
        emit_assignment(observation, "UNASSIGNED_MATCH_CONFIRMATION_PENDING", mode, 0, candidate_count,
                        best, second, margin, q, state.lifecycle, 0, conflict_rejections,
                        state.candidate_hits);
        return;
      }
      state.assigned = best_id;
      Person &person = persons_.at(best_id);
      state.lifecycle = LifecycleState::ACTIVE_VERIFIED;
      person.lifecycle = state.lifecycle;
      person.last_seen = observation.seconds;
      person.last_source = key.source;
      person.active_tracks[key.source] = {key.object, observation.seconds};
      update_world(person, observation);
      person.world_valid = observation.world_valid;
      person.world_x = observation.world_x;
      person.world_y = observation.world_y;
      update_claim(best_id, key, observation.seconds, q);
      for (const auto &pending : state.pending) {
        Observation pending_observation = observation;
        pending_observation.key.source = pending.source;
        pending_observation.seconds = pending.seconds;
        append_gallery(person, pending_observation, pending.vector, pending.quality, pending.view_bucket);
      }
      display_ids_[key] = {best_id, observation.seconds};
      emit_assignment(observation, "CROSS_CAMERA_REID_MATCH", mode, best_id, candidate_count,
                      best, second, margin, q, state.lifecycle, person.gallery.size(),
                      conflict_rejections, state.candidate_hits);
      state.candidate_id = 0;
      state.candidate_hits = 0;
      state.candidate_last_seen = -1.0;
      return;
    }
    if (best_id && best >= required_threshold && margin < required_margin) {
      state.candidate_id = 0;
      state.candidate_hits = 0;
      state.candidate_last_seen = -1.0;
      emit_assignment(observation, "UNASSIGNED_LOW_MARGIN", mode, 0, candidate_count,
                      best, second, margin, q, state.lifecycle, 0, conflict_rejections, state.candidate_hits);
      return;
    }
    if (state.good < min_good_) return;
    if (birth_camera) {
      const int id = next_id_++;
      Person person;
      person.id = id;
      person.lifecycle = LifecycleState::ACTIVE_VERIFIED;
      person.last_seen = observation.seconds;
      person.last_source = key.source;
      person.active_tracks[key.source] = {key.object, observation.seconds};
      update_world(person, observation);
      person.world_valid = observation.world_valid;
      person.world_x = observation.world_x;
      person.world_y = observation.world_y;
      update_claim(id, key, observation.seconds, q);
      for (const auto &pending : state.pending) {
        Observation pending_observation = observation;
        pending_observation.key.source = pending.source;
        pending_observation.seconds = pending.seconds;
        append_gallery(person, pending_observation, pending.vector, pending.quality, pending.view_bucket);
      }
      persons_[id] = std::move(person);
      state.assigned = id;
      state.lifecycle = LifecycleState::ACTIVE_VERIFIED;
      display_ids_[key] = {id, observation.seconds};
      emit_assignment(observation, "NEW_GLOBAL_PERSON_AFTER_CONFIRMATION", mode, id, candidate_count,
                      best, second, margin, q, state.lifecycle, persons_.at(id).gallery.size(),
                      conflict_rejections, state.candidate_hits);
      state.candidate_id = 0;
      state.candidate_hits = 0;
      state.candidate_last_seen = -1.0;
    } else {
      const char *reason = best > 0.0f ? "UNASSIGNED_LOW_SIMILARITY" : "UNASSIGNED_NO_MATCH";
      state.candidate_id = 0;
      state.candidate_hits = 0;
      state.candidate_last_seen = -1.0;
      emit_assignment(observation, reason, mode, 0, candidate_count, best, second, margin, q,
                      state.lifecycle, 0, conflict_rejections, state.candidate_hits);
    }
  }

  const float threshold_, min_margin_;
  const double max_gap_;
  const int gallery_size_;
  const double gallery_interval_;
  const int min_good_, match_confirmations_;
  const float cross_match_threshold_, cross_min_margin_, cross_min_quality_;
  const int cross_match_confirmations_;
  const double candidate_window_;
  const int sample_every_;
  const float same_camera_gap_, active_claim_gap_, simultaneous_gap_;
  const double revalidate_interval_;
  const int revalidate_failures_required_;
  const float min_detector_, min_tracker_;
  const bool staff_reid_requested_;
  const float staff_reid_threshold_, staff_reid_min_quality_;
  const int staff_reid_min_votes_, staff_reid_vote_window_;
  const std::string staff_reid_gallery_path_;
  const std::string enrollment_output_path_;
  const std::string output_path_;
  const std::unordered_set<guint> identity_sources_;
  const std::unordered_set<std::string> simultaneous_pairs_;
  FloorCalibration calibration_;
  const int queue_limit_;
  std::ofstream output_;
  std::thread worker_;
  std::mutex queue_mutex_, state_mutex_;
  std::condition_variable queue_condition_;
  std::deque<Observation> queue_;
  std::unordered_map<Key, guint64, KeyHash> last_sampled_;
  std::unordered_map<Key, LocalState, KeyHash> local_tracks_;
  std::unordered_map<Key, DisplayId, KeyHash> display_ids_;
  std::unordered_map<Key, ActiveClaim, KeyHash> active_claims_;
  std::unordered_map<int, Person> persons_;
  std::vector<StaffGalleryEntry> staff_reid_gallery_;
  bool staff_reid_enabled_ = false;
  std::mutex enrollment_mutex_;
  std::atomic<bool> stop_;
  std::atomic<uint64_t> dropped_{0};
  int next_id_;
};

typedef struct _GstGlobalIdentity {
  GstBaseTransform parent;
  IdentityBackend *backend;
} GstGlobalIdentity;

typedef struct _GstGlobalIdentityClass {
  GstBaseTransformClass parent_class;
} GstGlobalIdentityClass;

#define GST_TYPE_GLOBAL_IDENTITY (gst_global_identity_get_type())
#define GST_GLOBAL_IDENTITY(obj) ((GstGlobalIdentity *)(obj))
G_DEFINE_TYPE(GstGlobalIdentity, gst_global_identity, GST_TYPE_BASE_TRANSFORM)

static bool find_embedding(NvDsObjectMeta *object, std::vector<float> &embedding) {
  for (NvDsMetaList *item = object->obj_user_meta_list; item; item = item->next) {
    auto *meta = static_cast<NvDsUserMeta *>(item->data);
    if (!meta || meta->base_meta.meta_type != NVDS_TRACKER_OBJ_REID_META || !meta->user_meta_data) continue;
    auto *reid = static_cast<NvDsObjReid *>(meta->user_meta_data);
    if (reid->ptr_host && reid->featureSize == 256) {
      embedding.assign(reid->ptr_host, reid->ptr_host + reid->featureSize);
      return true;
    }
  }
  return false;
}

static GstFlowReturn transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = GST_GLOBAL_IDENTITY(base);
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch || !self->backend) return GST_FLOW_OK;
  std::vector<Observation> enrollment_observations;
  for (NvDsMetaList *frame_item = batch->frame_meta_list; frame_item; frame_item = frame_item->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(frame_item->data);
    if (!frame) continue;
    const guint64 timestamp = frame->buf_pts;
    const double seconds = timestamp_seconds(timestamp);
    std::vector<NvDsObjectMeta *> staff_objects;
    for (NvDsMetaList *object_item = frame->obj_meta_list; object_item; object_item = object_item->next) {
      auto *object = static_cast<NvDsObjectMeta *>(object_item->data);
      if (!object || object->class_id != 0) continue;
      const Key key{frame->source_id, object->object_id};
      const bool staff = self->backend->is_staff(key, seconds);
      int assigned = 0;
      if (staff) {
        g_free(object->text_params.display_text);
        object->text_params.display_text = g_strdup("");
        staff_objects.push_back(object);
      } else if (self->backend->lookup(key, seconds, assigned)) {
        const std::string text = "G" + std::to_string(assigned);
        g_free(object->text_params.display_text);
        object->text_params.display_text = g_strdup(text.c_str());
      } else if (env_bool("GLOBAL_ID_OSD", true)) {
        g_free(object->text_params.display_text);
        object->text_params.display_text = g_strdup("");
      }
      Observation observation;
      observation.key = key;
      observation.frame = frame->frame_num;
      observation.timestamp = timestamp;
      observation.seconds = seconds;
      observation.left = object->rect_params.left;
      observation.top = object->rect_params.top;
      observation.width = object->rect_params.width;
      observation.height = object->rect_params.height;
      observation.detector_confidence = object->confidence;
      observation.tracker_confidence = object->tracker_confidence;
      const bool sample_due = self->backend->should_sample(key, frame->frame_num);
      const bool has_embedding = find_embedding(object, observation.embedding);
      if (has_embedding) {
        std::vector<float> enrollment_embedding = observation.embedding;
        if (normalize(enrollment_embedding)) {
          observation.embedding = enrollment_embedding;
        } else {
          observation.embedding.clear();
        }
      }
      // The enrollment snapshot is also the operator's diagnostic view. Keep
      // every current tracked person in it, even when NvDCF has not attached
      // a Re-ID vector on this frame. Only embeddings are submitted to the
      // asynchronous Global-ID backend, preserving the existing sampling and
      // matching behavior.
      enrollment_observations.push_back(observation);
      if (observation.embedding.empty()) continue;
      if (!sample_due) continue;
      self->backend->submit(std::move(observation));
    }
    // This is downstream display filtering only.  NvDCF and NvDsObjReid have
    // already completed for this buffer, and the local track state remains in
    // the tracker; only confirmed staff metadata is removed from downstream
    // OSD/output branches.
    for (NvDsObjectMeta *staff_object : staff_objects)
      nvds_remove_obj_meta_from_frame(frame, staff_object);
  }
  self->backend->publish_enrollment_snapshot(enrollment_observations);
  return GST_FLOW_OK;
}

static void finalize(GObject *object) {
  auto *self = GST_GLOBAL_IDENTITY(object);
  delete self->backend;
  self->backend = nullptr;
  G_OBJECT_CLASS(gst_global_identity_parent_class)->finalize(object);
}

static void gst_global_identity_class_init(GstGlobalIdentityClass *klass) {
  auto *base = GST_BASE_TRANSFORM_CLASS(klass);
  base->transform_ip = transform_ip;
  auto *gobject = G_OBJECT_CLASS(klass);
  gobject->finalize = finalize;
  auto *element = GST_ELEMENT_CLASS(klass);
  static GstStaticPadTemplate sink_template =
      GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  static GstStaticPadTemplate src_template =
      GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  gst_element_class_add_static_pad_template(element, &sink_template);
  gst_element_class_add_static_pad_template(element, &src_template);
  gst_element_class_set_static_metadata(element, "Async Global Identity", "Filter/Metadata",
                                        "NVIDIA Re-ID Global-ID worker outside the DeepStream probe path", PACKAGE);
}

static void gst_global_identity_init(GstGlobalIdentity *self) {
  self->backend = new IdentityBackend();
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

}  // namespace

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "globalidentity", GST_RANK_NONE, GST_TYPE_GLOBAL_IDENTITY);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, globalidentity, "Async Global Identity",
                  plugin_init, "1.0", "Proprietary", PACKAGE, "https://nvidia.com")
