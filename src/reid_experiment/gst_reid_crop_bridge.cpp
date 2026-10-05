// SPDX-License-Identifier: Apache-2.0
// Isolated Phase 1B bridge. It never changes NvDsObjectMeta or video pixels.

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include <nvdsmeta.h>
#include <nvdsinfer.h>

#ifndef PACKAGE
#define PACKAGE "datamine-cv"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x52454944; // "REID"
constexpr uint32_t kVersion = 1;
constexpr uint32_t kBgr8 = 1;

#pragma pack(push, 1)
struct PacketHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t payload_bytes;
  uint32_t source_id;
  uint64_t local_track_id;
  uint64_t frame_number;
  uint64_t timestamp;
  float left;
  float top;
  float width;
  float height;
  float detector_confidence;
  float tracker_confidence;
  uint32_t crop_width;
  uint32_t crop_height;
  uint32_t format;
};
#pragma pack(pop)

struct Packet { std::vector<uint8_t> bytes; };

static bool env_bool(const char *name, bool fallback) {
  const char *value = g_getenv(name);
  if (!value) return fallback;
  return g_ascii_strcasecmp(value, "1") == 0 || g_ascii_strcasecmp(value, "true") == 0 ||
         g_ascii_strcasecmp(value, "yes") == 0;
}

static int env_int(const char *name, int fallback) {
  const char *value = g_getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  return end && *end == '\0' ? static_cast<int>(parsed) : fallback;
}

static float env_float(const char *name, float fallback) {
  const char *value = g_getenv(name);
  if (!value) return fallback;
  char *end = nullptr;
  float parsed = std::strtof(value, &end);
  return end && *end == '\0' ? parsed : fallback;
}

class CropBridge {
 public:
  CropBridge()
      : enabled_(env_bool("REID_EXPERIMENT_CROP_BRIDGE", false)),
        interval_(std::max(1, env_int("REID_EXPERIMENT_INTERVAL_FRAMES", 5))),
        min_width_(std::max(1, env_int("REID_EXPERIMENT_MIN_CROP_WIDTH", 32))),
        min_height_(std::max(1, env_int("REID_EXPERIMENT_MIN_CROP_HEIGHT", 64))),
        min_confidence_(env_float("REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE", 0.35f)),
        queue_limit_(static_cast<size_t>(std::max(1, env_int("REID_EXPERIMENT_QUEUE_SIZE", 64)))),
        socket_path_(g_getenv("REID_EXPERIMENT_SOCKET") ?
                         g_getenv("REID_EXPERIMENT_SOCKET") :
                         "/workspace/runs/reid_experiment/crops.sock") {
    if (enabled_) {
      worker_ = std::thread(&CropBridge::worker_loop, this);
      g_message("reid-experiment crop bridge enabled interval=%d queue=%zu socket=%s",
                interval_, queue_limit_, socket_path_.c_str());
    }
  }

  ~CropBridge() {
    stop_.store(true);
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void process(GstBuffer *buffer) {
    if (!enabled_ || !buffer) return;
    const auto invocation = invocations_.fetch_add(1) + 1;
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) { missing_batch_++; return; }
    NvBufSurface *surface = nullptr;
    // NvDsBatchMeta does not own the surface pointer; obtain it from the
    // GstBuffer through the standard DeepStream surface API.
    gpointer surface_data = nullptr;
    if (!gst_buffer_map(buffer, &buffer_map_, GST_MAP_READ)) { map_failed_++; return; }
    surface_data = buffer_map_.data;
    surface = reinterpret_cast<NvBufSurface *>(surface_data);
    for (NvDsMetaList *frame_item = batch->frame_meta_list; frame_item; frame_item = frame_item->next) {
      auto *frame = static_cast<NvDsFrameMeta *>(frame_item->data);
      if (!frame || frame->batch_id < 0) continue;
      frames_seen_++;
      if (surface->numFilled <= static_cast<uint32_t>(frame->batch_id)) { invalid_++; continue; }
      NvBufSurfaceParams &params = surface->surfaceList[frame->batch_id];
      // NV12 needs both luma (plane 0) and interleaved chroma (plane 1).
      // Mapping only plane 0 leaves mappedAddr.addr[1] null and would reject
      // every otherwise-valid crop.
      if (NvBufSurfaceMap(surface, frame->batch_id, -1, NVBUF_MAP_READ) != 0) { map_failed_++; continue; }
      NvBufSurfaceSyncForCpu(surface, frame->batch_id, -1);
      for (NvDsMetaList *object_item = frame->obj_meta_list; object_item; object_item = object_item->next) {
        auto *object = static_cast<NvDsObjectMeta *>(object_item->data);
        objects_seen_++;
        if (!object || object->class_id != 0) { wrong_class_++; continue; }
        persons_seen_++;
        if (object->object_id == UNTRACKED_OBJECT_ID) { untracked_++; continue; }
        if (object->confidence < min_confidence_) { low_confidence_++; continue; }
        if (frame->frame_num % static_cast<guint64>(interval_) != 0) { sampled_out_++; continue; }
        Packet packet;
        if (!make_packet(params, frame, object, packet)) continue;
        submit(std::move(packet));
      }
      NvBufSurfaceUnMap(surface, frame->batch_id, -1);
    }
    gst_buffer_unmap(buffer, &buffer_map_);
    if (invocation % 300 == 0) {
      g_message("reid-experiment bridge frames=%" G_GUINT64_FORMAT " objects=%" G_GUINT64_FORMAT
                " persons=%" G_GUINT64_FORMAT " submitted=%" G_GUINT64_FORMAT
                " map_failed=%" G_GUINT64_FORMAT " class=%" G_GUINT64_FORMAT
                " untracked=%" G_GUINT64_FORMAT " confidence=%" G_GUINT64_FORMAT
                " sampled=%" G_GUINT64_FORMAT " invalid=%" G_GUINT64_FORMAT,
                frames_seen_.load(), objects_seen_.load(), persons_seen_.load(), submitted_.load(),
                map_failed_.load(), wrong_class_.load(), untracked_.load(), low_confidence_.load(),
                sampled_out_.load(), invalid_.load());
    }
  }

  guint64 submitted() const { return submitted_.load(); }
  guint64 processed() const { return processed_.load(); }
  guint64 dropped() const { return dropped_.load(); }
  guint64 invalid() const { return invalid_.load(); }
  guint64 too_small() const { return too_small_.load(); }
  guint64 queue_full() const { return queue_full_.load(); }

 private:
  bool make_packet(NvBufSurfaceParams &params, NvDsFrameMeta *frame,
                   NvDsObjectMeta *object, Packet &packet) {
    const float left_f = object->rect_params.left;
    const float top_f = object->rect_params.top;
    const float right_f = left_f + object->rect_params.width;
    const float bottom_f = top_f + object->rect_params.height;
    const int left = std::clamp(static_cast<int>(std::floor(left_f)), 0, static_cast<int>(params.width));
    const int top = std::clamp(static_cast<int>(std::floor(top_f)), 0, static_cast<int>(params.height));
    const int right = std::clamp(static_cast<int>(std::ceil(right_f)), left, static_cast<int>(params.width));
    const int bottom = std::clamp(static_cast<int>(std::ceil(bottom_f)), top, static_cast<int>(params.height));
    const int width = right - left;
    const int height = bottom - top;
    if (width <= 0 || height <= 0) { invalid_++; return false; }
    if (width < min_width_ || height < min_height_) { too_small_++; return false; }
    // The bridge deliberately supports the RGBA/BGR/RGB surfaces used by the
    // experiment. An NV12 surface is rejected rather than silently producing
    // incorrect crops; the diagnostic counter makes the required caps clear.
    const auto format = params.colorFormat;
    int channels = 0;
    bool rgba = false;
    bool nv12 = format == NVBUF_COLOR_FORMAT_NV12 || format == NVBUF_COLOR_FORMAT_NV12_709 ||
                format == NVBUF_COLOR_FORMAT_NV12_2020 || format == NVBUF_COLOR_FORMAT_NV12_ER;
    if (nv12) {
      channels = 1;
    } else if (format == NVBUF_COLOR_FORMAT_RGBA) {
      channels = 4; rgba = true;
    } else if (format == NVBUF_COLOR_FORMAT_BGR || format == NVBUF_COLOR_FORMAT_RGB) {
      channels = 3;
    } else { invalid_++; return false; }
    auto *base = static_cast<uint8_t *>(params.mappedAddr.addr[0]);
    auto *chroma = static_cast<uint8_t *>(params.mappedAddr.addr[1]);
    if (!base || (nv12 && !chroma)) { invalid_++; return false; }
    packet.bytes.resize(sizeof(PacketHeader) + static_cast<size_t>(width) * height * 3);
    PacketHeader header{ kMagic, kVersion,
      static_cast<uint32_t>(width * height * 3), static_cast<uint32_t>(frame->source_id),
      object->object_id, static_cast<uint64_t>(frame->frame_num), frame->buf_pts,
      left_f, top_f, object->rect_params.width, object->rect_params.height,
      object->confidence, object->tracker_confidence, static_cast<uint32_t>(width),
      static_cast<uint32_t>(height), kBgr8 };
    std::memcpy(packet.bytes.data(), &header, sizeof(header));
    uint8_t *destination = packet.bytes.data() + sizeof(header);
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    for (int y = 0; y < height; ++y) {
      const uint8_t *source = base + static_cast<size_t>(top + y) * params.pitch +
                              static_cast<size_t>(left) * channels;
      for (int x = 0; x < width; ++x) {
        if (nv12) {
          const int pixel_x = left + x;
          const int pixel_y = top + y;
          const uint8_t y_value = source[x];
          const uint8_t *uv = chroma + static_cast<size_t>(pixel_y / 2) * params.pitch +
                              static_cast<size_t>(pixel_x / 2) * 2;
          const float yf = std::max(0.0f, static_cast<float>(y_value) - 16.0f) * 1.164f;
          const float uf = static_cast<float>(uv[0]) - 128.0f;
          const float vf = static_cast<float>(uv[1]) - 128.0f;
          destination[x * 3 + 0] = static_cast<uint8_t>(std::clamp(yf + 2.018f * uf, 0.0f, 255.0f));
          destination[x * 3 + 1] = static_cast<uint8_t>(std::clamp(yf - 0.391f * uf - 0.813f * vf, 0.0f, 255.0f));
          destination[x * 3 + 2] = static_cast<uint8_t>(std::clamp(yf + 1.596f * vf, 0.0f, 255.0f));
        } else if (rgba) {
          destination[x * 3 + 0] = source[x * 4 + 2];
          destination[x * 3 + 1] = source[x * 4 + 1];
          destination[x * 3 + 2] = source[x * 4 + 0];
        } else if (format == NVBUF_COLOR_FORMAT_RGB) {
          destination[x * 3 + 0] = source[x * 3 + 2];
          destination[x * 3 + 1] = source[x * 3 + 1];
          destination[x * 3 + 2] = source[x * 3 + 0];
        } else {
          std::memcpy(destination + x * 3, source + x * 3, 3);
        }
      }
      destination += row_bytes;
    }
    return true;
  }

  void submit(Packet packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= queue_limit_) { queue_.pop_front(); dropped_++; queue_full_++; }
    queue_.push_back(std::move(packet)); submitted_++; condition_.notify_one();
  }

  bool connect_socket() {
    if (socket_fd_ >= 0) return true;
    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) return false;
    sockaddr_un address{}; address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) return false;
    std::strncpy(address.sun_path, socket_path_.c_str(), sizeof(address.sun_path) - 1);
    if (connect(socket_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
      close(socket_fd_); socket_fd_ = -1; return false;
    }
    return true;
  }

  void worker_loop() {
    while (!stop_.load()) {
      Packet packet;
      { std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(100), [&] { return stop_.load() || !queue_.empty(); });
        if (stop_.load()) break;
        if (queue_.empty()) continue;
        packet = std::move(queue_.front()); queue_.pop_front();
      }
      if (!connect_socket()) { dropped_++; std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
      size_t sent = 0;
      while (sent < packet.bytes.size() && !stop_.load()) {
        ssize_t count = send(socket_fd_, packet.bytes.data() + sent, packet.bytes.size() - sent, MSG_NOSIGNAL);
        if (count <= 0) { close(socket_fd_); socket_fd_ = -1; dropped_++; break; }
        sent += static_cast<size_t>(count);
      }
      if (sent == packet.bytes.size()) processed_++;
    }
    if (socket_fd_ >= 0) close(socket_fd_);
  }

  bool enabled_;
  int interval_, min_width_, min_height_;
  float min_confidence_;
  size_t queue_limit_;
  std::string socket_path_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Packet> queue_;
  int socket_fd_ = -1;
  GstMapInfo buffer_map_{};
  std::atomic<guint64> submitted_{0}, processed_{0}, dropped_{0}, invalid_{0}, too_small_{0}, queue_full_{0};
  std::atomic<guint64> invocations_{0}, frames_seen_{0}, objects_seen_{0}, persons_seen_{0},
      missing_batch_{0}, map_failed_{0}, wrong_class_{0}, untracked_{0}, low_confidence_{0}, sampled_out_{0};
};

typedef struct _GstReidCropBridge { GstBaseTransform parent; CropBridge *bridge; } GstReidCropBridge;
typedef struct _GstReidCropBridgeClass { GstBaseTransformClass parent_class; } GstReidCropBridgeClass;
#define GST_TYPE_REID_CROP_BRIDGE (gst_reid_crop_bridge_get_type())
#define GST_REID_CROP_BRIDGE(obj) ((GstReidCropBridge *)(obj))
G_DEFINE_TYPE(GstReidCropBridge, gst_reid_crop_bridge, GST_TYPE_BASE_TRANSFORM)

static GstFlowReturn transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = GST_REID_CROP_BRIDGE(base);
  if (self->bridge) self->bridge->process(buffer);
  return GST_FLOW_OK;
}

static void finalize(GObject *object) {
  auto *self = GST_REID_CROP_BRIDGE(object);
  delete self->bridge; self->bridge = nullptr;
  G_OBJECT_CLASS(gst_reid_crop_bridge_parent_class)->finalize(object);
}

static void gst_reid_crop_bridge_class_init(GstReidCropBridgeClass *klass) {
  auto *element = GST_ELEMENT_CLASS(klass);
  auto *base = GST_BASE_TRANSFORM_CLASS(klass);
  base->transform_ip = transform_ip;
  G_OBJECT_CLASS(klass)->finalize = finalize;
  static GstStaticPadTemplate sink_template =
      GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  static GstStaticPadTemplate src_template =
      GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  gst_element_class_add_static_pad_template(element, &sink_template);
  gst_element_class_add_static_pad_template(element, &src_template);
  gst_element_class_set_static_metadata(element, "Experimental SOLIDER crop bridge", "Filter/Video",
                                         "Non-blocking person crop export for isolated Re-ID", "datamine-cv");
}

static void gst_reid_crop_bridge_init(GstReidCropBridge *self) {
  self->bridge = new CropBridge();
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  // Do not mark this element as passthrough: GstBaseTransform may then skip
  // transform_ip(), which is where selected crops are exported.  The buffer
  // is still forwarded unchanged because transform_ip returns GST_FLOW_OK.
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

} // namespace

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "reidcropbridge", GST_RANK_NONE, GST_TYPE_REID_CROP_BRIDGE);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, reidcropbridge,
                  "Isolated SOLIDER experimental crop bridge", plugin_init, "1.0", "Apache-2.0",
                  "datamine-cv", "https://example.invalid/datamine-cv")
