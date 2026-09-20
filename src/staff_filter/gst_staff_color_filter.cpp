#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <nvbufsurface.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <sys/stat.h>

#ifndef PACKAGE
#define PACKAGE "datamine-cv"
#endif

typedef struct {
  bool valid = false;
  float h0 = 0.0f, h1 = 179.0f;
  float s0 = 0.0f, s1 = 255.0f;
  float v0 = 0.0f, v1 = 255.0f;
  float l0 = 0.0f, l1 = 255.0f;
  float a0 = 0.0f, a1 = 255.0f;
  float b0 = 0.0f, b1 = 255.0f;
} StaffProfile;

typedef struct _GstStaffColorFilter {
  GstBaseTransform parent;
  gchar *profile_path;
  gdouble threshold;
  gboolean enabled;
  time_t profile_mtime;
  StaffProfile profile;
} GstStaffColorFilter;

typedef struct _GstStaffColorFilterClass {
  GstBaseTransformClass parent_class;
} GstStaffColorFilterClass;

#define GST_TYPE_STAFF_COLOR_FILTER (gst_staff_color_filter_get_type())
#define GST_STAFF_COLOR_FILTER(obj) ((GstStaffColorFilter *)(obj))
G_DEFINE_TYPE(GstStaffColorFilter, gst_staff_color_filter, GST_TYPE_BASE_TRANSFORM)

enum {
  PROP_0,
  PROP_PROFILE_PATH,
  PROP_THRESHOLD,
  PROP_ENABLED,
};

static void rgb_to_hsv_lab(unsigned char r, unsigned char g, unsigned char b,
                           float &h, float &s, float &v,
                           float &l8, float &a8, float &b8) {
  float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
  float mx = std::max(rf, std::max(gf, bf));
  float mn = std::min(rf, std::min(gf, bf));
  float d = mx - mn;
  h = 0.0f;
  if (d > 1e-6f) {
    if (mx == rf) h = 60.0f * std::fmod((gf - bf) / d, 6.0f);
    else if (mx == gf) h = 60.0f * ((bf - rf) / d + 2.0f);
    else h = 60.0f * ((rf - gf) / d + 4.0f);
    if (h < 0.0f) h += 360.0f;
  }
  s = mx <= 1e-6f ? 0.0f : d / mx;
  v = mx;
  auto linear = [](float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
  };
  float R = linear(rf), G = linear(gf), B = linear(bf);
  float X = (R * 0.4124564f + G * 0.3575761f + B * 0.1804375f) / 0.95047f;
  float Y = (R * 0.2126729f + G * 0.7151522f + B * 0.0721750f);
  float Z = (R * 0.0193339f + G * 0.1191920f + B * 0.9503041f) / 1.08883f;
  auto f = [](float q) { return q > 0.008856f ? std::cbrt(q) : 7.787f * q + 16.0f / 116.0f; };
  float fx = f(X), fy = f(Y), fz = f(Z);
  float L = 116.0f * fy - 16.0f;
  float A = 500.0f * (fx - fy) + 128.0f;
  float Bc = 200.0f * (fy - fz) + 128.0f;
  h /= 2.0f;
  s *= 255.0f;
  v *= 255.0f;
  l8 = std::clamp(L * 255.0f / 100.0f, 0.0f, 255.0f);
  a8 = std::clamp(A, 0.0f, 255.0f);
  b8 = std::clamp(Bc, 0.0f, 255.0f);
}

static bool in_range(float value, float low, float high) {
  return value >= low && value <= high;
}

static bool read_profile(GstStaffColorFilter *self) {
  if (!self->profile_path || !*self->profile_path) return false;
  struct stat info{};
  if (stat(self->profile_path, &info) != 0 || info.st_mtime == self->profile_mtime) return self->profile.valid;
  try {
    YAML::Node root = YAML::LoadFile(self->profile_path);
    StaffProfile next;
    next.valid = root["enabled"] && root["enabled"].as<bool>();
    auto hsv = root["hsv"], lab = root["lab"];
    if (next.valid && hsv && lab && hsv["lower"] && hsv["upper"] && lab["lower"] && lab["upper"]) {
      auto hl = hsv["lower"], hu = hsv["upper"], ll = lab["lower"], lu = lab["upper"];
      next.h0 = hl[0].as<float>(); next.s0 = hl[1].as<float>(); next.v0 = hl[2].as<float>();
      next.h1 = hu[0].as<float>(); next.s1 = hu[1].as<float>(); next.v1 = hu[2].as<float>();
      next.l0 = ll[0].as<float>(); next.a0 = ll[1].as<float>(); next.b0 = ll[2].as<float>();
      next.l1 = lu[0].as<float>(); next.a1 = lu[1].as<float>(); next.b1 = lu[2].as<float>();
    } else {
      next.valid = false;
    }
    self->profile = next;
    self->profile_mtime = info.st_mtime;
    g_message("staffcolorfilter: profile %s (%s)", self->profile_path, next.valid ? "enabled" : "not ready");
    return next.valid;
  } catch (const std::exception &error) {
    self->profile.valid = false;
    self->profile_mtime = info.st_mtime;
    g_warning("staffcolorfilter: cannot read %s: %s", self->profile_path, error.what());
    return false;
  }
}

static bool read_rgb(const NvBufSurfaceParams &surface, int x, int y,
                     unsigned char &r, unsigned char &g, unsigned char &b) {
  if (!surface.mappedAddr.addr[0]) return false;
  const auto format = surface.colorFormat;
  if (format == NVBUF_COLOR_FORMAT_RGBA || format == NVBUF_COLOR_FORMAT_RGB) {
    const unsigned char *row = static_cast<const unsigned char *>(surface.mappedAddr.addr[0]) + y * surface.pitch;
    const unsigned char *pixel = row + x * (format == NVBUF_COLOR_FORMAT_RGBA ? 4 : 3);
    if (format == NVBUF_COLOR_FORMAT_RGBA) { r = pixel[0]; g = pixel[1]; b = pixel[2]; }
    else { r = pixel[0]; g = pixel[1]; b = pixel[2]; }
    return true;
  }
  const unsigned char *yplane = static_cast<const unsigned char *>(surface.mappedAddr.addr[0]);
  const unsigned char *uvplane = static_cast<const unsigned char *>(surface.mappedAddr.addr[1]);
  if (!uvplane) return false;
  int Y = yplane[y * surface.pitch + x];
  int uv = (y / 2) * surface.pitch + (x & ~1);
  int U = uvplane[uv] - 128;
  int V = uvplane[uv + 1] - 128;
  int R = static_cast<int>(Y + 1.402f * V);
  int G = static_cast<int>(Y - 0.344136f * U - 0.714136f * V);
  int B = static_cast<int>(Y + 1.772f * U);
  r = static_cast<unsigned char>(std::clamp(R, 0, 255));
  g = static_cast<unsigned char>(std::clamp(G, 0, 255));
  b = static_cast<unsigned char>(std::clamp(B, 0, 255));
  return true;
}

static float staff_score(const GstStaffColorFilter *self, const NvBufSurfaceParams &surface,
                         const NvOSD_RectParams &box) {
  if (!self->profile.valid || box.width < 8.0f || box.height < 16.0f) return 0.0f;
  int width = static_cast<int>(surface.width), height = static_cast<int>(surface.height);
  int x0 = std::clamp(static_cast<int>(box.left + box.width * 0.15f), 0, width - 1);
  int x1 = std::clamp(static_cast<int>(box.left + box.width * 0.85f), x0 + 1, width);
  int y0 = std::clamp(static_cast<int>(box.top + box.height * 0.10f), 0, height - 1);
  int y1 = std::clamp(static_cast<int>(box.top + box.height * 0.65f), y0 + 1, height);
  int total = 0, matched = 0, quadrants[4] = {0, 0, 0, 0}, qtotal[4] = {0, 0, 0, 0};
  for (int y = y0; y < y1; y += 4) {
    for (int x = x0; x < x1; x += 4) {
      unsigned char r, g, b;
      if (!read_rgb(surface, x, y, r, g, b)) continue;
      float h, s, v, l, a, lab_b;
      rgb_to_hsv_lab(r, g, b, h, s, v, l, a, lab_b);
      bool ok = in_range(h, self->profile.h0, self->profile.h1) &&
                in_range(s, self->profile.s0, self->profile.s1) &&
                in_range(v, self->profile.v0, self->profile.v1) &&
                in_range(l, self->profile.l0, self->profile.l1) &&
                in_range(a, self->profile.a0, self->profile.a1) &&
                in_range(lab_b, self->profile.b0, self->profile.b1);
      int q = (x >= (x0 + x1) / 2 ? 1 : 0) + (y >= (y0 + y1) / 2 ? 2 : 0);
      total++; qtotal[q]++;
      if (ok) { matched++; quadrants[q]++; }
    }
  }
  if (total < 12) return 0.0f;
  float coverage = static_cast<float>(matched) / total;
  int strong_quadrants = 0;
  for (int q = 0; q < 4; ++q)
    if (qtotal[q] >= 3 && static_cast<float>(quadrants[q]) / qtotal[q] >= 0.35f) strong_quadrants++;
  return strong_quadrants >= 2 ? coverage : coverage * 0.5f;
}

static GstFlowReturn transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = GST_STAFF_COLOR_FILTER(base);
  if (!self->enabled || !read_profile(self)) return GST_FLOW_OK;
  GstMapInfo map{};
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return GST_FLOW_OK;
  auto *surface = reinterpret_cast<NvBufSurface *>(map.data);
  auto *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!surface || !batch) { gst_buffer_unmap(buffer, &map); return GST_FLOW_OK; }
  for (NvDsMetaList *lf = batch->frame_meta_list; lf; lf = lf->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(lf->data);
    if (!frame || frame->batch_id >= surface->batchSize) continue;
    auto &params = surface->surfaceList[frame->batch_id];
    bool mapped = false;
    // CUDA device NVMM (the dGPU production path) cannot be synchronously
    // mapped to a CPU pointer here.  Fail open for that frame instead of
    // repeatedly calling NvBufSurfaceMap and spamming unsupported-map errors.
    if (!params.mappedAddr.addr[0] &&
        surface->memType != NVBUF_MEM_SYSTEM &&
        surface->memType != NVBUF_MEM_CUDA_PINNED) {
      continue;
    }
    if (!params.mappedAddr.addr[0]) {
      if (NvBufSurfaceMap(surface, frame->batch_id, -1, NVBUF_MAP_READ) != 0) continue;
      NvBufSurfaceSyncForCpu(surface, frame->batch_id, -1);
      mapped = true;
    }
    for (NvDsMetaList *lo = frame->obj_meta_list; lo;) {
      NvDsMetaList *next = lo->next;
      auto *object = static_cast<NvDsObjectMeta *>(lo->data);
      if (object && object->class_id == 0 &&
          staff_score(self, params, object->rect_params) >= self->threshold) {
        nvds_remove_obj_meta_from_frame(frame, object);
      }
      lo = next;
    }
    if (mapped) NvBufSurfaceUnMap(surface, frame->batch_id, -1);
  }
  gst_buffer_unmap(buffer, &map);
  return GST_FLOW_OK;
}

static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *) {
  auto *self = GST_STAFF_COLOR_FILTER(object);
  if (id == PROP_PROFILE_PATH) { g_free(self->profile_path); self->profile_path = g_value_dup_string(value); self->profile_mtime = 0; }
  else if (id == PROP_THRESHOLD) self->threshold = g_value_get_double(value);
  else if (id == PROP_ENABLED) self->enabled = g_value_get_boolean(value);
}

static void get_property(GObject *object, guint id, GValue *value, GParamSpec *) {
  auto *self = GST_STAFF_COLOR_FILTER(object);
  if (id == PROP_PROFILE_PATH) g_value_set_string(value, self->profile_path);
  else if (id == PROP_THRESHOLD) g_value_set_double(value, self->threshold);
  else if (id == PROP_ENABLED) g_value_set_boolean(value, self->enabled);
}

static void finalize(GObject *object) {
  auto *self = GST_STAFF_COLOR_FILTER(object);
  g_free(self->profile_path);
  G_OBJECT_CLASS(gst_staff_color_filter_parent_class)->finalize(object);
}

static void gst_staff_color_filter_class_init(GstStaffColorFilterClass *klass) {
  auto *gobject = G_OBJECT_CLASS(klass);
  auto *base = GST_BASE_TRANSFORM_CLASS(klass);
  static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
      "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
      "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass), &sink_template);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass), &src_template);
  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "Staff color filter", "Filter/Metadata",
      "Removes conservatively classified staff detections before NvDCF", "datamine-cv");
  gobject->set_property = set_property;
  gobject->get_property = get_property;
  gobject->finalize = finalize;
  g_object_class_install_property(gobject, PROP_PROFILE_PATH,
      g_param_spec_string("profile-path", "Profile path", "YAML staff color profile", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(gobject, PROP_THRESHOLD,
      g_param_spec_double("staff-color-threshold", "Staff color threshold", "Minimum upper-body score", 0.0, 1.0, 0.62, G_PARAM_READWRITE));
  g_object_class_install_property(gobject, PROP_ENABLED,
      g_param_spec_boolean("enabled", "Enabled", "Enable staff removal", TRUE, G_PARAM_READWRITE));
  base->transform_ip = transform_ip;
}

static void gst_staff_color_filter_init(GstStaffColorFilter *self) {
  self->threshold = 0.62;
  self->enabled = TRUE;
  self->profile_mtime = 0;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "staffcolorfilter", GST_RANK_NONE, GST_TYPE_STAFF_COLOR_FILTER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, staffcolorfilter,
                  "Conservative staff uniform pre-tracker filter", plugin_init,
                  "1.0", "Proprietary", PACKAGE, "https://datamine-cv.local")
