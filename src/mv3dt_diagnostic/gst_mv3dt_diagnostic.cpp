#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#ifndef PACKAGE
#define PACKAGE "datamine-cv"
#endif

typedef struct _GstMv3dtDiagnostic {
  GstBaseTransform parent;
  gchar *output_path;
  FILE *output;
  GMutex lock;
} GstMv3dtDiagnostic;

typedef struct _GstMv3dtDiagnosticClass {
  GstBaseTransformClass parent_class;
} GstMv3dtDiagnosticClass;

G_DEFINE_TYPE(GstMv3dtDiagnostic, gst_mv3dt_diagnostic, GST_TYPE_BASE_TRANSFORM)

#define GST_TYPE_MV3DT_DIAGNOSTIC (gst_mv3dt_diagnostic_get_type())
#define GST_MV3DT_DIAGNOSTIC(obj) ((GstMv3dtDiagnostic *)(obj))

enum { PROP_0, PROP_OUTPUT_PATH };

static gboolean read_world_feet(NvDsObjectMeta *object, float &x, float &y) {
  for (GList *item = object->obj_user_meta_list; item; item = item->next) {
    auto *user = static_cast<NvDsUserMeta *>(item->data);
    if (!user || user->base_meta.meta_type != NVDS_OBJ_WORLD_FOOT_LOCATION || !user->user_meta_data)
      continue;
    const auto *point = static_cast<const float *>(user->user_meta_data);
    x = point[0];
    y = point[1];
    return std::isfinite(x) && std::isfinite(y);
  }
  return FALSE;
}

static GstFlowReturn transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = GST_MV3DT_DIAGNOSTIC(base);
  if (!self->output) return GST_FLOW_OK;
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (!batch) return GST_FLOW_OK;

  g_mutex_lock(&self->lock);
  for (GList *frame_item = batch->frame_meta_list; frame_item; frame_item = frame_item->next) {
    auto *frame = static_cast<NvDsFrameMeta *>(frame_item->data);
    if (!frame) continue;
    for (GList *object_item = frame->obj_meta_list; object_item; object_item = object_item->next) {
      auto *object = static_cast<NvDsObjectMeta *>(object_item->data);
      if (!object || object->class_id != 0 || object->object_id == UNTRACKED_OBJECT_ID) continue;
      const float image_x = object->rect_params.left + object->rect_params.width * 0.5f;
      const float image_y = object->rect_params.top + object->rect_params.height;
      float world_x = 0.0f, world_y = 0.0f;
      const gboolean valid = read_world_feet(object, world_x, world_y);
      const guint64 timestamp = frame->ntp_timestamp ? frame->ntp_timestamp : GST_BUFFER_PTS(buffer);
      std::fprintf(self->output,
                   "{\"type\":\"native_world_observation\",\"source_id\":%u,"
                   "\"frame_number\":%u,\"timestamp\":%llu,\"reported_tracker_id\":%llu,"
                   "\"image_feet\":[%.6f,%.6f],\"world_feet\":",
                   frame->source_id, frame->frame_num,
                   static_cast<unsigned long long>(timestamp),
                   static_cast<unsigned long long>(object->object_id), image_x, image_y);
      if (valid)
        std::fprintf(self->output, "[%.8f,%.8f]", world_x, world_y);
      else
        std::fprintf(self->output, "null");
      std::fprintf(self->output, ",\"tracker_confidence\":%.6f}\n", object->tracker_confidence);
    }
  }
  std::fflush(self->output);
  g_mutex_unlock(&self->lock);
  return GST_FLOW_OK;
}

static void set_property(GObject *object, guint property_id, const GValue *value, GParamSpec *) {
  auto *self = GST_MV3DT_DIAGNOSTIC(object);
  if (property_id == PROP_OUTPUT_PATH) {
    g_free(self->output_path);
    self->output_path = g_value_dup_string(value);
    if (self->output) {
      std::fclose(self->output);
      self->output = nullptr;
    }
    if (self->output_path) self->output = std::fopen(self->output_path, "a");
  }
}

static void finalize(GObject *object) {
  auto *self = GST_MV3DT_DIAGNOSTIC(object);
  g_mutex_lock(&self->lock);
  if (self->output) std::fclose(self->output);
  self->output = nullptr;
  g_mutex_unlock(&self->lock);
  g_free(self->output_path);
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_mv3dt_diagnostic_parent_class)->finalize(object);
}

static void gst_mv3dt_diagnostic_class_init(GstMv3dtDiagnosticClass *klass) {
  auto *base = GST_BASE_TRANSFORM_CLASS(klass);
  base->transform_ip = transform_ip;
  auto *gobject = G_OBJECT_CLASS(klass);
  gobject->set_property = set_property;
  gobject->finalize = finalize;
  g_object_class_install_property(
      gobject, PROP_OUTPUT_PATH,
      g_param_spec_string("output-path", "Output path", "JSONL output path", "/workspace/runs/mv3dt_world.jsonl",
                          static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS)));
  auto *element = GST_ELEMENT_CLASS(klass);
  static GstStaticPadTemplate sink_template =
      GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  static GstStaticPadTemplate src_template =
      GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
  gst_element_class_add_static_pad_template(element, &sink_template);
  gst_element_class_add_static_pad_template(element, &src_template);
  gst_element_class_set_static_metadata(element, "MV3DT World Diagnostic", "Filter/Metadata",
                                        "Read-only NVIDIA tracker world-foot metadata diagnostic", PACKAGE);
}

static void gst_mv3dt_diagnostic_init(GstMv3dtDiagnostic *self) {
  g_mutex_init(&self->lock);
  self->output_path = g_strdup("/workspace/runs/mv3dt_world.jsonl");
  self->output = std::fopen(self->output_path, "a");
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "mv3dtdiagnostic", GST_RANK_NONE, GST_TYPE_MV3DT_DIAGNOSTIC);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mv3dtdiagnostic, "MV3DT world diagnostic",
                  plugin_init, "1.0", "Proprietary", PACKAGE, "https://nvidia.com")
