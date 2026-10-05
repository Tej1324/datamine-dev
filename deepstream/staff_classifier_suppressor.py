"""Hide staff objects after the TAO secondary classifier has run."""

from __future__ import annotations

import os

from pyservicemaker import BatchMetadataOperator

from track_continuity import frame_token, get_track_continuity


def _text(value) -> str:
    return str(value or "").strip().lower().replace("_", " ")


class StaffClassifierSuppressor(BatchMetadataOperator):
    """Remove staff rectangles/text from the downstream OSD metadata.

    DeepStream's Python wrapper has exposed classifier metadata under slightly
    different iterable names across releases, so this deliberately accepts the
    known wrapper variants. It is fail-open for unknown metadata, preventing a
    classifier integration issue from hiding customers.
    """

    def __init__(self):
        super().__init__()
        self.debug = os.getenv("STAFF_CLASSIFIER_DEBUG", "0") == "1"
        self.logged = False
        self.continuity = get_track_continuity()

    @staticmethod
    def _classifier_items(obj):
        for name in ("classifier_meta_items", "classifier_metas", "classifier_meta_list"):
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return list(value)
                except TypeError:
                    pass
        return []

    @staticmethod
    def _label_items(meta):
        for name in ("label_info_items", "label_infos", "label_info_list"):
            value = getattr(meta, name, None)
            if value is not None:
                try:
                    return list(value)
                except TypeError:
                    pass
        return []

    @classmethod
    def _is_staff(cls, obj) -> bool:
        for classifier in cls._classifier_items(obj):
            for label in cls._label_items(classifier):
                text = _text(getattr(label, "result_label", None) or
                             getattr(label, "label", None) or
                             getattr(label, "result_label_name", None))
                if text == "staff" or "staff" in text:
                    return True
        return False

    def handle_metadata(self, batch_meta):
        for frame in batch_meta.frame_items:
            for obj in frame.object_items:
                if int(obj.class_id) != 0:
                    continue
                rect = obj.rect_params
                continuity = self.continuity.resolve(
                    int(frame.source_id),
                    int(obj.object_id),
                    frame_token(frame),
                    self.continuity.object_embedding(obj),
                    (float(rect.left), float(rect.top), float(rect.width), float(rect.height)),
                    classifier_staff=self._is_staff(obj),
                )
                if continuity["staff"]:
                    rect = obj.rect_params
                    rect.border_width = 0
                    obj.rect_params = rect
                    text = obj.text_params
                    text.display_text = ""
                    obj.text_params = text
                elif os.getenv("FOOTFALL_LOGICAL_ID_OSD", "0") == "1":
                    text = obj.text_params
                    text.display_text = f"person {continuity['logical_track_id']}"
                    obj.text_params = text
