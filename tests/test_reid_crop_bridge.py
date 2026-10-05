import struct

import numpy as np
import pytest

from deepstream.reid_experiment_crop_worker import HEADER, MAGIC, VERSION


def test_crop_packet_header_contract():
    values = (MAGIC, VERSION, 24, 1, 22, 33, 44, 1.0, 2.0, 3.0, 4.0, 0.9, 0.8, 2, 4, 1)
    encoded = HEADER.pack(*values)
    assert HEADER.size == struct.calcsize("<IIIIQQQffffffIII")
    unpacked = HEADER.unpack(encoded)
    assert unpacked[:7] == values[:7]
    assert unpacked[7:13] == pytest.approx(values[7:13], abs=1e-6)
    assert unpacked[13:] == values[13:]


def test_bgr_crop_payload_shape():
    crop = np.zeros((4, 2, 3), dtype=np.uint8)
    assert crop.nbytes == 24
