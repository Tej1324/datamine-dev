"""Wrap a TAO binary scalar margin as explicit staff/customer logits."""

from pathlib import Path
import sys

import onnx
from onnx import helper


src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model = onnx.load(str(src))
graph = model.graph
old = graph.output[0]
margin = old.name
neg_name = "staff_score_neg_margin"
out_name = "staff_customer_logits"
graph.node.extend([
    helper.make_node("Neg", [margin], [neg_name], name="StaffScoreNegate"),
    helper.make_node("Concat", [neg_name, margin], [out_name], axis=1, name="StaffCustomerLogits"),
])
graph.output.remove(old)
graph.output.extend([helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, [None, 2])])
onnx.checker.check_model(model)
onnx.save(model, str(dst))
print(dst)
