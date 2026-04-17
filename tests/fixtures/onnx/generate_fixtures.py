from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


ROOT = Path(__file__).resolve().parent


def save_model(name: str, graph: onnx.GraphProto) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 11
    onnx.save(model, ROOT / name)


def identity_f32() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity_f32", [x], [y])
    save_model("identity_f32.onnx", graph)


def linear_4in_1out() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "linear_4in_1out", [x], [y], [w, b])
    save_model("linear_4in_1out.onnx", graph)


def two_input_two_output() -> None:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2])
    sum_out = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2])
    diff_out = helper.make_tensor_value_info("diff", TensorProto.FLOAT, [2])
    add = helper.make_node("Add", ["lhs", "rhs"], ["sum"])
    sub = helper.make_node("Sub", ["lhs", "rhs"], ["diff"])
    graph = helper.make_graph([add, sub], "two_input_two_output", [lhs, rhs], [sum_out, diff_out])
    save_model("two_input_two_output.onnx", graph)


def dynamic_batch_linear() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "dynamic_batch_linear", [x], [y], [w, b])
    save_model("dynamic_batch_linear.onnx", graph)


def static_batch_8() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8, 1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "static_batch_8", [x], [y], [w, b])
    save_model("static_batch_8.onnx", graph)


def malformed() -> None:
    (ROOT / "malformed.onnx").write_text("not a valid onnx model\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    identity_f32()
    linear_4in_1out()
    two_input_two_output()
    dynamic_batch_linear()
    static_batch_8()
    malformed()


if __name__ == "__main__":
    main()
