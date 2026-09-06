import os
import torch
from src.edge.gatv2_model import WorkZoneSTGNN

try:
    import tensorrt as trt
    HAS_TRT = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    trt = None
    HAS_TRT = False
    TRT_LOGGER = None


def export_onnx(model_weights: str, onnx_output_path: str):
    """
    Exports the WorkZoneSTGNN model to ONNX format with dynamic batch and edge shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorkZoneSTGNN().to(device)
    if os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # Dummy batch for graph tracing
    dummy_node_hist = torch.randn(4, 30, 8, dtype=torch.float32, device=device)
    dummy_cls_ids = torch.tensor([0, 2, 0, 3], dtype=torch.long, device=device)
    dummy_edge_idx = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long, device=device)
    dummy_edge_attr = torch.randn(4, 5, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        (dummy_node_hist, dummy_cls_ids, dummy_edge_idx, dummy_edge_attr),
        onnx_output_path,
        opset_version=17,
        input_names=["node_history", "class_ids", "edge_index", "edge_attr"],
        output_names=["mode_probs", "mu_x", "mu_y", "sigma_x", "sigma_y", "rho"],
        dynamic_axes={
            "node_history": {0: "num_nodes"},
            "class_ids": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_attr": {0: "num_edges"}
        }
    )
    print(f"Exported ONNX model successfully to {onnx_output_path}")


def build_tensorrt_engine(onnx_path: str, engine_path: str):
    """
    Compiles the ONNX model into an optimized TensorRT FP16/INT8 engine for Jetson AGX Orin.
    """
    if not HAS_TRT or trt is None:
        print("TensorRT is not installed in the current environment. Skipping engine compilation.")
        return

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file into TensorRT.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic Optimization Profile for Variable Nodes and Edges
    profile = builder.create_optimization_profile()
    profile.set_shape("node_history", min=(1, 30, 8), opt=(10, 30, 8), max=(50, 30, 8))
    profile.set_shape("class_ids", min=(1,), opt=(10,), max=(50,))
    profile.set_shape("edge_index", min=(2, 1), opt=(2, 30), max=(2, 200))
    profile.set_shape("edge_attr", min=(1, 5), opt=(30, 5), max=(200, 5))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT Engine built and serialized to {engine_path}")


if __name__ == "__main__":
    onnx_file = "stgnn.onnx"
    engine_file = "stgnn_orin.engine"
    export_onnx("", onnx_file)
    build_tensorrt_engine(onnx_file, engine_file)
