import copy
import os
import warnings

import onnx
import onnxruntime
import torch
import torch.onnx.utils as onnx_utils
from onnxruntime.quantization import QuantType, quant_pre_process
from onnxruntime.quantization.quantize import quantize_dynamic

from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SAMOnnxModel
from src.args import parse_export_args
from src.load_checkpoint import get_sam_vit_t


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    num_classes: int = 4,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    print("Loading model...")
    if model_type == "vit_t":
        sam = get_sam_vit_t(checkpoint_path=checkpoint, resume=False, num_mask_outputs=num_classes)
    else:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = SAMOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            onnx_utils.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    _fix_shared_initializers_identity(output)

    ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
    # set cpu provider default
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(output, providers=providers)
    _ = ort_session.run(None, ort_inputs)
    print("Model has successfully been run with ONNXRuntime.")


def _fix_shared_initializers_identity(model_path):
    print(f"Loading exported model {model_path} to resolve shared initializers/Identity nodes...")
    model = onnx.load(model_path)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    identity_nodes_to_remove = []
    new_initializers = []

    for node in graph.node:
        if node.op_type == "Identity":
            inp = node.input[0]
            out = node.output[0]
            if inp in initializers:
                src_init = initializers[inp]
                new_init = copy.deepcopy(src_init)
                new_init.name = out
                new_initializers.append(new_init)
                identity_nodes_to_remove.append(node)

    if identity_nodes_to_remove:
        for new_init in new_initializers:
            graph.initializer.append(new_init)
        for node in identity_nodes_to_remove:
            graph.node.remove(node)
        onnx.save(model, model_path)
        print(
            f"Successfully resolved and removed {len(identity_nodes_to_remove)} Identity nodes forwarding shared initializers."
        )


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parse_export_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        num_classes=args.num_classes,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_out is not None:
        print(f"Quantizing model and writing to {args.quantize_out}...")

        temp_dir = os.path.dirname(args.quantize_out) or "."
        temp_preprocessed = os.path.join(temp_dir, "temp_preprocessed.onnx")
        try:
            print("Running preprocessing (quant_pre_process)...")
            quant_pre_process(
                args.output,
                temp_preprocessed,
                skip_symbolic_shape=False,
            )

            print("Running quantize_dynamic...")
            quantize_dynamic(
                model_input=temp_preprocessed,
                model_output=args.quantize_out,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QInt8,
                extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
            )
            print("Quantization completed successfully!")
        finally:
            if os.path.exists(temp_preprocessed):
                try:
                    os.remove(temp_preprocessed)
                except Exception:
                    pass
