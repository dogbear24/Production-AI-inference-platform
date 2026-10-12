from pathlib import Path

import torch
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


OUTPUT_PATH = Path(
    "models/resnet18.onnx"
)


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading pretrained ResNet-18..."
    )

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.eval()

    example_input = torch.randn(
        1,
        3,
        224,
        224,
    )

    print(
        "Exporting ONNX model..."
    )

    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        input_names=[
            "input"
        ],
        output_names=[
            "logits"
        ],
        dynamic_shapes=(
            {
                0: "batch_size"
            },
        ),
        dynamo=True,
        external_data=False,
    )

    onnx_program.save(
        str(OUTPUT_PATH)
    )

    print(
        "Export complete:"
    )

    print(
        OUTPUT_PATH.resolve()
    )


if __name__ == "__main__":
    main()