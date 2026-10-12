import os

from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
)


class ONNXImageClassifier:

    def __init__(
        self,
        model_path: str = (
            "models/resnet18.onnx"
        ),
    ):

        self.model_path = Path(
            os.getenv(
                "ONNX_MODEL_PATH",
                model_path,
            )
        )

        self.weights = (
            ResNet18_Weights.DEFAULT
        )

        self.transforms = (
            self.weights.transforms()
        )

        self.categories = (
            self.weights.meta[
                "categories"
            ]
        )

        self.session = None

        self.input_name = None
        self.output_name = None

        self.device_name = (
            "ONNX Runtime "
            "(not loaded)"
        )

    def load(self):

        if self.session is not None:
            return 0.0

        if not self.model_path.exists():

            raise RuntimeError(
                "ONNX model not found: "
                f"{self.model_path}"
            )

        started = perf_counter()

        requested_provider = (
            os.getenv(
                "ONNX_PROVIDER",
                "auto",
            ).lower()
        )

        available = (
            ort.get_available_providers()
        )

        print(
            "[onnx] Available providers:",
            available,
        )

        if requested_provider == "cuda":

            if (
                "CUDAExecutionProvider"
                not in available
            ):

                raise RuntimeError(
                    "CUDAExecutionProvider "
                    "was requested but is "
                    "not available."
                )

            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        elif requested_provider == "cpu":

            providers = [
                "CPUExecutionProvider"
            ]

        else:

            if (
                "CUDAExecutionProvider"
                in available
            ):

                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

            else:

                providers = [
                    "CPUExecutionProvider"
                ]

        self.session = (
            ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.output_name = (
            self.session
            .get_outputs()[0]
            .name
        )

        active_provider = (
            self.session
            .get_providers()[0]
        )

        self.device_name = (
            "ONNX Runtime / "
            f"{active_provider}"
        )

        print(
            "[onnx] Active provider:",
            active_provider,
        )

        return (
            perf_counter()
            - started
        ) * 1000

    def predict_batch(
        self,
        image_bytes_list:
            list[bytes],
        top_ks:
            list[int],
    ) -> dict:

        if not image_bytes_list:

            raise ValueError(
                "Batch must contain "
                "at least one image."
            )

        if (
            len(image_bytes_list)
            != len(top_ks)
        ):

            raise ValueError(
                "Each image must have "
                "a matching top_k."
            )

        model_load_ms = (
            self.load()
        )

        preprocess_started = (
            perf_counter()
        )

        tensors = []

        for image_bytes in (
            image_bytes_list
        ):

            try:

                image = Image.open(
                    BytesIO(
                        image_bytes
                    )
                ).convert(
                    "RGB"
                )

            except Exception as exc:

                raise ValueError(
                    "A batch item is "
                    "not a valid image."
                ) from exc

            tensor = (
                self.transforms(
                    image
                )
            )

            tensors.append(
                tensor.numpy()
            )

        batch = np.stack(
            tensors,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        preprocess_ms = (
            perf_counter()
            - preprocess_started
        ) * 1000

        inference_started = (
            perf_counter()
        )

        logits = (
            self.session.run(
                [
                    self.output_name
                ],
                {
                    self.input_name:
                        batch
                },
            )[0]
        )

        inference_ms = (
            perf_counter()
            - inference_started
        ) * 1000

        # Numerically stable softmax
        shifted = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True,
            )
        )

        exp_values = np.exp(
            shifted
        )

        probabilities = (
            exp_values
            / np.sum(
                exp_values,
                axis=1,
                keepdims=True,
            )
        )

        batch_predictions = []

        for row, top_k in zip(
            probabilities,
            top_ks,
        ):

            indices = np.argsort(
                row
            )[-top_k:][::-1]

            predictions = []

            for index in indices:

                predictions.append({
                    "class_id":
                        int(index),

                    "label":
                        self.categories[
                            int(index)
                        ],

                    "probability":
                        round(
                            float(
                                row[index]
                            ),
                            6,
                        ),
                })

            batch_predictions.append(
                predictions
            )

        return {
            "batch_size":
                len(
                    image_bytes_list
                ),

            "model_load_ms":
                round(
                    model_load_ms,
                    3,
                ),

            "preprocess_ms":
                round(
                    preprocess_ms,
                    3,
                ),

            "inference_ms":
                round(
                    inference_ms,
                    3,
                ),

            "predictions":
                batch_predictions,
        }