from io import BytesIO
from time import perf_counter

import torch
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18
)


class ImageClassifier:

    def __init__(self):

        self.model = None
        self.transforms = None
        self.categories = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.last_model_load_ms = 0.0

    @property
    def is_loaded(self):
        return self.model is not None

    @property
    def device_name(self):

        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0)

        return "cpu"

    def load(self):

        if self.is_loaded:
            return 0.0

        start = perf_counter()

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(
            weights=weights
        ).to(self.device)

        self.model.eval()

        self.transforms = weights.transforms()

        self.categories = (
            weights.meta["categories"]
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.last_model_load_ms = (
            perf_counter() - start
        ) * 1000

        return self.last_model_load_ms

    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ):

        model_load_ms = self.load()

        try:

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

        except Exception as exc:

            raise ValueError(
                "The uploaded file is not a valid image."
            ) from exc

        # -------------------------
        # PREPROCESSING TIMER
        # -------------------------

        preprocess_start = perf_counter()

        tensor = (
            self.transforms(image)
            .unsqueeze(0)
            .to(self.device)
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        preprocess_ms = (
            perf_counter() - preprocess_start
        ) * 1000

        # -------------------------
        # INFERENCE TIMER
        # -------------------------

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_start = perf_counter()

        with torch.inference_mode():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            scores, indices = torch.topk(
                probabilities,
                k=top_k
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (
            perf_counter() - inference_start
        ) * 1000

        # -------------------------
        # FORMAT RESULTS
        # -------------------------

        predictions = []

        for score, index in zip(
            scores.cpu(),
            indices.cpu()
        ):

            predictions.append({
                "class_id":
                    int(index.item()),

                "label":
                    self.categories[index.item()],

                "probability":
                    round(
                        float(score.item()),
                        6
                    )
            })

        return {

            "model_load_ms":
                round(model_load_ms, 3),

            "preprocess_ms":
                round(preprocess_ms, 3),

            "inference_ms":
                round(inference_ms, 3),

            "predictions":
                predictions
        }
        
        
    def predict_batch(
        self,
        image_bytes_list: list[bytes],
        top_ks: list[int],
    ) -> dict:

        if not image_bytes_list:
            raise ValueError(
                "Batch must contain at least one image."
            )

        if len(image_bytes_list) != len(top_ks):
            raise ValueError(
                "Each image must have a matching top_k."
            )

        model_load_ms = self.load()

        images = []

        for image_bytes in image_bytes_list:

            try:

                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

            except Exception as exc:

                raise ValueError(
                    "A batch item is not a valid image."
                ) from exc

            images.append(image)

        # ==================================
        # BATCH PREPROCESSING
        # ==================================

        preprocess_start = perf_counter()

        tensors = [
            self.transforms(image)
            for image in images
        ]

        batch_tensor = torch.stack(
            tensors,
            dim=0,
        ).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        preprocess_ms = (
            perf_counter()
            - preprocess_start
        ) * 1000

        # ==================================
        # BATCH INFERENCE
        # ==================================

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_start = perf_counter()

        with torch.inference_mode():

            logits = self.model(
                batch_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (
            perf_counter()
            - inference_start
        ) * 1000

        # ==================================
        # SPLIT RESULTS
        # ==================================

        batch_predictions = []

        for probabilities_row, top_k in zip(
            probabilities,
            top_ks,
        ):

            scores, indices = torch.topk(
                probabilities_row,
                k=top_k,
            )

            predictions = []

            for score, index in zip(
                scores.cpu(),
                indices.cpu(),
            ):

                predictions.append({
                    "class_id":
                        int(index.item()),

                    "label":
                        self.categories[
                            index.item()
                        ],

                    "probability":
                        round(
                            float(score.item()),
                            6,
                        ),
                })

            batch_predictions.append(
                predictions
            )

        return {
            "batch_size":
                len(image_bytes_list),

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
            
            