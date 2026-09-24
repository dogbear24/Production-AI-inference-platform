
from io import BytesIO

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


class ImageClassifier:
    def __init__(self) -> None:
        self.model = None
        self.transforms = None
        self.categories = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0)
        return "cpu"

    def load(self) -> None:
        if self.is_loaded:
            return

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        self.categories = weights.meta["categories"]

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict]:
        self.load()

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc

        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            scores, indices = torch.topk(probabilities, k=top_k)

        results = []
        for score, index in zip(scores.cpu(), indices.cpu()):
            results.append(
                {
                    "class_id": int(index.item()),
                    "label": self.categories[index.item()],
                    "probability": round(float(score.item()), 6),
                }
            )
        return results
