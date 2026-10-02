from typing import Sequence
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from src.preprocessing import fixed_image_standardization, apply_clahe


class FaceEmbedder:
    """
    Deep facial feature embedder using pre-trained InceptionResnetV1 (FaceNet).
    Extracts 512-dimensional L2-normalized feature vectors.
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        device: torch.device | str | None = None,
        use_clahe: bool = True
    ):
        """
        Initialize FaceNet embedder.

        Args:
            pretrained: Model weights ('vggface2' or 'casia-webface').
            device: Computing device ('cuda' or 'cpu').
            use_clahe: Whether to apply CLAHE lighting normalization before embedding.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pre-trained FaceNet backbone
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

        # Freeze all weights for inference
        for param in self.model.parameters():
            param.requires_grad = False

        self.pretrained = pretrained
        self.embedding_dim = 512
        self.use_clahe = use_clahe
        self.to_tensor = transforms.ToTensor()

    def _preprocess_crop(self, crop: Image.Image | np.ndarray) -> torch.Tensor:
        """
        Convert a 160x160 face crop to standardized tensor (3, 160, 160).
        Applies CLAHE lighting normalization, then FaceNet standardization: (x - 127.5) / 128.0.
        """
        if isinstance(crop, np.ndarray):
            crop_pil = Image.fromarray(crop.astype("uint8")).convert("RGB")
        else:
            crop_pil = crop.convert("RGB")

        if crop_pil.size != (160, 160):
            crop_pil = crop_pil.resize((160, 160), Image.Resampling.BILINEAR)

        # Apply CLAHE to normalize lighting variations
        if self.use_clahe:
            crop_pil = apply_clahe(crop_pil)

        # (x - 127.5) / 128.0
        np_arr = np.array(crop_pil, dtype=np.float32)
        tensor = torch.from_numpy(np_arr).permute(2, 0, 1)  # (3, 160, 160)
        return fixed_image_standardization(tensor)

    @torch.no_grad()
    def embed_crop(self, crop: Image.Image | np.ndarray, tta: bool = False) -> np.ndarray:
        """
        Generate 512-d L2-normalized embedding for a single 160x160 face crop.

        Args:
            crop: Face crop (PIL Image or numpy array).
            tta: If True, uses Test-Time Augmentation (averages embeddings of
                 the original and horizontally flipped crops for pose invariance).

        Returns:
            np.ndarray of shape (512,)
        """
        if not tta:
            tensor = self._preprocess_crop(crop).unsqueeze(0).to(self.device)
            embedding = self.model(tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.squeeze(0).cpu().numpy().astype(np.float32)

        # Test-Time Augmentation: fuse original and horizontally flipped embeddings
        if isinstance(crop, np.ndarray):
            crop_pil = Image.fromarray(crop.astype("uint8")).convert("RGB")
        else:
            crop_pil = crop.convert("RGB")

        flip_pil = crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
        t1 = self._preprocess_crop(crop_pil)
        t2 = self._preprocess_crop(flip_pil)
        batch = torch.stack([t1, t2], dim=0).to(self.device)
        embs = self.model(batch)
        fused = embs[0] + embs[1]
        fused = torch.nn.functional.normalize(fused.unsqueeze(0), p=2, dim=1)
        return fused.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_batch(self, crops: Sequence[Image.Image | np.ndarray]) -> np.ndarray:
        """
        Generate 512-d L2-normalized embeddings for a batch of face crops.

        Args:
            crops: Sequence of face crops.

        Returns:
            np.ndarray of shape (N, 512)
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors = [self._preprocess_crop(c) for c in crops]
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        embeddings = self.model(batch_tensor)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)
