from datetime import datetime
import os
import pickle
from typing import Any
import numpy as np


class FaceRecognizer:
    """
    Cosine similarity-based Face Recognition engine.
    Matches face embeddings against enrolled student templates.
    """

    def __init__(
        self,
        embeddings_db_path: str | None = None,
        similarity_threshold: float = 0.70
    ):
        self.threshold = similarity_threshold
        self.students: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}

        # Cached flat matrices for high-speed vectorized batch matching
        self._all_embeddings: np.ndarray | None = None
        self._embedding_labels: list[str] = []

        if embeddings_db_path and os.path.exists(embeddings_db_path):
            self.load_database(embeddings_db_path)

    def load_database(self, path: str) -> None:
        """Load enrolled embeddings database from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.students = data.get("students", {})
        self.metadata = data.get("metadata", {})
        self._rebuild_lookup_cache()

    def save_database(self, path: str) -> None:
        """Save enrolled embeddings database to pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "students": self.students,
            "metadata": {
                "total_students": len(self.students),
                "total_embeddings": sum(s["image_count"] for s in self.students.values()),
                "model": "InceptionResnetV1 (vggface2)",
                "embedding_dim": 512,
                "saved_at": datetime.now().isoformat()
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def set_threshold(self, threshold: float) -> None:
        """Update similarity threshold."""
        self.threshold = float(threshold)

    def enroll_student(
        self,
        roll_number: str,
        embeddings: list[np.ndarray] | np.ndarray
    ) -> None:
        """
        Enroll or update a student's embeddings.

        Args:
            roll_number: Student identifier (folder name).
            embeddings: List or array of 512-d L2-normalized embeddings.
        """
        if isinstance(embeddings, list):
            embs = np.array(embeddings, dtype=np.float32)
        else:
            embs = embeddings.astype(np.float32)

        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        # Ensure L2 normalization
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embs = embs / norms

        # Compute normalized centroid (mean template)
        mean_vector = np.mean(embs, axis=0)
        mean_norm = np.linalg.norm(mean_vector)
        centroid = mean_vector / (mean_norm if mean_norm > 0 else 1e-10)

        self.students[roll_number] = {
            "embeddings": embs,
            "centroid": centroid.astype(np.float32),
            "image_count": len(embs)
        }
        self._rebuild_lookup_cache()

    def _rebuild_lookup_cache(self) -> None:
        """Build flat matrix and label array for fast vectorized dot-product matching."""
        if not self.students:
            self._all_embeddings = None
            self._embedding_labels = []
            return

        all_embs = []
        labels = []
        for roll, data in sorted(self.students.items()):
            for emb in data["embeddings"]:
                all_embs.append(emb)
                labels.append(roll)

        self._all_embeddings = np.array(all_embs, dtype=np.float32)
        self._embedding_labels = labels

    def recognize_single(
        self,
        embedding: np.ndarray,
        threshold: float | None = None,
        top_k: int = 3,
        min_confidence_gap: float = 0.03
    ) -> dict[str, Any]:
        """
        Recognize an individual 512-d query embedding using fused scoring.

        Scoring combines three complementary signals for robust matching:
        - Centroid similarity: how close to the student's average face template
        - Max similarity: best single-image match (captures closest pose/angle)
        - Top-K mean similarity: average of top K nearest embeddings (filters outliers)

        Final score = 0.4 * centroid_sim + 0.3 * max_sim + 0.3 * top_k_mean_sim

        A confidence gap check rejects matches where the top two candidates are
        too close in score, indicating the model cannot reliably distinguish them.

        Args:
            embedding: 512-d L2-normalized query embedding.
            threshold: Cosine similarity threshold (defaults to self.threshold).
            top_k: Number of top candidate students to return.
            min_confidence_gap: Minimum score gap between #1 and #2 to accept a match.

        Returns:
            {
                "identity": roll_number or "UNKNOWN",
                "similarity": float,
                "is_match": bool,
                "threshold": float,
                "top_candidates": [{"roll_number": str, "similarity": float, "centroid_similarity": float, "fused_score": float}, ...]
            }
        """
        thresh = self.threshold if threshold is None else float(threshold)

        if not self.students or self._all_embeddings is None:
            return {
                "identity": "UNKNOWN",
                "similarity": 0.0,
                "is_match": False,
                "threshold": thresh,
                "top_candidates": []
            }

        # Normalize query embedding
        q = embedding.flatten().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Compute cosine similarity with every enrolled embedding via dot product: q . E^T
        sims = np.dot(self._all_embeddings, q)

        # Aggregate per-student: max similarity, centroid similarity, and top-K mean
        student_scores: dict[str, dict[str, float]] = {}
        for roll, s_data in self.students.items():
            centroid_sim = float(np.dot(s_data["centroid"], q))

            # Gather all similarities for this student's embeddings
            student_sims = []
            for emb in s_data["embeddings"]:
                student_sims.append(float(np.dot(emb, q)))

            max_sim = max(student_sims)
            # Top-K mean: average of the K highest similarities (more robust than just max)
            sorted_sims = sorted(student_sims, reverse=True)
            top_k_sims = sorted_sims[:min(3, len(sorted_sims))]
            top_k_mean = sum(top_k_sims) / len(top_k_sims)

            # Fused score: weighted combination of three signals
            fused_score = 0.4 * centroid_sim + 0.3 * max_sim + 0.3 * top_k_mean

            student_scores[roll] = {
                "max_similarity": max_sim,
                "centroid_similarity": centroid_sim,
                "top_k_mean": top_k_mean,
                "fused_score": fused_score
            }

        # Rank students by fused score
        ranked = sorted(
            [
                {
                    "roll_number": roll,
                    "similarity": round(scores["max_similarity"], 4),
                    "centroid_similarity": round(scores["centroid_similarity"], 4),
                    "fused_score": round(scores["fused_score"], 4)
                }
                for roll, scores in student_scores.items()
            ],
            key=lambda x: x["fused_score"],
            reverse=True
        )

        top_candidates = ranked[:top_k]
        best_candidate = ranked[0] if ranked else None
        second_best = ranked[1] if len(ranked) > 1 else None

        if best_candidate and best_candidate["fused_score"] >= thresh:
            # Confidence gap check: reject if top two candidates are too close
            if second_best and (best_candidate["fused_score"] - second_best["fused_score"]) < min_confidence_gap:
                identity = "UNKNOWN"
                best_sim = best_candidate["fused_score"]
                is_match = False
            else:
                identity = best_candidate["roll_number"]
                best_sim = best_candidate["fused_score"]
                is_match = True
        else:
            identity = "UNKNOWN"
            best_sim = best_candidate["fused_score"] if best_candidate else 0.0
            is_match = False

        return {
            "identity": identity,
            "similarity": best_sim,
            "is_match": is_match,
            "threshold": thresh,
            "top_candidates": top_candidates
        }

    def recognize_batch(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Recognize multiple embeddings."""
        return [self.recognize_single(emb, threshold=threshold) for emb in embeddings]
