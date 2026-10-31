"""Explicit, reproducible Hugging Face SAM3 candidate generation.

Only SAM's own instance-score filter and mask binarization run here. There is no
cleanup, NMS, connected-component filtering, or additional acceptance filter.
Heavy dependencies and checkpoint loading are deferred until construction.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


TRANSFORMERS_COMMIT = "d7dd443a4cdbdc39b8d45c98f3edd2c0840bdbcb"


def _dependencies() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        # Resolve Transformers' lazy exports now so dependency errors are useful.
        getattr(transformers, "Sam3Model")
        getattr(transformers, "Sam3Processor")
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "SAM3 dependencies could not be imported. Install this project's "
            "requirements-sam3.txt in a separate environment, with matching "
            "PyTorch/torchvision builds. Candidate selection and dry runs do not "
            "require SAM3. Original error: " + str(exc)
        ) from exc
    return torch, transformers


def _package_metadata(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
        result: dict[str, Any] = {"version": distribution.version}
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            # Keep the revision, not installation paths or URLs containing tokens.
            vcs = json.loads(direct_url).get("vcs_info", {})
            if vcs.get("commit_id"):
                result["commit_id"] = vcs["commit_id"]
        return result
    except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
        return {"version": None}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class Sam3Backend:
    """Generate candidates one image/prompt pair at a time in supplied order.

    ``confidence_threshold=0.3`` follows the pinned processor's strict ``score >
    0.3`` operation. It is a configurable SAM postprocessing parameter, not a
    fundamental lower bound imposed by SAM. ``mask_threshold=0.5`` binarizes the
    resized probabilities using strict ``>``. Neither operation removes specks
    or fills holes.

    Strict deterministic PyTorch operations are enabled process-wide. Unsupported
    nondeterministic operations fail rather than silently weakening this setting.
    Reproduction requires the same input bytes, prompts, checkpoint revision,
    package versions and hardware; cross-platform bitwise equality is not promised.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        revision: str | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        seed: int = 0,
        local_files_only: bool = False,
    ) -> None:
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("mask_threshold", mask_threshold),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty model identifier or directory")

        # This must be set before initializing CUDA/cuBLAS, including model loads.
        previous_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch, transformers = _dependencies()
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("The reproducible SAM3 backend supports cpu or cuda devices")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch environment.")
            if torch.cuda.is_initialized() and previous_cublas_config != ":4096:8":
                # A failed construction must not make a retry look correctly
                # configured after CUDA was already initialized elsewhere.
                if previous_cublas_config is None:
                    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                else:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas_config
                raise RuntimeError(
                    "CUDA was initialized before deterministic cuBLAS configuration. "
                    "Start a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8."
                )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        self.confidence_threshold = float(confidence_threshold)
        self.mask_threshold = float(mask_threshold)
        is_local_directory = os.path.isdir(model_id)
        load_args: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None and not is_local_directory:
            load_args["revision"] = revision
        try:
            self.model = transformers.Sam3Model.from_pretrained(
                model_id, attn_implementation="eager", dtype=torch.float32, **load_args
            ).to(self.device)
            self.model.eval()
            resolved_revision = getattr(self.model.config, "_commit_hash", None)
            # Resolve the processor from the same snapshot when a branch was used.
            processor_args = dict(load_args)
            if resolved_revision and not is_local_directory:
                processor_args["revision"] = resolved_revision
            self.processor = transformers.Sam3Processor.from_pretrained(model_id, **processor_args)
        except (OSError, ImportError) as exc:
            raise RuntimeError(
                f"Could not load SAM3 checkpoint {model_id!r} (requested revision {revision!r}). "
                "For local directories, verify that model, config, tokenizer and processor "
                "files are present; revision does not apply to local directories. "
                "Install requirements-sam3.txt; "
                "for facebook/sam3, obtain model access on Hugging Face and authenticate "
                "with `hf auth login` (or HF_TOKEN). For offline use, cache both model and "
                "processor first, then use --local-files-only. Original error: " + str(exc)
            ) from exc

        self.metadata: dict[str, Any] = {
            "backend": "huggingface_sam3",
            "model_id": model_id,
            "model_source": "local_directory" if is_local_directory else "huggingface_hub",
            "requested_revision": revision,
            "revision_applies_to_source": not is_local_directory,
            "resolved_revision": resolved_revision,
            "processor_revision": processor_args.get("revision"),
            "local_files_only": local_files_only,
            "confidence_threshold": self.confidence_threshold,
            "confidence_comparison": ">",
            "mask_threshold": self.mask_threshold,
            "mask_comparison": ">",
            "mask_cleanup": "none",
            "additional_score_filter": "none",
            "inference_batch_size": 1,
            "candidate_order": "supplied prompt order, then processor-returned model order",
            "seed": seed,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor(),
            "dtype": "float32",
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "tf32": False,
            "torch_num_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "packages": {name: _package_metadata(name) for name in (
                "torch", "torchvision", "transformers", "numpy", "Pillow", "huggingface-hub"
            )},
            "reproducibility_scope": (
                "Same inputs, prompt order, model/processor revision, software and hardware. "
                "Cross-version or cross-hardware bitwise equality is not guaranteed. "
                "Pin the resolved revision for subsequent runs. Local model directories "
                "must be kept unchanged by the caller."
            ),
        }

    def predict(self, images: list[Image.Image], prompts: list[list[str]]) -> list[dict[str, Any]]:
        """Return masks, boxes, scores and provenance for every returned candidate.

        ``candidate_indices`` are zero-based positions in the concatenated result.
        ``prompt_indices`` identify the supplied prompt. ``prompt_candidate_indices``
        identify each prediction's position in that prompt's *filtered* SAM result,
        not its original query index. Returned masks always have shape ``(N,H,W)``.
        """
        if len(images) != len(prompts):
            raise ValueError("Provide exactly one ordered prompt list per image")
        for prompt_list in prompts:
            if not isinstance(prompt_list, (list, tuple)) or not prompt_list:
                raise ValueError("Each image needs a nonempty ordered list of prompts")
            if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompt_list):
                raise ValueError("Prompts must be nonempty strings")

        final: list[dict[str, Any]] = []
        with self._torch.inference_mode():
            for image, prompt_list in zip(images, prompts):
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("SAM3 inputs must have positive width and height")
                rgb_image = image.convert("RGB")
                masks_parts, boxes_parts, scores_parts = [], [], []
                prompt_texts: list[str] = []
                prompt_indices: list[int] = []
                prompt_candidate_indices: list[int] = []
                for prompt_index, prompt in enumerate(prompt_list):
                    inputs = self.processor(images=rgb_image, text=prompt, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    predictions = self.processor.post_process_instance_segmentation(
                        outputs,
                        threshold=self.confidence_threshold,
                        mask_threshold=self.mask_threshold,
                        target_sizes=[(height, width)],
                    )
                    if len(predictions) != 1:
                        raise RuntimeError("SAM3 returned an unexpected number of image results")
                    result = predictions[0]
                    scores = _numpy(result["scores"])
                    boxes = _numpy(result["boxes"])
                    masks = _numpy(result["masks"])
                    if scores.ndim != 1 or boxes.shape != (len(scores), 4):
                        raise RuntimeError("SAM3 returned inconsistent box and score shapes")
                    count = len(scores)
                    if count == 0:
                        # The pinned processor leaves zero-count masks at model resolution.
                        if masks.ndim != 3 or masks.shape[0] != 0:
                            raise RuntimeError("SAM3 returned inconsistent empty masks")
                        continue
                    if masks.shape != (count, height, width):
                        raise RuntimeError("SAM3 returned masks that do not match the input dimensions")
                    # The processor owns binarization. Never interpret probabilities,
                    # NaNs or arbitrary integer labels as valid foreground via bool().
                    if not np.all((masks == 0) | (masks == 1)):
                        raise RuntimeError("SAM3 processor returned nonbinary or nonfinite masks; expected only 0 and 1")
                    masks_parts.append(masks.astype(np.bool_, copy=False))
                    boxes_parts.append(boxes)
                    scores_parts.append(scores)
                    prompt_texts.extend([prompt] * count)
                    prompt_indices.extend([prompt_index] * count)
                    prompt_candidate_indices.extend(range(count))
                    # No sorting or further score filtering: preserve processor order.

                count = len(prompt_texts)
                final.append({
                    "masks": np.concatenate(masks_parts) if count else np.empty((0, height, width), dtype=np.bool_),
                    "boxes": np.concatenate(boxes_parts) if count else np.empty((0, 4), dtype=np.float32),
                    "scores": np.concatenate(scores_parts) if count else np.empty((0,), dtype=np.float32),
                    "prompt_texts": prompt_texts,
                    "prompt_indices": np.asarray(prompt_indices, dtype=np.int64),
                    "prompt_candidate_indices": np.asarray(prompt_candidate_indices, dtype=np.int64),
                    "candidate_indices": np.arange(count, dtype=np.int64),
                })
        return final
