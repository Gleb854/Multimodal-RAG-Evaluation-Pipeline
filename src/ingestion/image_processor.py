# image_processor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json
import re

from PIL import Image


@dataclass
class ProcessedChunk:
    content: str
    metadata: dict


class ImageProcessor:
    """
    Goals:
    - Skip tiny images (icons/logos/crops)
    - Skip vision if caption already informative
    - Cache vision results by image hash to avoid repeated calls
    - Use proxy if provided
    - Never hard-fail the whole pipeline if vision errors
    """

    # --- tuneable thresholds (safe defaults for papers) ---
    MIN_FILE_BYTES = 10_000          # skip very small files
    MIN_DIM_PX = 220                # skip if max(width,height) < MIN_DIM_PX
    MIN_AREA_PX2 = 80_000           # skip if width*height < MIN_AREA_PX2

    # ---- gating keywords ----
    FIGURE_HINT_RE = re.compile(
        r"\b(fig(ure)?|diagram|architecture|workflow|pipeline|framework|system model|block diagram|topology|schematic|overview|setup|layout)\b",
        re.IGNORECASE,
    )
    PLOT_HINT_RE = re.compile(
        r"\b(plot|curve|accuracy|precision|recall|auc|loss|throughput|latency|cdf|pdf|vs\.?|versus|x-axis|y-axis)\b",
        re.IGNORECASE,
    )
    LOW_VALUE_RE = re.compile(
        r"\b(logo|icon|copyright|©|all rights reserved|arxiv|preprint|submitted|accepted|author|affiliation)\b",
        re.IGNORECASE,
    )

    # caption informativeness heuristics
    MIN_CAPTION_CHARS = 80
    MIN_CAPTION_TOKENS = 12

    # figure/table number extraction
    _RE_FIG_NUM = re.compile(r"(?:fig\.|figure)\s*([0-9]+|[ivxlcdm]+)", re.IGNORECASE)
    _RE_TABLE_NUM = re.compile(r"(?:table)\s*([0-9]+|[ivxlcdm]+)", re.IGNORECASE)

    def __init__(
        self,
        output_dir: Path,
        openai_api_key: Optional[str],
        vision_model: str = "gpt-4o",
        proxy_url: Optional[str] = None,
        cache_path: Optional[Path] = None,
        enable_cache: bool = True,
        log_prefix: str = "[ImageProcessor]",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.openai_api_key = openai_api_key
        self.vision_model = vision_model
        self.proxy_url = proxy_url
        self.enable_cache = enable_cache
        self.log_prefix = log_prefix

        # hard guard: deprecated model name
        if self.vision_model.strip() == "gpt-4-vision-preview":
            raise ValueError(
                "vision_model='gpt-4-vision-preview' is deprecated. "
                "Use a current multimodal model, e.g. 'gpt-4o-mini' or 'gpt-4o'."
            )

        self.cache_path = cache_path or (self.output_dir / ".image_vision_cache.json")
        self._cache: Dict[str, str] = {}
        if self.enable_cache:
            self._cache = self._load_cache()

        # lazy client init (only if we will call vision)
        self._client = None

        print(
            f"{self.log_prefix} proxy={'ON' if bool(self.proxy_url) else 'OFF'}, "
            f"vision_model={self.vision_model}, openai_key_set={bool(self.openai_api_key)}; "
            f"images_dir={self.output_dir}"
        )

    # -------------------- public API --------------------

    def process(
        self,
        image_path: Optional[str],
        caption: str,
        metadata: Dict[str, Any],
        generate_description: bool = True,
    ) -> ProcessedChunk:
        caption = (caption or "").strip()
        meta = dict(metadata or {})
        meta["element_type"] = "image"
        meta["image_path"] = image_path

        # --- C) meta keys for retrieval filtering ---
        meta.setdefault("doc_type", "image")
        meta.setdefault("source_pdf", meta.get("source") or meta.get("filename") or meta.get("document") or None)
        meta.setdefault("page", meta.get("page") or None)
        meta.setdefault("image_index", meta.get("image_index") or meta.get("idx") or None)

        # Extract figure_number from caption for retrieval boosting
        if caption and not meta.get("figure_number"):
            fig_match = self._RE_FIG_NUM.search(caption)
            if fig_match:
                meta["figure_number"] = fig_match.group(1).upper()
        
        # Extract table_number if caption mentions table (rare for images, but possible)
        if caption and not meta.get("table_number"):
            tbl_match = self._RE_TABLE_NUM.search(caption)
            if tbl_match:
                meta["table_number"] = tbl_match.group(1).upper()

        if not image_path:
            meta["vision_decision"] = "skip_vision"
            meta["vision_skip_reason"] = "no_image_path"
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=None, meta=meta, note="no image_path"),
                metadata=meta,
            )

        p = Path(image_path)
        if not p.exists():
            meta["vision_decision"] = "skip_vision"
            meta["vision_skip_reason"] = "image_missing"
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=None, meta=meta, note="image missing"),
                metadata=meta,
            )

        # compute hash early for traceability (and retrieval filtering if needed)
        img_hash = self._sha1_file(p)
        meta["image_sha1"] = img_hash

        # A) skip tiny images
        if self._is_too_small(p):
            meta["vision_decision"] = "skip_index"
            meta["vision_skip_reason"] = "too_small"
            meta["figure_candidate"] = False
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=None, meta=meta, note="skipped: too small"),
                metadata=meta,
            )

        # B) gating decision: cover/logo/low-value vs figure-like
        decision, reason, is_figure_candidate = self._gating_decision(p, caption, meta)
        meta["figure_candidate"] = bool(is_figure_candidate)
        meta["vision_decision"] = decision
        meta["vision_skip_reason"] = reason

        # If caption already informative -> skip vision (keeps current behavior but tracked)
        if self._caption_informative(caption):
            meta["vision_decision"] = "skip_vision"
            meta["vision_skip_reason"] = "caption_informative"
            return ProcessedChunk(
                content=self._format_image_doc(
                    caption=caption,
                    description="[Skipped vision: caption already informative]",
                    meta=meta,
                    note=None,
                ),
                metadata=meta,
            )

        # If gating says "skip_vision" (low value) — do not call vision
        if decision == "skip_vision":
            return ProcessedChunk(
                content=self._format_image_doc(
                    caption=caption,
                    description=None,
                    meta=meta,
                    note=f"skipped vision: {reason}",
                ),
                metadata=meta,
            )

        if not generate_description:
            meta["vision_decision"] = "skip_vision"
            meta["vision_skip_reason"] = "vision_disabled"
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=None, meta=meta, note="vision disabled"),
                metadata=meta,
            )

        if not self.openai_api_key:
            meta["vision_decision"] = "skip_vision"
            meta["vision_skip_reason"] = "no_openai_key"
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=None, meta=meta, note="no openai key"),
                metadata=meta,
            )

        # cache
        if self.enable_cache and img_hash in self._cache:
            desc = self._cache[img_hash]
            meta["vision_cached"] = True
            meta["vision_decision"] = "cache_hit"
            return ProcessedChunk(
                content=self._format_image_doc(caption=caption, description=desc, meta=meta, note=None),
                metadata=meta,
            )

        # do vision
        try:
            desc = self._describe_with_vision(p, caption=caption)
        except Exception as e:
            meta["vision_decision"] = "error"
            meta["vision_skip_reason"] = f"{type(e).__name__}: {e}"
            return ProcessedChunk(
                content=self._format_image_doc(
                    caption=caption,
                    description=None,
                    meta=meta,
                    note=f"vision error: {type(e).__name__}: {e}",
                ),
                metadata=meta,
            )

        if self.enable_cache:
            self._cache[img_hash] = desc
            self._save_cache()

        meta["vision_decision"] = "ok"
        return ProcessedChunk(
            content=self._format_image_doc(caption=caption, description=desc, meta=meta, note=None),
            metadata=meta,
        )

    # -------------------- heuristics --------------------

    def _gating_decision(self, p: Path, caption: str, meta: Dict[str, Any]):
        """
        Returns: (decision, reason, is_figure_candidate)
          decision in {"do_vision", "skip_vision"}
        """
        clean = re.sub(r"\s+", " ", (caption or "")).strip()

        # If caption screams "low-value" (logo/cover/legal) and not figure-like => skip vision
        if clean and self.LOW_VALUE_RE.search(clean) and not self.FIGURE_HINT_RE.search(clean):
            return "skip_vision", "low_value_caption", False

        # If caption indicates it is a figure/diagram/plot => do vision (unless informative caption already)
        if clean and (self.FIGURE_HINT_RE.search(clean) or self.PLOT_HINT_RE.search(clean)):
            return "do_vision", "figure_or_plot_hint", True

        # Additional heuristic: very large single-image on first page with no figure hint often = cover
        page = meta.get("page")
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w = h = None

        if page in (0, 1, "0", "1") and (not clean or len(clean) < 20):
            if w and h and (w * h) > 1_500_000:
                return "skip_vision", "probable_cover_first_page", False

        # Default: allow vision only if image looks substantial and caption isn't empty-garbage
        if not clean:
            return "skip_vision", "no_caption_no_hint", False

        # Short label-like captions (e.g. "Access Point Microgrid") — often useful diagrams => do vision
        tokens = [t for t in re.split(r"[\s,/;:()\[\]{}]+", clean) if t]
        if 2 <= len(tokens) <= 8 and len(clean) <= 60:
            return "do_vision", "short_label_caption", True

        # Otherwise: skip by default (conservative)
        return "skip_vision", "no_strong_signal", False

    def _is_too_small(self, p: Path) -> bool:
        try:
            if p.stat().st_size < self.MIN_FILE_BYTES:
                return True
        except Exception:
            # if we cannot stat, do not skip here
            pass

        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            # if PIL can't read, don't treat as tiny; let vision try (or fail gracefully)
            return False

        if max(w, h) < self.MIN_DIM_PX:
            return True
        if (w * h) < self.MIN_AREA_PX2:
            return True

        return False

    def _caption_informative(self, caption: str) -> bool:
        if not caption:
            return False

        clean = re.sub(r"\s+", " ", caption).strip()

        # If it's basically only labels/short words (like "Access Point Microgrid"), it's NOT informative
        tokens = [t for t in re.split(r"[\s,/;:()\[\]{}]+", clean) if t]
        if len(clean) >= self.MIN_CAPTION_CHARS and len(tokens) >= self.MIN_CAPTION_TOKENS:
            return True

        # Also treat captions with full sentences as informative
        if any(x in clean for x in [". ", "; ", ": "]):
            # still require some length
            return len(tokens) >= 10

        return False

    # -------------------- vision call --------------------

    def _get_client(self):
        if self._client is not None:
            return self._client

        from openai import OpenAI
        import httpx

        if self.proxy_url:
            # httpx 0.28+ uses proxy=..., older versions use proxies=...
            http_client = None
            try:
                http_client = httpx.Client(proxy=self.proxy_url, timeout=httpx.Timeout(60.0))
            except TypeError:
                http_client = httpx.Client(proxies=self.proxy_url, timeout=httpx.Timeout(60.0))

            self._client = OpenAI(api_key=self.openai_api_key, http_client=http_client)
        else:
            self._client = OpenAI(api_key=self.openai_api_key)

        return self._client

    def _describe_with_vision(self, image_path: Path, caption: str) -> str:
        client = self._get_client()
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        prompt = (
            "Describe the figure/diagram for retrieval QA.\n"
            "Focus on: (1) components, (2) relationships/flows/arrows, (3) what the figure conveys.\n"
            "If the figure is a plot: mention axes, curves, trends, and conclusion.\n"
            "Be concise but specific.\n"
        )
        if caption:
            prompt += f"\nCaption (may be partial/ocr): {caption}\n"

        # Using Chat Completions style (commonly supported). If you use Responses API, adapt accordingly.
        resp = client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + self._b64(img_bytes)}},
                    ],
                }
            ],
            temperature=0.0,
        )

        text = (resp.choices[0].message.content or "").strip()
        return text if text else "No description returned."

    @staticmethod
    def _b64(b: bytes) -> str:
        import base64
        return base64.b64encode(b).decode("utf-8")

    # -------------------- cache helpers --------------------

    def _sha1_file(self, p: Path) -> str:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        try:
            if self.cache_path.exists():
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        try:
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception:
            pass

    # -------------------- formatting --------------------

    def _format_image_doc(self, caption: str, description: Optional[str], meta: dict, note: Optional[str]) -> str:
        page = meta.get("page")
        idx = meta.get("image_index")
        fig_no = meta.get("figure_number")

        header = "Image"
        if idx is not None and page is not None:
            header = f"Image {idx} (Page {page})"
        elif page is not None:
            header = f"Image (Page {page})"

        parts = [f"{header}:"]
        if fig_no:
            parts.append(f"Figure: Fig. {fig_no}")
        if caption:
            parts.append(f"Caption: {caption}")
        if description:
            parts.append(f"Description: {description}")
        if note:
            parts.append(f"Note: {note}")
        return " ".join(parts).strip()
