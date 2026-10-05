"""VLM-guided multi-room spatial reasoning for photo-tier stitching.

Step 1 of the VLM photo stitching workflow:
- Ingests protocol photos (entry door photo, corner photos, exit/hallway photo).
- Prompts Qwen2-VL (2B or 7B) with visual evidence across rooms to identify
  visible doorways, hallway sightlines, and shared openings.
- Returns a validated JSON adjacency map:
    {
      "connections": [
        {"room_a": "room1", "room_b": "room2", "shared_opening": "interior_door"},
        {"room_a": "room2", "room_b": "room3", "shared_opening": "hallway"}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

log = logging.getLogger("floorplan.vlm")

MODEL_ALIASES = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "2b": "Qwen/Qwen2-VL-2B-Instruct",
    "7b": "Qwen/Qwen2-VL-7B-Instruct",
}

DEFAULT_MAX_IMAGE_DIM = 768


def resolve_model_name(name_or_alias: str) -> str:
    """Map friendly aliases like 'qwen2-vl-2b' to full HuggingFace repo IDs."""
    key = name_or_alias.strip().lower()
    return MODEL_ALIASES.get(key, name_or_alias)


def natural_sort_key(path_or_str: str | Path) -> list:
    name = Path(path_or_str).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def select_protocol_photos(
    room_images: dict[str, list[str]],
    max_photos_per_room: int = 4,
) -> dict[str, list[tuple[str, str]]]:
    """Select protocol photos for each room:
    1. Entry door photo looking into room (first photo).
    2. Exit photo looking out into hallway/adjacent space (last photo).
    3. Intermediate hallway/transition sightlines if the room contains connecting corridor shots.

    Returns:
        dict mapping room_id -> list of (photo_path, role_label)
    """
    selected: dict[str, list[tuple[str, str]]] = {}

    for room_id, paths in room_images.items():
        if not paths:
            continue
        sorted_paths = sorted(paths, key=natural_sort_key)
        n = len(sorted_paths)
        room_picks: list[tuple[str, str]] = []

        if n == 1:
            room_picks.append((sorted_paths[0], "entry_or_overview"))
        elif n <= max_photos_per_room:
            # Small set: first is entry, last is exit/doorway, middle are interior
            room_picks.append((sorted_paths[0], "entry_door"))
            for p in sorted_paths[1:-1]:
                room_picks.append((p, "interior_perspective"))
            room_picks.append((sorted_paths[-1], "doorway_exit_to_hallway"))
        else:
            # Protocol capture: photo 1 = entry, photos 2-5 = 4 corners, photo 6 = doorway looking out
            # For hallway rooms (n > 6), include intermediate sightlines
            room_picks.append((sorted_paths[0], "entry_door"))
            if n > 6:
                # Hallway or transitional space: take representative connecting shots
                step = n // (max_photos_per_room - 1)
                for idx in range(step, n - 1, step):
                    if len(room_picks) < max_photos_per_room - 1:
                        room_picks.append((sorted_paths[idx], "hallway_connecting_view"))
            room_picks.append((sorted_paths[-1], "doorway_exit_to_hallway"))

        selected[room_id] = room_picks

    return selected


def prepare_image(image_path: str | Path, max_dim: int = DEFAULT_MAX_IMAGE_DIM) -> Image.Image:
    """Load, apply EXIF rotation, convert to RGB, and resize so max dimension <= max_dim."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def build_vlm_prompt(
    selected_photos: dict[str, list[tuple[str, str]]],
) -> tuple[list[dict[str, Any]], str]:
    """Build Qwen2-VL chat messages containing downsampled images with clear room role tags."""
    content: list[dict[str, Any]] = []

    content.append({
        "type": "text",
        "text": (
            "You are an expert architectural vision reasoning model. You are given photos taken from "
            "multiple rooms in a building capture following a strict protocol:\n"
            "- Entry photo: Taken at/outside the entrance door looking into the room.\n"
            "- Doorway exit photo: Taken standing in the doorway looking out into the hallway or adjoining room.\n"
            "- Hallway views: Taken inside corridors connecting different room doors.\n\n"
            "Below are the photos organized by room ID and role:\n"
        )
    })

    room_ids = list(selected_photos.keys())
    for r_id in room_ids:
        content.append({
            "type": "text",
            "text": f"\n--- Photos for Room: '{r_id}' ---\n"
        })
        for photo_path, role in selected_photos[r_id]:
            img = prepare_image(photo_path)
            content.append({
                "type": "text",
                "text": f"[{r_id} - {role} ({Path(photo_path).name})]:"
            })
            content.append({
                "type": "image",
                "image": img,
            })

    instructions = (
        f"\nTask: Analyze the visual evidence across all {len(room_ids)} rooms "
        f"({', '.join(room_ids)}). Look at door frames, wall paint/trim, flooring transitions, "
        "and visible sightlines through doorways.\n"
        "Identify which rooms are physically adjacent and connect directly to each other.\n"
        "Return ONLY a clean JSON object conforming strictly to this format:\n"
        "{\n"
        '  "connections": [\n'
        '    {"room_a": "room1", "room_b": "room2", "shared_opening": "interior_door"},\n'
        '    {"room_a": "room2", "room_b": "room3", "shared_opening": "hallway"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- 'room_a' and 'room_b' MUST be valid room IDs from the input.\n"
        "- Do not connect a room to itself.\n"
        "- List each pair only once (undirected connection).\n"
        "- 'shared_opening' can be 'interior_door', 'hallway', or 'open_doorway'.\n"
        "- Output raw JSON only. Do not add conversational text or markdown explanation outside the JSON."
    )
    content.append({"type": "text", "text": instructions})

    messages = [{"role": "user", "content": content}]
    return messages, instructions


def parse_vlm_adjacency_json(raw_text: str, valid_room_ids: list[str]) -> dict:
    """Extract and validate the adjacency JSON from the VLM output."""
    raw_text = raw_text.strip()

    # Match JSON block within markdown or standalone
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw_text[start : end + 1]
        else:
            json_str = raw_text

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning("VLM response was not valid JSON: %s. Response was: %r", exc, raw_text[:200])
        return {"connections": []}

    if not isinstance(data, dict) or "connections" not in data or not isinstance(data["connections"], list):
        log.warning("VLM JSON missing 'connections' list: %s", data)
        return {"connections": []}

    norm_map = {rid.lower(): rid for rid in valid_room_ids}
    cleaned_conns: list[dict[str, str]] = []
    seen_pairs: set[frozenset[str]] = set()

    for item in data["connections"]:
        if not isinstance(item, dict):
            continue
        ra = str(item.get("room_a", "")).strip()
        rb = str(item.get("room_b", "")).strip()
        opening = str(item.get("shared_opening", "interior_door")).strip()

        # Map to valid canonical room IDs
        canon_a = norm_map.get(ra.lower())
        canon_b = norm_map.get(rb.lower())

        if not canon_a or not canon_b:
            log.warning("VLM returned connection with unknown room: %s <-> %s", ra, rb)
            continue
        if canon_a == canon_b:
            continue

        pair = frozenset([canon_a, canon_b])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cleaned_conns.append({
            "room_a": canon_a,
            "room_b": canon_b,
            "shared_opening": opening if opening in ("interior_door", "hallway", "open_doorway") else "interior_door",
        })

    return {"connections": cleaned_conns}


def load_qwen2_vl_model(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "auto",
    load_in_4bit: bool = False,
):
    """Load Qwen2-VL model and processor with optimal device mapping and precision."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    full_model_name = resolve_model_name(model_name)
    log.info("Loading VLM model: %s (device=%s, 4bit=%s)", full_model_name, device, load_in_4bit)

    # Determine device and torch dtype
    if device == "auto":
        if torch.cuda.is_available():
            target_device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = "mps"
            dtype = torch.float16
        else:
            target_device = "cpu"
            dtype = torch.float32
    else:
        target_device = device
        dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

    kwargs: dict[str, Any] = {"torch_dtype": dtype}

    if target_device == "cuda":
        if load_in_4bit:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                log.info("4-bit quantization enabled via bitsandbytes")
            except ImportError:
                log.warning(
                    "bitsandbytes is not installed; falling back to standard bfloat16 on CUDA. "
                    "To enable 4-bit, run: pip install -U 'bitsandbytes>=0.46.1'"
                )
        kwargs["device_map"] = "auto"
    elif target_device == "mps":
        # MPS device mapping
        kwargs["device_map"] = None

    processor = AutoProcessor.from_pretrained(
        full_model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=DEFAULT_MAX_IMAGE_DIM * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(full_model_name, **kwargs)

    if target_device == "mps":
        model = model.to("mps")
    elif target_device == "cpu" and kwargs.get("device_map") is None:
        model = model.to("cpu")

    model.eval()
    return model, processor, target_device


def infer_room_adjacency_vlm(
    room_images: dict[str, list[str]],
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "auto",
    load_in_4bit: bool = False,
    cache_path: Optional[str | Path] = None,
    max_image_dim: int = DEFAULT_MAX_IMAGE_DIM,
) -> dict:
    """Run Step 1 (Vision Reasoning) to predict physical room adjacency graph.

    Args:
        room_images: dict mapping room_id -> list of image file paths
        model_name: "qwen2-vl-2b", "qwen2-vl-7b", or HuggingFace repo path
        device: "auto", "cuda", "mps", or "cpu"
        load_in_4bit: use 4-bit NF4 quantization on CUDA
        cache_path: optional JSON path to read from / write to
        max_image_dim: maximum image dimension when passing to VLM

    Returns:
        Dict adhering to {"connections": [{"room_a": ..., "room_b": ..., "shared_opening": ...}]}
    """
    valid_rooms = list(room_images.keys())
    if len(valid_rooms) < 2:
        return {"connections": []}

    # Check cache
    if cache_path:
        cp = Path(cache_path)
        if cp.is_file():
            try:
                cached_data = json.loads(cp.read_text())
                parsed = parse_vlm_adjacency_json(json.dumps(cached_data), valid_rooms)
                if parsed.get("connections"):
                    log.info("Loaded VLM adjacency map from cache: %s (%d connections)", cp, len(parsed["connections"]))
                    return parsed
            except Exception as e:
                log.warning("Could not read cache %s: %s", cp, e)

    selected = select_protocol_photos(room_images)
    messages, _ = build_vlm_prompt(selected)

    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        log.error("Required VLM dependencies not found: %s", e)
        return {"connections": []}

    model, processor, target_device = load_qwen2_vl_model(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
    )

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if target_device in ("cuda", "mps"):
        inputs = inputs.to(target_device)

    log.info("Running VLM inference with %s on %d protocol images across %d rooms...",
             model_name, len(image_inputs or []), len(valid_rooms))

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    log.debug("VLM raw output:\n%s", response_text)
    result = parse_vlm_adjacency_json(response_text, valid_rooms)

    # Save to cache if requested
    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(result, indent=2))
        log.info("Saved VLM adjacency map to %s", cp)

    return result
