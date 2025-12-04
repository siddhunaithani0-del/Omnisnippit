import os
import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.video.fx.all as vfx
from ultralytics import YOLO
from streamlit_sortables import sort_items  # for drag & drop reordering

PREVIEW_DIR = "previews"
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Do NOT trim shots shorter than this
MIN_PROTECT_SECONDS = 0.7

# Target aspect for vertical reels
TARGET_ASPECT = 9 / 16

# YOLO model for person + laptop detection
yolo = YOLO("yolov8n.pt")


st.set_page_config(layout="wide")
# ---------------- FULL-WIDTH STREAMLIT TOP BAR OVERRIDE ----------------
st.markdown(
    """
    <style>
    /* Override default Streamlit header */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #ff0077, #ff8800) !important;
        height: 60px !important;
    }

    /* Hide default Streamlit logo area */
    header[data-testid="stHeader"]::before {
        content: "OMNISNIPPET";
        position: absolute;
        left: 24px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 2px;
        color: white;
    }

    /* Push app content below colored header */
    .block-container {
        padding-top: 90px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Say hello to OMNISNIPPET. The smarter way to adapt videos.")


# ==========================================================
#        COMPOSITION ANALYSIS (MODEL + PRODUCT)
# ==========================================================

def sample_frames(video_path, start, end, num_samples=4):
    """
    Sample a few frames evenly from a shot using OpenCV only.
    This avoids MoviePy 'failed to read the first frame' issues.
    Returns a list of RGB frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration = total_frames / fps

    # clamp start/end to video duration
    s = max(0.0, min(start, duration))
    e = max(0.0, min(end, duration))

    if e <= s:
        cap.release()
        return []

    # avoid sampling exactly the very last frame
    frame_margin = 1.0 / fps
    e = max(s + frame_margin, e - 1e-3)

    times = np.linspace(s, e, num_samples)
    frames = []

    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def detect_person_product(frame_rgb):
    """Detect person + laptop in a frame using YOLO."""
    H, W, _ = frame_rgb.shape
    res = yolo.predict(frame_rgb, verbose=False)[0]

    persons, laptops = [], []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "laptop":
            laptops.append((x1, y1, x2, y2))

    return persons, laptops, (W, H)


def biggest_box(boxes):
    """Return the largest bounding box."""
    if not boxes:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def union_boxes(boxes):
    """Return union of multiple boxes."""
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def compute_vertical_crop(W, H, key_box):
    """Compute vertical 9:16 crop around key region (person + product)."""
    target_w = int(H * TARGET_ASPECT)

    # Already vertical/narrow → no horizontal crop
    if target_w >= W:
        return [0, 0, W, H]

    if key_box is None:
        x_center = W // 2
    else:
        x1k, y1k, x2k, y2k = key_box
        x_center = (x1k + x2k) / 2

    x1 = int(x_center - target_w / 2)
    x2 = x1 + target_w

    # Clamp inside image
    if x1 < 0:
        x1 = 0
        x2 = target_w
    if x2 > W:
        x2 = W
        x1 = W - target_w

    return [x1, 0, x2, H]


def composition_ok(crop_box, person_box):
    """
    Enforce framing rules:
    - Person fully inside crop
    - Person height between 35–80% of frame height
    - Head near upper third of frame
    """
    if person_box is None or crop_box is None:
        return False

    cx1, cy1, cx2, cy2 = crop_box
    px1, py1, px2, py2 = person_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    # Transform person box into crop coordinates
    px1c = px1 - cx1
    px2c = px2 - cx1
    py1c = py1 - cy1
    py2c = py2 - cy1

    # Person must be fully inside crop
    if px1c < 0 or px2c > crop_w or py1c < 0 or py2c > crop_h:
        return False

    person_h = py2c - py1c
    scale = person_h / crop_h

    # Person not too tiny or too huge
    if scale < 0.35 or scale > 0.8:
        return False

    # Head position (top of person) roughly upper third
    head_y = py1c
    if not (0.1 * crop_h <= head_y <= 0.45 * crop_h):
        return False

    return True


def analyze_shot(video_path, start, end):
    """
    Analyze a shot:
    - sample frames
    - detect person + laptop
    - compute key region
    - compute vertical 9:16 crop
    - decide if framing is 'good' or not

    Returns:
        crop_box = [x1, y1, x2, y2] in original frame coords
        is_good  = True / False
    """
    frames = sample_frames(video_path, start, end, num_samples=4)
    if not frames:
        return None, False

    all_persons = []
    all_laptops = []
    W = H = None

    for f in frames:
        persons, laptops, (W, H) = detect_person_product(f)
        all_persons.extend(persons)
        all_laptops.extend(laptops)

    if W is None or H is None:
        return None, False

    main_person = biggest_box(all_persons)

    if main_person is None:
        crop_box = compute_vertical_crop(W, H, None)
        return crop_box, False

    key_boxes = [main_person] + all_laptops
    key_region = union_boxes(key_boxes)

    crop_box = compute_vertical_crop(W, H, key_region)
    is_good = composition_ok(crop_box, main_person)

    return crop_box, is_good


def apply_crop_to_clip(clip, crop_box):
    """Apply spatial crop to a MoviePy clip (no resize here)."""
    if crop_box is None:
        return clip
    x1, y1, x2, y2 = crop_box
    return vfx.crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)


def compute_square_crop_from_box(W, H, crop_box):
    """
    Compute a 1:1 crop using the same YOLO-based logic as 9:16:
    - Use the horizontal center of the existing vertical crop_box (person+product center)
    - Make a square around that center
    - Clamp inside the original frame
    If crop_box is None, fall back to simple center square crop.
    """
    side = int(min(W, H))

    # Fallback: normal center crop (no YOLO info)
    if crop_box is None:
        x1 = (W - side) // 2
        y1 = (H - side) // 2
        return [x1, y1, x1 + side, y1 + side]

    cx1, cy1, cx2, cy2 = crop_box
    x_center = (cx1 + cx2) / 2.0

    x1 = int(x_center - side / 2.0)
    # Clamp horizontally
    if x1 < 0:
        x1 = 0
    if x1 + side > W:
        x1 = W - side

    # Vertical: just center the square
    y1 = (H - side) // 2

    return [x1, y1, x1 + side, y1 + side]


# --------------------------------------------------------
# SHOT POST-PROCESSING – MERGE ONLY SHORT SHOTS
# --------------------------------------------------------
def merge_short_shots(shots, min_len_sec=0.5):
    """
    Merge very short shots into the previous one so a single continuous
    shot isn't broken into lots of 0.04s / 0.08s chunks.

    shots: list of (start, end) tuples in seconds.
    min_len_sec: any shot shorter than this is merged into previous.
    """
    if not shots:
        return []

    merged = [[shots[0][0], shots[0][1]]]

    for s, e in shots[1:]:
        dur = e - s
        if dur < min_len_sec:
            # extend previous shot
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


# --------------------------------------------------------
# TRUE SHOT DETECTION (HARD CUTS)
# --------------------------------------------------------
def detect_shots(video_path, threshold=25.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    shots = []
    prev_gray = None
    shot_start = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)

            if score > threshold:
                t = frame_idx / fps
                shots.append((shot_start, t))
                shot_start = t

        prev_gray = gray
        frame_idx += 1

    duration = frame_idx / fps
    shots.append((shot_start, duration))
    cap.release()

    # merge only very short shots (to avoid micro-chops)
    shots = merge_short_shots(shots, min_len_sec=0.5)
    return shots


# --------------------------------------------------------
# PREVIEW GENERATION (NO AUDIO, KEEP ORIGINAL ASPECT, ALWAYS REGENERATE)
# --------------------------------------------------------
def generate_previews(video_path, shot_meta):
    """
    shot_meta: list of dicts with keys:
        start, end, crop, is_good
    """
    base = VideoFileClip(video_path)
    preview_paths = []

    for i, shot in enumerate(shot_meta):
        s = float(shot["start"])
        e = float(shot["end"])
        out_path = os.path.join(PREVIEW_DIR, f"shot_{i:03d}.mp4")

        # ALWAYS REGENERATE PREVIEW (delete old cached file)
        if os.path.exists(out_path):
            os.remove(out_path)

        s = max(0.0, s)
        e = min(e, base.duration)
        if e <= s:
            continue

        # ORIGINAL ASPECT, JUST RESIZED FOR SPEED
        clip = base.subclip(s, e).without_audio().resize(height=720)

        clip.write_videofile(
            out_path,
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
        )
        clip.close()

        preview_paths.append(out_path)

    base.close()
    return preview_paths


# -------------- NEW: create/get thumbnail from preview --------------
def get_thumbnail_from_preview(preview_path):
    """
    Create a JPG thumbnail from the first frame of the preview video
    (stored next to the preview). If it already exists, reuse it.
    """
    base_name = os.path.splitext(os.path.basename(preview_path))[0]
    thumb_path = os.path.join(PREVIEW_DIR, f"{base_name}.jpg")

    if not os.path.exists(thumb_path):
        cap = cv2.VideoCapture(preview_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(thumb_path, frame)

    if os.path.exists(thumb_path):
        return thumb_path
    else:
        return None
# -------------------------------------------------------------------


# --------------------------------------------------------
# FINAL RENDER (TRIMMING + OPTIONAL VERTICAL / SQUARE / 16:9 CROP + CONTINUOUS BGM)
# --------------------------------------------------------
def render_final(video_path, selected_shots, outname, target_duration,
                 vertical_9_16=False, square_1_1=False, youtube_16_9=False):
    """
    selected_shots: list of dicts {start, end, crop, is_good}
    vertical_9_16: export vertical reels (9:16)
    square_1_1: export square (1:1)
    youtube_16_9: export 16:9 (YouTube 1920x1080)
    """
    base = VideoFileClip(video_path)

    # Clean and clamp shots
    valid_shots = []
    for shot in selected_shots:
        s = max(0.0, float(shot["start"]))
        e = min(float(shot["end"]), base.duration)
        if e > s:
            new_shot = dict(shot)
            new_shot["start"] = s
            new_shot["end"] = e
            valid_shots.append(new_shot)

    if not valid_shots:
        st.error("No valid shots selected.")
        base.close()
        return

    orig_lengths = [sh["end"] - sh["start"] for sh in valid_shots]
    total_original = sum(orig_lengths)

    # ---------- Duration logic ----------
    if target_duration <= 0 or target_duration >= total_original:
        segments = [
            (sh, sh["start"], sh["end"])
            for sh in valid_shots
        ]
        st.info("No trimming required.")
    else:
        # Protect very short shots
        protected_indices = [i for i, L in enumerate(orig_lengths) if L < MIN_PROTECT_SECONDS]
        adjustable_indices = [i for i, L in enumerate(orig_lengths) if L >= MIN_PROTECT_SECONDS]

        protected_duration = sum(orig_lengths[i] for i in protected_indices)
        adjustable_total = total_original - protected_duration

        if protected_duration >= target_duration or adjustable_total <= 0:
            # Must scale everything
            scale = target_duration / total_original
            st.info(
                f"Total selected duration {total_original:.2f}s > target "
                f"{target_duration:.2f}s → trimming ALL shots with scale {scale:.3f}"
            )
            new_lengths = [L * scale for L in orig_lengths]
        else:
            # Scale only long shots
            scale = (target_duration - protected_duration) / adjustable_total
            st.info(
                f"Total selected duration {total_original:.2f}s > target {target_duration:.2f}s "
                f"→ trimming only shots ≥ {MIN_PROTECT_SECONDS:.2f}s with scale {scale:.3f}"
            )
            new_lengths = orig_lengths[:]
            for i in adjustable_indices:
                new_lengths[i] = orig_lengths[i] * scale

        # Fix rounding on last adjustable shot
        diff = target_duration - sum(new_lengths)
        if abs(diff) > 0.05 and adjustable_indices:
            last_i = adjustable_indices[-1]
            new_lengths[last_i] = max(0.05, new_lengths[last_i] + diff)

        segments = []
        for sh, new_len in zip(valid_shots, new_lengths):
            s = sh["start"]
            new_end = min(s + new_len, base.duration)
            if new_end > s:
                segments.append((sh, s, new_end))

    # ---------- Build clips ----------
    clips = []
    for sh, s, e in segments:
        sub = base.subclip(s, e).without_audio()

        if vertical_9_16 and sh.get("crop") is not None:
            # Existing vertical 9:16 behavior (YOLO-based)
            sub = apply_crop_to_clip(sub, sh["crop"]).resize(height=1920)
        elif square_1_1:
            # 1:1 smart crop using the same YOLO-driven crop box
            w, h = sub.w, sub.h
            square_box = compute_square_crop_from_box(w, h, sh.get("crop"))
            x1, y1, x2, y2 = square_box
            sub = vfx.crop(sub, x1=x1, y1=y1, x2=x2, y2=y2).resize(height=1080)
        elif youtube_16_9:
            # Simple 16:9 YouTube export (1920x1080)
            sub = sub.resize((1920, 1080))

        clips.append(sub)

    if not clips:
        st.error("Nothing to render after trimming.")
        base.close()
        return

    # Concatenate video
    final = concatenate_videoclips(clips, method="compose")

    # Continuous background music from original video
    try:
        if base.audio:
            bgm = base.audio.subclip(0, final.duration)
            final = final.set_audio(bgm)
            final.write_videofile(outname, codec="libx264", audio_codec="aac")
        else:
            final.write_videofile(outname, codec="libx264", audio=False)
    except Exception as e:
        print("Audio error, writing silent video:", e)
        final.write_videofile(outname, codec="libx264", audio=False)

    final.close()
    for c in clips:
        c.close()
    base.close()


# --------------------------------------------------------
# UI
# --------------------------------------------------------
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv"])

if uploaded_video:
    video_path = "input.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.subheader("Source Video")
    st.video(video_path)

    # -------- Analyze button --------
    if st.button("Analyze & Detect Shots"):
        with st.spinner("Detecting shots and analyzing composition..."):
            raw_shots = detect_shots(video_path)

            shot_meta = []
            for (s, e) in raw_shots:
                crop_box, is_good = analyze_shot(video_path, s, e)
                shot_meta.append(
                    {
                        "start": s,
                        "end": e,
                        "crop": crop_box,
                        "is_good": is_good,
                    }
                )

            previews = generate_previews(video_path, shot_meta)

        st.session_state["shots"] = shot_meta
        st.session_state["previews"] = previews

    # -------- Shot selection UI --------
    if "shots" in st.session_state:
        st.subheader("Detected Shots")

        selected_indices = []
        cols = st.columns(4)

        for i, (shot, preview) in enumerate(zip(st.session_state["shots"], st.session_state["previews"])):  # noqa: E501
            with cols[i % 4]:
                st.video(preview)
                s = shot["start"]
                e = shot["end"]
                comp_tag = "✅ GOOD" if shot["is_good"] else "⚠️ WEAK"
                st.write(f"{s:.2f}s → {e:.2f}s  (len {(e - s):.2f}s) {comp_tag}")
                if st.checkbox(f"Select {i}", key=f"cb_{i}"):
                    selected_indices.append(i)

        # ---------- Reorder selected shots (drag & drop) ----------
        if selected_indices:
            st.markdown("##### Reorder Selected Shots (drag to change order)")

            # HIGHLIGHTED total duration of all selected shots (NEW STYLE)
            total_selected_duration_ui = sum(
                st.session_state["shots"][i]["end"] - st.session_state["shots"][i]["start"]
                for i in selected_indices
            )
            st.markdown(
                f"""
                <div style="
                    background-color:#ffe8e8;
                    color:#b00020;
                    padding:6px 12px;
                    border-radius:999px;
                    display:inline-block;
                    font-weight:600;
                    margin-bottom:8px;
                ">
                    Total duration of selected shots: {total_selected_duration_ui:.2f}s
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Labels for the sortables (carry index so we can decode later)
            sortable_labels = []
            for idx in selected_indices:
                sh = st.session_state["shots"][idx]
                s = sh["start"]
                e = sh["end"]
                label = f"{idx}: {s:.2f}s → {e:.2f}s"
                sortable_labels.append(label)

            # Horizontal drag & drop strip
            sorted_labels = sort_items(
                sortable_labels,
                direction="horizontal"
            )

            # Decode back to indices
            ordered_indices = [int(lbl.split(":", 1)[0]) for lbl in sorted_labels]

            # NEW: thumbnail images (uniform grid) in the current order
            st.markdown("###### Selected shot thumbnails (in current order)")
            thumb_count = len(ordered_indices)
            if thumb_count > 0:
                max_cols = 6  # fixed number of columns so each cell has same width
                for row_start in range(0, thumb_count, max_cols):
                    row_indices = ordered_indices[row_start:row_start + max_cols]
                    thumb_cols = st.columns(max_cols)
                    for col_idx in range(max_cols):
                        with thumb_cols[col_idx]:
                            if col_idx < len(row_indices):
                                idx = row_indices[col_idx]
                                preview_path = st.session_state["previews"][idx]
                                thumb_path = get_thumbnail_from_preview(preview_path)
                                if thumb_path is not None:
                                    st.image(thumb_path, use_container_width=True)
                                else:
                                    st.video(preview_path)
                                st.caption(f"Shot {idx}")
                            else:
                                st.empty()
            # --- END NEW ---
        else:
            ordered_indices = selected_indices
        # ---------- END reorder block ----------

        st.subheader("Final Output Settings")

        target_dur = st.number_input("Target Duration (seconds)", value=10.0)
        output_name = st.text_input("Output File Name", value="final_output.mp4")

        vertical_flag = st.checkbox(
            "Export as vertical 9:16 (Reels / TikTok format)",
            value=True,
        )

        square_flag = st.checkbox(
            "Export as square 1:1 (Feed / WhatsApp)",
            value=False,
        )

        youtube_flag = st.checkbox(
            "Export as 16:9 (YouTube)",
            value=False,
        )

        # Handle multiple format selections with clear priority
        selected_format_count = sum([vertical_flag, square_flag, youtube_flag])
        if selected_format_count > 1:
            st.warning(
                "Multiple formats selected. Priority is: 9:16 > 1:1 > 16:9 (YouTube). "
                "Only the highest priority selected format will be used."
            )

        effective_vertical = vertical_flag
        effective_square = square_flag and not effective_vertical
        effective_youtube = youtube_flag and not (effective_vertical or effective_square)

        # If both 9:16 and 1:1 were originally selected, this maintains your older warning
        if vertical_flag and square_flag:
            st.warning("Both 9:16 and 1:1 are selected. Using 9:16 for export.")

        if st.button("Render Final Video"):
            # Use the reordered indices (from drag & drop)
            selected_shots = [st.session_state["shots"][i] for i in ordered_indices]

            if not selected_shots:
                st.error("Select at least one shot.")
            else:
                # Ensure selected shots total >= target duration
                total_selected_duration = sum(
                    float(sh["end"]) - float(sh["start"]) for sh in selected_shots
                )

                if total_selected_duration < float(target_dur):
                    st.warning(
                        f"Selected shots total duration is only "
                        f"{total_selected_duration:.2f}s, which is less than "
                        f"the target duration {float(target_dur):.2f}s. "
                        "Please select more shots or reduce the target duration."
                    )
                else:
                    with st.spinner("Rendering final video..."):
                        render_final(
                            video_path,
                            selected_shots,
                            output_name,
                            target_dur,
                            vertical_9_16=effective_vertical,
                            square_1_1=effective_square,
                            youtube_16_9=effective_youtube,
                        )

                    st.success("Final video rendered!")
                    st.video(output_name)


# --------------------------------------------------------
# HIGHLIGHTED TAGLINE (LAST LINE)
# --------------------------------------------------------
st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#ff0077,#ff8800);
        color:white;
        padding:16px;
        border-radius:12px;
        text-align:center;
        font-size:22px;
        font-weight:700;
        margin-top:40px;
    ">
        More speed. More control. More efficiency.
    </div>
    """,
    unsafe_allow_html=True,
)
