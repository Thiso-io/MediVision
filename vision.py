import cv2
import pytesseract
import pyttsx3
import re
import time
import numpy as np
import threading
import queue
import subprocess
import os
from ultralytics import YOLO

# If Tesseract is not found automatically on your Mac, uncomment this:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

model = YOLO("yolov8n.pt")

# Core settings
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540
CONF_THRESHOLD = 0.5
OCR_COOLDOWN = 4.0
MEDICAL_OBJECTS = ["bottle", "book", "cup", "cell phone", "wine glass"]
VISUAL_ALCOHOL_LABELS = ["wine glass"]

medicine_info = {
    "paracetamol": "Pain reliever. Do not exceed the recommended dose.",
    "ibuprofen": "Anti inflammatory. Best taken with food.",
    "aspirin": "Pain relief and blood thinning.",
    "alcohol": "Alcohol can interact dangerously with many medicines.",
    "water": "Important for hydration."
}

alcohol_keywords = [
    "alcohol",
    "wine",
    "beer",
    "vodka",
    "whiskey",
    "whisky",
    "rum",
    "gin",
    "champagne",
    "cider",
    "prosecco"
]

dangerous_combinations = {
    frozenset(["ibuprofen", "alcohol"]): "Danger detected. Ibuprofen and alcohol may increase the risk of stomach bleeding.",
    frozenset(["paracetamol", "alcohol"]): "Danger detected. Paracetamol and alcohol together may increase the risk of liver damage.",
    frozenset(["aspirin", "ibuprofen"]): "Danger detected. Aspirin and ibuprofen together may increase the risk of bleeding.",
    frozenset(["aspirin", "alcohol"]): "Danger detected. Aspirin and alcohol together may increase stomach bleeding risk."
}

# Window / layout
WINDOW_NAME = "MediVision"
MAX_WINDOW_W = 2560
MAX_WINDOW_H = 1664
DESIGN_CANVAS_H = 1080

# State
last_scan_time = 0
last_summary = "Scanning..."
last_name = ""
last_dose = ""
last_quantity = ""
last_warning = ""
last_ocr_preview = ""
last_danger_alert = "No dangerous combination detected."
last_announced = ""
last_danger_spoken = ""

recent_detections = []
DETECTION_MEMORY_SECONDS = 45
MAX_RECENT_DISPLAY = 6
cached_detected_boxes = []

# ---------- COLOURS (BGR) ----------
BG_TOP = (70, 34, 24)
BG_BOTTOM = (24, 12, 10)
SIDEBAR = (42, 24, 22)
CARD = (62, 38, 34)
CARD_2 = (70, 44, 40)
BORDER = (120, 94, 88)

TEXT_PRIMARY = (245, 240, 235)
TEXT_SECONDARY = (190, 175, 165)
TEXT_MUTED = (145, 130, 125)

NEON_CYAN = (255, 235, 95)
NEON_TEAL = (210, 230, 100)
NEON_VIOLET = (245, 120, 175)
NEON_RED = (110, 110, 255)
CAMERA_BOX = (180, 255, 170)

# ---------- CACHES ----------
_gradient_cache = {}
_window_bg_cache = {}

# ---------- AUDIO ----------
speech_queue = queue.Queue()


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)
    except Exception as e:
        print("Speech init error:", e)
        return

    while True:
        message = speech_queue.get()
        if message is None:
            break
        try:
            engine.stop()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print("Speech worker error:", e)


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def queue_speak(message):
    if not message:
        return
    try:
        while speech_queue.qsize() > 2:
            speech_queue.get_nowait()
    except Exception:
        pass
    speech_queue.put(message)


def play_alarm():
    try:
        sound_path = "/System/Library/Sounds/Sosumi.aiff"
        if os.path.exists(sound_path):
            subprocess.Popen(
                ["afplay", sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


# ---------- OCR / DETECTION ----------
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def extract_medicine_details(text):
    text_lower = text.lower()

    medicine_name = None
    dosage = None
    quantity = None

    for med in medicine_info:
        if med != "alcohol" and med in text_lower:
            medicine_name = med
            break

    if medicine_name is None:
        for word in alcohol_keywords:
            if word in text_lower:
                medicine_name = "alcohol"
                break

    dose_match = re.search(r"\b\d+\s*mg\b", text_lower)
    if dose_match:
        dosage = dose_match.group()

    qty_match = re.search(r"\b\d+\s*(tablets|capsules|caplets)\b", text_lower)
    if qty_match:
        quantity = qty_match.group()

    return medicine_name, dosage, quantity


def cleanup_old_detections():
    global recent_detections
    now = time.time()
    recent_detections = [
        (name, timestamp) for name, timestamp in recent_detections
        if now - timestamp <= DETECTION_MEMORY_SECONDS
    ]


def add_detection_to_memory(medicine_name):
    global recent_detections
    cleanup_old_detections()
    now = time.time()
    recent_detections = [
        (name, timestamp) for name, timestamp in recent_detections
        if name != medicine_name
    ]
    recent_detections.append((medicine_name, now))


def format_time_ago(seconds_ago):
    seconds_ago = int(seconds_ago)
    if seconds_ago < 60:
        return f"{seconds_ago}s ago"
    if seconds_ago < 3600:
        return f"{seconds_ago // 60}m ago"
    return f"{seconds_ago // 3600}h ago"


def get_recent_scans_lines():
    cleanup_old_detections()
    now = time.time()

    if not recent_detections:
        return ["-"]

    lines = []
    for name, timestamp in reversed(recent_detections[-MAX_RECENT_DISPLAY:]):
        pretty_time = format_time_ago(now - timestamp)
        lines.append(f"{name.title()}   {pretty_time}")

    return lines


def check_dangerous_combinations():
    global last_danger_alert, last_danger_spoken

    cleanup_old_detections()
    unique_recent = []
    for name, _ in recent_detections:
        if name not in unique_recent:
            unique_recent.append(name)

    found_alert = None
    for combo, warning in dangerous_combinations.items():
        if combo.issubset(set(unique_recent)):
            found_alert = warning
            break

    if found_alert:
        last_danger_alert = found_alert
        if found_alert != last_danger_spoken:
            play_alarm()
            queue_speak(found_alert)
            last_danger_spoken = found_alert
    else:
        last_danger_alert = "No dangerous combination detected."


def update_detected_medicine(medicine_name, dosage, quantity, ocr_text):
    global last_name, last_dose, last_quantity, last_warning
    global last_ocr_preview, last_summary, last_announced

    last_name = medicine_name.title()
    last_dose = dosage if dosage else "Unknown"
    last_quantity = quantity if quantity else "Unknown"
    last_warning = medicine_info.get(medicine_name, "")
    last_ocr_preview = ocr_text[:180]

    last_summary = f"Detected {last_name}"
    if dosage:
        last_summary += f" | Dose: {dosage}"
    if quantity:
        last_summary += f" | Quantity: {quantity}"

    full_speech = f"{last_name} detected."
    if dosage:
        full_speech += f" Dose {dosage}."
    if quantity and last_quantity != "Unknown":
        full_speech += f" Quantity {quantity}."
    if last_warning:
        full_speech += f" {last_warning}"

    if full_speech != last_announced:
        queue_speak(full_speech)
        last_announced = full_speech

    add_detection_to_memory(medicine_name)
    check_dangerous_combinations()


def detect_boxes_only(frame):
    results = model(frame, verbose=False)
    boxes = []

    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue
        if label not in MEDICAL_OBJECTS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2, label, conf))

    return boxes


def analyse_frame(frame):
    global last_summary, last_name, last_dose, last_quantity
    global last_warning, last_ocr_preview, cached_detected_boxes

    detected_boxes = detect_boxes_only(frame)
    cached_detected_boxes = detected_boxes
    best_ocr_text = ""

    for x1, y1, x2, y2, label, conf in detected_boxes:
        if label in VISUAL_ALCOHOL_LABELS:
            update_detected_medicine("alcohol", None, None, f"Visual detection: {label}")
            return detected_boxes

    for x1, y1, x2, y2, label, conf in detected_boxes:
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        processed = preprocess_for_ocr(cropped)
        text = pytesseract.image_to_string(processed)

        if len(text.strip()) > len(best_ocr_text.strip()):
            best_ocr_text = text

        medicine_name, dosage, quantity = extract_medicine_details(text)
        if medicine_name:
            update_detected_medicine(medicine_name, dosage, quantity, text)
            return detected_boxes

    processed_full = preprocess_for_ocr(frame)
    full_text = pytesseract.image_to_string(processed_full)

    if len(full_text.strip()) > len(best_ocr_text.strip()):
        best_ocr_text = full_text

    medicine_name, dosage, quantity = extract_medicine_details(full_text)
    if medicine_name:
        update_detected_medicine(medicine_name, dosage, quantity, full_text)
        return detected_boxes

    last_name = ""
    last_dose = ""
    last_quantity = ""
    last_warning = "No medicine detected yet."
    last_ocr_preview = best_ocr_text[:180] if best_ocr_text.strip() else ""
    last_summary = "Scanning for medicine..."
    cleanup_old_detections()
    check_dangerous_combinations()
    return detected_boxes


# ---------- DRAWING ----------
def vertical_gradient(height, width, top_bgr, bottom_bgr):
    key = (height, width, top_bgr, bottom_bgr)
    if key in _gradient_cache:
        return _gradient_cache[key].copy()

    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = [int(top_bgr[c] * (1 - t) + bottom_bgr[c] * t) for c in range(3)]
        bg[y, :] = color

    _gradient_cache[key] = bg
    return bg.copy()


def blend_rect(img, x1, y1, x2, y2, color, alpha=0.2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_card(img, x1, y1, x2, y2, glow_color=NEON_CYAN, fill=CARD):
    blend_rect(img, x1, y1, x2, y2, fill, 0.85)
    cv2.rectangle(img, (x1, y1), (x2, y2), glow_color, 1)
    cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), BORDER, 1)


def wrap_text_pixels(text, max_width_pixels, font_scale=0.48, thickness=1):
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        text_width = cv2.getTextSize(
            trial,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )[0][0]

        if text_width <= max_width_pixels:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def draw_multiline_text(
    img,
    text,
    x,
    y,
    max_width_pixels,
    line_height=20,
    color=TEXT_PRIMARY,
    scale=0.48,
    thickness=1,
    max_lines=5
):
    lines = wrap_text_pixels(
        text,
        max_width_pixels=max_width_pixels,
        font_scale=scale,
        thickness=thickness
    )[:max_lines]

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_sidebar_icon(img, cx, cy, radius, color, icon_type="dot"):
    cv2.circle(img, (cx, cy), radius, CARD_2, -1)
    cv2.circle(img, (cx, cy), radius, color, 1)

    s = max(5, radius // 3)

    if icon_type == "scan":
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), color, 2)
    elif icon_type == "history":
        cv2.circle(img, (cx, cy), max(6, radius // 2), color, 2)
        cv2.line(img, (cx, cy), (cx, cy - max(3, radius // 4)), color, 2)
        cv2.line(img, (cx, cy), (cx + max(3, radius // 4), cy), color, 2)
    elif icon_type == "alert":
        pts = np.array([
            [cx, cy - s - 2],
            [cx - s - 2, cy + s],
            [cx + s + 2, cy + s]
        ], np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    else:
        cv2.circle(img, (cx, cy), max(4, radius // 4), color, -1)


def draw_recent_scans(img, x, y, line_h, scale):
    lines = get_recent_scans_lines()
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA
        )


def draw_ring_chart(img, center, radius, progress, color):
    thickness = max(6, radius // 4)
    cv2.circle(img, center, radius, CARD_2, thickness)
    end_angle = int(360 * progress)
    cv2.ellipse(img, center, (radius, radius), -90, 0, end_angle, color, thickness)

    pct = f"{int(progress * 100)}%"
    scale = max(0.45, radius / 55)
    ts = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    cv2.putText(
        img,
        pct,
        (center[0] - ts[0] // 2, center[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_PRIMARY,
        2,
        cv2.LINE_AA
    )


def draw_mini_chart(img, x1, y1, x2, y2, color):
    values = [0.40, 0.52, 0.48, 0.66, 0.60, 0.78, 0.72]
    pts = []
    width = x2 - x1
    height = y2 - y1

    for i, v in enumerate(values):
        px = x1 + int(i * width / (len(values) - 1))
        py = y2 - int(v * height)
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    for px, py in pts:
        cv2.circle(img, (px, py), 3, color, -1)


def draw_danger_banner(img, x1, y1, x2, y2, title_scale, body_scale, line_h):
    if "Danger detected" not in last_danger_alert:
        return

    color = NEON_RED if int(time.time() * 2) % 2 == 0 else NEON_VIOLET
    draw_card(img, x1, y1, x2, y2, glow_color=color, fill=(70, 42, 52))
    cv2.putText(
        img,
        "CRITICAL INTERACTION",
        (x1 + 16, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        TEXT_PRIMARY,
        2,
        cv2.LINE_AA
    )
    draw_multiline_text(
        img,
        last_danger_alert,
        x1 + 16,
        y1 + 54,
        max_width_pixels=(x2 - x1) - 28,
        line_height=line_h,
        color=TEXT_PRIMARY,
        scale=body_scale,
        max_lines=3
    )


def fit_frame_to_box(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, scale


def place_on_full_window(canvas, out_w=MAX_WINDOW_W, out_h=MAX_WINDOW_H):
    """
    Scale dashboard to fit inside the requested output size, then place it
    centered on a full-size gradient frame. This removes the grey empty area.
    """
    bg_key = (out_w, out_h)
    if bg_key not in _window_bg_cache:
        _window_bg_cache[bg_key] = vertical_gradient(out_h, out_w, BG_TOP, BG_BOTTOM)
    out = _window_bg_cache[bg_key].copy()

    h, w = canvas.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    out[y:y + new_h, x:x + new_w] = resized
    return out


def draw_ui(camera_frame, detected_boxes):
    """
    Build dashboard on a fixed design canvas, then place it on a full-size
    output frame matching the chosen window size.
    """
    cam_src_h, cam_src_w = camera_frame.shape[:2]

    canvas_h = DESIGN_CANVAS_H
    base_scale = canvas_h / 860.0

    sidebar_w = 86
    gap = 18
    right_w = 430
    outer_pad = 18
    top_bar_h = 46

    cam_card_h = canvas_h - (outer_pad * 2 + top_bar_h + gap + 8)
    cam_card_w = int(cam_card_h * (cam_src_w / cam_src_h))

    canvas_w = outer_pad + sidebar_w + gap + cam_card_w + gap + right_w + outer_pad
    canvas = vertical_gradient(canvas_h, canvas_w, BG_TOP, BG_BOTTOM)

    title_scale = 0.58 * base_scale
    subtitle_scale = 0.48 * base_scale
    small_scale = 0.43 * base_scale
    tiny_scale = 0.39 * base_scale
    line_h = max(16, int(20 * base_scale))

    # Sidebar
    sidebar_x1 = outer_pad
    sidebar_y1 = outer_pad
    sidebar_x2 = sidebar_x1 + sidebar_w
    sidebar_y2 = canvas_h - outer_pad
    blend_rect(canvas, sidebar_x1, sidebar_y1, sidebar_x2, sidebar_y2, SIDEBAR, 0.9)
    cv2.rectangle(canvas, (sidebar_x1, sidebar_y1), (sidebar_x2, sidebar_y2), BORDER, 1)

    cv2.putText(
        canvas,
        "M",
        (sidebar_x1 + 18, sidebar_y1 + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95 * base_scale,
        TEXT_PRIMARY,
        2,
        cv2.LINE_AA
    )

    icon_cx = sidebar_x1 + sidebar_w // 2
    icon_r = max(16, int(20 * base_scale))
    draw_sidebar_icon(canvas, icon_cx, sidebar_y1 + 90, icon_r, NEON_CYAN, "scan")
    draw_sidebar_icon(canvas, icon_cx, sidebar_y1 + 155, icon_r, NEON_TEAL, "history")
    draw_sidebar_icon(canvas, icon_cx, sidebar_y1 + 220, icon_r, NEON_VIOLET, "alert")

    # Main regions
    content_x = sidebar_x2 + gap
    top_bar_y = outer_pad
    top_bar_x1 = content_x
    top_bar_x2 = canvas_w - outer_pad
    top_bar_y2 = top_bar_y + top_bar_h

    draw_card(canvas, top_bar_x1, top_bar_y, top_bar_x2, top_bar_y2, glow_color=NEON_VIOLET)
    cv2.putText(
        canvas,
        "MediVision Dashboard",
        (top_bar_x1 + 16, top_bar_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * base_scale,
        TEXT_PRIMARY,
        2,
        cv2.LINE_AA
    )

    # Camera card
    cam_x = content_x
    cam_y = top_bar_y2 + gap
    cam_w = cam_card_w
    cam_h = cam_card_h

    draw_card(canvas, cam_x, cam_y, cam_x + cam_w, cam_y + cam_h, glow_color=NEON_CYAN)

    inner_cam_pad = 8
    fitted_frame, scale = fit_frame_to_box(
        camera_frame,
        cam_w - inner_cam_pad * 2,
        cam_h - inner_cam_pad * 2
    )
    fh, fw = fitted_frame.shape[:2]
    offset_x = cam_x + (cam_w - fw) // 2
    offset_y = cam_y + (cam_h - fh) // 2
    canvas[offset_y:offset_y + fh, offset_x:offset_x + fw] = fitted_frame

    header_h = 42
    blend_rect(canvas, cam_x, cam_y, cam_x + cam_w, cam_y + header_h, (18, 14, 18), 0.45)
    cv2.putText(
        canvas,
        "Live Scan Feed",
        (cam_x + 14, cam_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58 * base_scale,
        TEXT_PRIMARY,
        2,
        cv2.LINE_AA
    )

    for x1, y1, x2, y2, label, conf in detected_boxes:
        rx1 = offset_x + int(x1 * scale)
        ry1 = offset_y + int(y1 * scale)
        rx2 = offset_x + int(x2 * scale)
        ry2 = offset_y + int(y2 * scale)

        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), CAMERA_BOX, 2)
        tag = f"{label} {int(conf * 100)}%"
        tag_scale = 0.42 * base_scale
        tw = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, 1)[0][0]
        tag_h = 22
        cv2.rectangle(
            canvas,
            (rx1, max(offset_y, ry1 - tag_h)),
            (rx1 + tw + 14, max(offset_y, ry1 - 4)),
            NEON_TEAL,
            -1
        )
        cv2.putText(
            canvas,
            tag,
            (rx1 + 7, max(offset_y + 15, ry1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            tag_scale,
            (22, 22, 22),
            1,
            cv2.LINE_AA
        )

    # Right column
    panel_x = cam_x + cam_w + gap
    panel_y = cam_y
    panel_w = right_w
    panel_h = cam_h

    section_gap = 12
    h_result = 88
    h_metrics = 82
    h_danger = 92
    h_mid = 138
    h_bottom = panel_h - (h_result + h_metrics + h_danger + h_mid + section_gap * 4)
    if h_bottom < 150:
        h_bottom = 150

    # Result
    y = panel_y
    draw_card(canvas, panel_x, y, panel_x + panel_w, y + h_result, glow_color=NEON_VIOLET)
    cv2.putText(canvas, "AI Scan Result", (panel_x + 14, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT_PRIMARY, 2, cv2.LINE_AA)
    draw_multiline_text(
        canvas,
        last_summary,
        panel_x + 14,
        y + 56,
        max_width_pixels=panel_w - 28,
        line_height=line_h,
        color=TEXT_SECONDARY,
        scale=subtitle_scale,
        max_lines=3
    )

    # Metrics
    y += h_result + section_gap
    box_w = (panel_w - 16) // 3
    metrics = [
        ("Medicine", last_name if last_name else "-", NEON_CYAN),
        ("Dose", last_dose if last_dose else "-", NEON_TEAL),
        ("Qty", last_quantity if last_quantity else "-", NEON_VIOLET),
    ]
    for i, (title, value, color) in enumerate(metrics):
        x1 = panel_x + i * (box_w + 8)
        x2 = x1 + box_w
        draw_card(canvas, x1, y, x2, y + h_metrics, glow_color=color)
        cv2.putText(canvas, title, (x1 + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, small_scale, TEXT_SECONDARY, 1, cv2.LINE_AA)
        draw_multiline_text(
            canvas,
            value,
            x1 + 10,
            y + 50,
            max_width_pixels=box_w - 18,
            line_height=17,
            color=TEXT_PRIMARY,
            scale=subtitle_scale,
            max_lines=2
        )

    # Danger
    y += h_metrics + section_gap
    draw_danger_banner(canvas, panel_x, y, panel_x + panel_w, y + h_danger,
                       title_scale, subtitle_scale, line_h)

    # Middle split
    y += h_danger + section_gap
    col_gap = 8
    col_w = (panel_w - col_gap) // 2
    left_x = panel_x
    right_x = panel_x + col_w + col_gap

    draw_card(canvas, left_x, y, left_x + col_w, y + h_mid, glow_color=NEON_TEAL)
    cv2.putText(canvas, "Drug Info", (left_x + 12, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT_PRIMARY, 2, cv2.LINE_AA)
    draw_multiline_text(
        canvas,
        last_warning if last_warning else "-",
        left_x + 12,
        y + 54,
        max_width_pixels=col_w - 24,
        line_height=line_h,
        color=TEXT_SECONDARY,
        scale=small_scale,
        max_lines=5
    )

    draw_card(canvas, right_x, y, right_x + col_w, y + h_mid, glow_color=NEON_CYAN)
    cv2.putText(canvas, "Confidence", (right_x + 12, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT_PRIMARY, 2, cv2.LINE_AA)
    progress = 0.84 if last_name else 0.36
    if "Danger detected" in last_danger_alert:
        progress = 0.96
    ring_radius = 28
    draw_ring_chart(canvas, (right_x + col_w // 2, y + 82), ring_radius, progress, NEON_CYAN)

    # Bottom split
    y += h_mid + section_gap
    bottom_h = panel_y + panel_h - y

    draw_card(canvas, left_x, y, left_x + col_w, y + bottom_h, glow_color=NEON_VIOLET)
    cv2.putText(canvas, "History", (left_x + 12, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT_PRIMARY, 2, cv2.LINE_AA)
    draw_recent_scans(canvas, left_x + 12, y + 54, 22, small_scale)

    draw_card(canvas, right_x, y, right_x + col_w, y + bottom_h, glow_color=NEON_TEAL)
    cv2.putText(canvas, "Analytics", (right_x + 12, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT_PRIMARY, 2, cv2.LINE_AA)

    chart_top = y + 48
    chart_bottom = y + 108
    draw_mini_chart(canvas, right_x + 12, chart_top, right_x + col_w - 12, chart_bottom, NEON_TEAL)

    ocr_label_y = chart_bottom + 24
    cv2.putText(canvas, "OCR Preview", (right_x + 12, ocr_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, tiny_scale, TEXT_SECONDARY, 1, cv2.LINE_AA)
    draw_multiline_text(
        canvas,
        last_ocr_preview if last_ocr_preview else "-",
        right_x + 12,
        ocr_label_y + 20,
        max_width_pixels=col_w - 24,
        line_height=16,
        color=TEXT_MUTED,
        scale=tiny_scale,
        max_lines=4
    )

    cv2.putText(canvas, "Q quit   |   S scan now", (cam_x + 8, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, tiny_scale, TEXT_SECONDARY, 1, cv2.LINE_AA)

    return place_on_full_window(canvas)


def main():
    global last_scan_time, cached_detected_boxes

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAX_WINDOW_W, MAX_WINDOW_H)
    print("Press Q to quit. Press S to force a scan.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        now = time.time()
        if now - last_scan_time >= OCR_COOLDOWN:
            cached_detected_boxes = analyse_frame(frame)
            last_scan_time = now

        ui_frame = draw_ui(frame, cached_detected_boxes)
        cv2.imshow(WINDOW_NAME, ui_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cached_detected_boxes = analyse_frame(frame)
            last_scan_time = time.time()

    try:
        speech_queue.put(None)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()