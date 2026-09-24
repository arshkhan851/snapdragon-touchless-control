"""
Touchless Laptop Control
=========================
Gesture-controlled media playback + presence-based auto-lock for
Snapdragon-powered HP laptops.

Built for the Snapdragon AI Lab Build & Present Challenge.

Models used (both pre-optimized by Qualcomm and validated on
Snapdragon X Elite / X Plus, per aihub.qualcomm.com):
  - MediaPipe-Hand-Gesture-Recognition -> controls media playback
  - MediaPipe-Face-Detection           -> detects presence -> auto-lock

Everything runs 100% on-device using your webcam. No video frame is
ever uploaded anywhere.

Gesture -> Action:
  Open_Palm    -> Play
  Closed_Fist  -> Pause
  Thumb_Up     -> Volume Up
  Thumb_Down   -> Volume Down

If no face is seen for FACE_ABSENCE_LOCK_SEC seconds, the screen locks
automatically (Windows). This is a presence/privacy feature, NOT a
biometric login replacement -- actual sign-in still goes through your
normal Windows password / Windows Hello.

Setup:
  py -3.11 -m pip install qai_hub_models keyboard opencv-python
Run:
  py -3.11 gesture_media_control.py
Quit:
  press 'q' with the video window focused
"""

import ctypes
import time

import cv2

from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from qai_hub_models.models.mediapipe_hand_gesture.app import MediaPipeHandGestureApp
from qai_hub_models.models.mediapipe_hand_gesture.model import MediaPipeHandGesture

try:
    import keyboard
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'keyboard'. Run: pip install keyboard"
    ) from exc


# ---------------------------------------------------------------------------
# Config -- tune these to taste
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
GESTURE_HOLD_FRAMES = 3        # frames a gesture must be held before it fires
GESTURE_COOLDOWN_SEC = 1.2     # minimum time between two triggers
FACE_ABSENCE_LOCK_SEC = 30     # seconds with no face before auto-lock

GESTURE_ACTIONS = {
    "Open_Palm": ("Play", lambda: keyboard.send("play/pause media")),
    "Closed_Fist": ("Pause", lambda: keyboard.send("play/pause media")),
    "Thumb_Up": ("Volume Up", lambda: keyboard.send("volume up")),
    "Thumb_Down": ("Volume Down", lambda: keyboard.send("volume down")),
}


def lock_windows() -> None:
    ctypes.windll.user32.LockWorkStation()


# ---------------------------------------------------------------------------
# Load models (weights download automatically on first run)
# ---------------------------------------------------------------------------
print("Loading models... (first run may take a minute)")

hand_model = MediaPipeHandGesture.from_pretrained()
hand_app = MediaPipeHandGestureApp.from_pretrained(hand_model)

face_model = MediaPipeFace.from_pretrained()
face_app = MediaPipeFaceApp.from_pretrained(face_model)

print("Models loaded.")

# ---------------------------------------------------------------------------
# Webcam loop
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

last_gesture = None
gesture_streak = 0
last_trigger_time = 0.0
last_face_seen = time.time()
locked_already = False

print("\nControls:")
for g, (label, _) in GESTURE_ACTIONS.items():
    print(f"  {g:12s} -> {label}")
print(f"  No face for {FACE_ABSENCE_LOCK_SEC}s -> Lock screen")
print("\nPress 'q' in the video window to quit.\n")

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---- Hand gesture ----
    gesture = None
    try:
        raw = hand_app.predict_landmarks_from_image(frame_rgb, raw_output=True)
        # raw = (boxes, keypoints, roi_corners, landmarks, is_right_hand, gesture_labels)
        gesture_labels = raw[5][0] if len(raw) > 5 and raw[5] else []
        if gesture_labels:
            gesture = gesture_labels[0]
    except Exception:
        gesture = None

    now = time.time()
    if gesture is not None and gesture in GESTURE_ACTIONS:
        if gesture == last_gesture:
            gesture_streak += 1
        else:
            gesture_streak = 1
            last_gesture = gesture

        if (
            gesture_streak == GESTURE_HOLD_FRAMES
            and (now - last_trigger_time) > GESTURE_COOLDOWN_SEC
        ):
            label, action = GESTURE_ACTIONS[gesture]
            print(f"[{time.strftime('%H:%M:%S')}] Gesture: {gesture} -> {label}")
            action()
            last_trigger_time = now
    else:
        last_gesture = None
        gesture_streak = 0

    # ---- Face presence (for auto-lock) ----
    try:
        raw_face = face_app.predict_landmarks_from_image(frame_rgb, raw_output=True)
        # raw_face = (boxes, keypoints, roi_corners, landmarks)
        face_present = raw_face[0][0].nelement() > 0
    except Exception:
        face_present = True  # fail safe: never lock on a model hiccup

    if face_present:
        last_face_seen = now
        locked_already = False
    elif not locked_already and (now - last_face_seen) > FACE_ABSENCE_LOCK_SEC:
        print(
            f"[{time.strftime('%H:%M:%S')}] No face for "
            f"{FACE_ABSENCE_LOCK_SEC}s -> Locking screen"
        )
        lock_windows()
        locked_already = True

    # ---- On-screen overlay ----
    overlay = frame_bgr.copy()
    status = gesture if gesture else "No gesture"
    face_status = "Face: present" if face_present else "Face: absent"
    cv2.putText(
        overlay, f"Gesture: {status}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.putText(
        overlay, face_status, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    cv2.imshow("Touchless Laptop Control", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
