import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="Artifact Authenticity App", layout="wide")

st.title("Artifact Authenticity Scanner")
st.write(
    "Upload a short video of your artefact. The program analyses the dominant colours "
    "to decide whether it's **REAL (red)**, **FAKE (blue)**, or **INCONCLUSIVE**. "
    "All your results will be saved in a gallery below."
)

# -----------------------------
# Session State (Gallery)
# -----------------------------
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # list of dicts: each = one analysis result

# -----------------------------
# Video Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a video of the artefact (mp4, mov, avi):",
    type=["mp4", "mov", "avi"]
)

# -----------------------------
# Analysis Function
# -----------------------------
def analyse_video(video_path, frame_step=8, resize_to=(64, 64)):
    """Analyse dominant colours in the video and classify authenticity."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    frame_idx = 0
    avg_colors = []
    counts = {"red": 0, "blue": 0, "other": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            small = cv2.resize(frame, resize_to)
            mean_bgr = small.mean(axis=0).mean(axis=0)
            b, g, r = mean_bgr

            if (r > b) and (r > g):
                counts["red"] += 1
            elif (b > r) and (b > g):
                counts["blue"] += 1
            else:
                counts["other"] += 1

            avg_colors.append(mean_bgr)

        frame_idx += 1

    cap.release()

    if not avg_colors:
        raise RuntimeError("No frames processed. Try a shorter or clearer video.")

    avg_bgr = np.mean(np.array(avg_colors), axis=0)
    b_avg, g_avg, r_avg = avg_bgr

    # Decision based on majority
    majority = max(counts, key=lambda k: counts[k])
    if majority == "red":
        decision = "REAL (Red tones dominant)"
    elif majority == "blue":
        decision = "FAKE (Blue tones dominant)"
    else:
        if (r_avg > b_avg) and (r_avg > g_avg):
            decision = "REAL (Red tones on average)"
        elif (b_avg > r_avg) and (b_avg > g_avg):
            decision = "FAKE (Blue tones on average)"
        else:
            decision = "INCONCLUSIVE"

    confidence = counts[majority] / sum(counts.values())

    return {
        "avg_bgr": avg_bgr,
        "counts": counts,
        "decision": decision,
        "confidence": confidence
    }

# -----------------------------
# Process Uploaded Video
# -----------------------------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.info("Processing video, please wait...")
    try:
        result = analyse_video(temp_video_path)
        bgr = result["avg_bgr"]
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        decision = result["decision"]
        confidence = f"{result['confidence']*100:.1f}%"

        # Create visual colour patch
        patch = np.zeros((100, 300, 3), dtype=np.uint8)
        patch[:] = [rgb[2], rgb[1], rgb[0]]

        # Display result
        st.subheader("Analysis Result")
        st.image(patch, caption=f"Average Colour: {rgb}")
        st.write(f"**Decision:** {decision}")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Frame Breakdown:** {result['counts']}")

        # Save to gallery
        st.session_state.gallery.append({
            "filename": uploaded_file.name,
            "decision": decision,
            "confidence": confidence,
            "colour_patch": patch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        os.unlink(temp_video_path)

# -----------------------------
# Gallery Display
# -----------------------------
st.markdown("---")
st.header("Gallery of Previous Analyses")

if len(st.session_state.gallery) == 0:
    st.info("No videos analysed yet. Upload one above to start your gallery!")
else:
    for item in reversed(st.session_state.gallery):  # newest first
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            st.image(item["colour_patch"], caption=item["timestamp"], use_column_width=True)
        with col2:
            st.markdown(f"**File:** {item['filename']}")
            st.markdown(f"**Decision:** {item['decision']}")
            st.markdown(f"**Confidence:** {item['confidence']}")
        with col3:
            st.markdown("**Details:**")
            st.write("Stored in this session only (cleared on refresh).")
    st.markdown("---")
