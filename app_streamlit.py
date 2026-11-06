"""
Real-Time Driver Drowsiness Detection Dashboard
Built with Streamlit, OpenCV, and SQLite.
Features:
- Live Webcam Feed with Facial Landmark HUD
- Offline Video File Upload for recorded driving analysis
- Real-Time EAR, MAR, Blink Rate, and Head Pose Gauges
- Red Emergency Audio-Visual Alarm on Severe Drowsiness
- SQLite Detection Event Table with CSV Download
"""

import streamlit as st
import cv2
import time
import tempfile
import pandas as pd
from datetime import datetime
from landmarks_detector import FacialLandmarkDetector
from database import DrowsinessDatabase

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Automotive Dark Dashboard Styling
st.markdown("""
<style>
    .reportview-container { background: #0b0f17; }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .state-alert { color: #10b981; font-weight: 700; font-size: 24px; }
    .state-mild { color: #f59e0b; font-weight: 700; font-size: 24px; }
    .state-severe { color: #ef4444; font-weight: 700; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State & Database
db = DrowsinessDatabase()
detector = FacialLandmarkDetector()

# Sidebar: Controls & Thresholds
st.sidebar.title("Configuration & Telemetry")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Upload Recorded Video", "Telemetry Database"])

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Thresholds")
ear_threshold = st.sidebar.slider("Eye Aspect Ratio (EAR) Threshold", 0.15, 0.35, 0.23, 0.01)
mar_threshold = st.sidebar.slider("Mouth Aspect Ratio (MAR) Yawn Threshold", 0.40, 0.85, 0.60, 0.02)
head_nod_threshold = st.sidebar.slider("Head Nod Pitch Threshold (deg)", 10.0, 30.0, 18.0, 1.0)
closure_seconds_threshold = st.sidebar.slider("Micro-Sleep Alarm Trigger (Sec)", 1.0, 3.5, 1.5, 0.1)

st.sidebar.markdown("---")
enable_alarm = st.sidebar.checkbox("Enable Acoustic Emergency Alarm", value=True)
show_landmarks = st.sidebar.checkbox("Overlay 468 Facial Landmarks", value=True)

def play_audio_alarm():
    """Plays alarm audio sound in Streamlit via HTML5 audio tag"""
    alarm_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    </audio>
    """
    st.components.v1.html(alarm_html, height=0)

# Main Dashboard View
st.title("🚗 Real-Time Driver Drowsiness Detection System")
st.caption("Multimodal Fatigue Monitoring with Facial Landmarks (EAR, MAR, Head Pose) & Deep Learning")

if mode in ["Live Webcam", "Upload Recorded Video"]:
    col_video, col_telemetry = st.columns([7, 5])

    with col_telemetry:
        st.subheader("Driver State Telemetry")
        state_placeholder = st.empty()
        m_col1, m_col2 = st.columns(2)
        ear_metric = m_col1.empty()
        mar_metric = m_col2.empty()

        m_col3, m_col4 = st.columns(2)
        pitch_metric = m_col3.empty()
        blinks_metric = m_col4.empty()

        alert_banner = st.empty()
        chart_placeholder = st.empty()

    with col_video:
        video_placeholder = st.empty()
        status_text = st.empty()

    # Video Source Setup
    cap = None
    if mode == "Live Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_video = st.file_uploader("Upload Driver Dashcam / Video File (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

    if cap and cap.isOpened():
        ear_history = []
        mar_history = []
        blink_count = 0
        eyes_closed_start = None

        stop_button = st.button("Stop Video Processing")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Video stream ended or camera disconnected.")
                break

            frame = cv2.flip(frame, 1) if mode == "Live Webcam" else frame
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, pixel_lm = detector.extract_landmarks(rgb)

            ear = 0.32
            mar = 0.25
            pitch, yaw, roll = 0.0, 0.0, 0.0
            driver_state = "Alert"
            alarm_active = False

            if pixel_lm:
                left_eye = [pixel_lm[i] for i in detector.LEFT_EYE_IDX]
                right_eye = [pixel_lm[i] for i in detector.RIGHT_EYE_IDX]
                mouth = [pixel_lm[i] for i in detector.MOUTH_IDX]

                ear = (detector.calculate_ear(left_eye) + detector.calculate_ear(right_eye)) / 2.0
                mar = detector.calculate_mar(mouth)
                pitch, yaw, roll = detector.estimate_head_pose(pixel_lm, w, h)

                # Eye closure tracking
                now = time.time()
                if ear < ear_threshold:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    closure_duration = now - eyes_closed_start
                else:
                    if eyes_closed_start is not None:
                        if (now - eyes_closed_start) < 0.6:
                            blink_count += 1
                    eyes_closed_start = None
                    closure_duration = 0.0

                # Classification Logic
                if closure_duration >= closure_seconds_threshold or pitch >= head_nod_threshold:
                    driver_state = "Severe Drowsiness"
                    alarm_active = True
                elif mar >= mar_threshold or ear < ear_threshold:
                    driver_state = "Mild Drowsiness"
                else:
                    driver_state = "Alert"

                # Draw Visuals on Frame
                if show_landmarks:
                    for pt in left_eye + right_eye:
                        cv2.circle(rgb, (pt[0], pt[1]), 2, (6, 182, 212), -1)
                    for pt in mouth:
                        cv2.circle(rgb, (pt[0], pt[1]), 2, (236, 72, 153), -1)

            # Update Telemetry Display
            state_color = "state-alert" if driver_state == "Alert" else ("state-mild" if driver_state == "Mild Drowsiness" else "state-severe")
            state_placeholder.markdown(f'<div class="metric-card"><p>DRIVER STATE</p><h2 class="{state_color}">{driver_state.upper()}</h2></div>', unsafe_allow_html=True)

            ear_metric.metric("Eye Aspect Ratio (EAR)", f"{ear:.3f}", delta=f"{(ear - ear_threshold):.3f}")
            mar_metric.metric("Mouth Aspect Ratio (MAR)", f"{mar:.3f}", delta=f"{(mar - mar_threshold):.3f}")
            pitch_metric.metric("Head Pitch (Nod)", f"{pitch:.1f}°")
            blinks_metric.metric("Blink Count", f"{blink_count}")

            if alarm_active:
                alert_banner.error("🚨 EMERGENCY: DRIVER DROWSINESS DETECTED! WAKE UP!")
                if enable_alarm:
                    play_audio_alarm()
            else:
                alert_banner.empty()

            # Record to database if state is non-alert
            if driver_state != "Alert":
                db.log_event(driver_state, ear, mar, pitch, blink_count, 0.0, 92.0, alarm_active)

            # Historical Chart
            ear_history.append(ear)
            mar_history.append(mar)
            if len(ear_history) > 50:
                ear_history.pop(0)
                mar_history.pop(0)

            df_chart = pd.DataFrame({"EAR": ear_history, "MAR": mar_history})
            chart_placeholder.line_chart(df_chart)

            video_placeholder.image(rgb, channels="RGB", use_container_width=True)
            time.sleep(0.03)

        cap.release()

elif mode == "Telemetry Database":
    st.subheader("Historical Drowsiness Incident Records")
    stats = db.get_summary_statistics()

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Scanned Events", stats['total_events'])
    sc2.metric("Severe Drowsiness Events", stats['severe_events'])
    sc3.metric("Emergency Alarms Fired", stats['alarms_triggered'])
    sc4.metric("Average EAR", f"{stats['avg_ear']:.3f}")

    recent_logs = db.fetch_recent_logs(limit=100)
    if recent_logs:
        df_logs = pd.DataFrame(recent_logs)
        st.dataframe(df_logs, use_container_width=True)

        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Telemetry Log (CSV)",
            data=csv_data,
            file_name=f"drowsiness_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No recorded detection events yet. Start the webcam or upload a video to log events.")
