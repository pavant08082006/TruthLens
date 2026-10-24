import streamlit as st
import requests, base64, json, io, zipfile
from PIL import Image

st.set_page_config(page_title="TruthLens | Local Media Check", layout="wide")
st.title("🛡️ TruthLens — Local Media Forensics")
st.caption("Local MVP • image and MP4 analysis • results are indicative, not proof")
api_url = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
uploaded = st.file_uploader("Upload a JPG, PNG, or short MP4", type=["jpg", "png", "mp4"])

def decode_png(encoded):
    return Image.open(io.BytesIO(base64.b64decode(encoded)))

if uploaded:
    if uploaded.type.startswith("image/"):
        st.image(uploaded, caption="Original input", width=480)
    else:
        st.info(f"Selected video: {uploaded.name} ({uploaded.size / 1024 / 1024:.1f} MB)")
    if st.button("Analyze media", type="primary"):
        with st.spinner("Analyzing locally…"):
            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/api/v1/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=300
                )
                response.raise_for_status()
                report = response.json()
                st.session_state["report"] = report
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}. Start it with: uvicorn main:app --host 127.0.0.1 --port 8000")

if "report" in st.session_state:
    report = st.session_state["report"]
    verdict = report.get("verdict", "inconclusive")
    confidence = report.get("overall_confidence")
    if verdict == "inconclusive":
        st.warning("INCONCLUSIVE — no trained deepfake detector is installed. ELA alone cannot prove a file is fake.")
    else:
        st.success(f"{verdict.replace('_', ' ').title()} — confidence: {confidence:.0%}")
    st.write(f"**Media:** {report['media_type']} · **Frames analyzed:** {report['sampled_frames']}")
    for i, frame in enumerate(report.get("frame_results", [])):
        st.subheader(f"Frame at {frame['second']} second(s)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if report["media_type"] == "image":
                st.image(uploaded.getvalue() if uploaded and uploaded.type.startswith("image/") else b"", caption="Original")
            else:
                st.caption("Original frame available in backend sampling; see report JSON.")
        with c2:
            st.image(decode_png(frame["visuals"]["gradcam_png_base64"]), caption="Face boxes / model visualization")
        with c3:
            st.image(decode_png(frame["visuals"]["ela_png_base64"]), caption="ELA map")
        st.markdown("**Evidence**")
        st.write(f"- Model status: `{frame['model_status']}`")
        st.write(f"- Faces detected: {frame['face_count']}")
        st.write(f"- ELA mean pixel difference: {frame['forensic_analysis']['ela_mean_difference']:.2f}")
        st.write(f"- ELA note: {frame['forensic_analysis']['notes']}")
        if frame["ai_detection_score"] is None:
            st.write("- AI/deepfake score: unavailable (no trained detector weights)")
        else:
            st.write(f"- AI/deepfake score: {frame['ai_detection_score']:.1%}")

    st.caption(report["forensic_analysis"]["limitations"])
    # Export JSON and visual maps together.
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", json.dumps(report, indent=2))
        for i, frame in enumerate(report.get("frame_results", [])):
            zf.writestr(f"frame_{i}_ela.png", base64.b64decode(frame["visuals"]["ela_png_base64"]))
            zf.writestr(f"frame_{i}_visualization.png", base64.b64decode(frame["visuals"]["gradcam_png_base64"]))
    st.download_button("Download Full Report", data=archive.getvalue(),
                       file_name="truthlens_report.zip", mime="application/zip")
