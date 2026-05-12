"""
Sentinel – Streamlit Dashboard
Jose and Aylin
DSC 333 Final Project · Spring 2026
==============================
Note, to run use:
    streamlit run sentinel.py

Requires sentinel_api.py and sentinel_detect.py to be running.

To run sentinel_api.py:
    uvicorn sentinel_api:app --port 8000

To run sentinel_detect.py:
    python3 sentinel_detect.py

To make things simpler, use the start.sh script to start up
everything at once. It handles the virtual enviroment as well.
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import cv2
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
API = os.getenv("API_BASE_URL", "http://localhost:8000")

# Set GCP credentials env var on startup if the file exists
_creds_path = os.path.abspath(os.path.join("secrets", "gcp_credentials.json"))
if os.path.exists(_creds_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# GLOBAL STYLESHEET
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161922;
    --border:  #2a2d3a;
    --accent:  #00e5ff;
    --accent2: #ff4d6d;
    --green:   #00e676;
    --yellow:  #ffd600;
    --text:    #e8eaf0;
    --muted:   #6b7280;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; font-size: 2rem !important; }
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease;
}
.stButton > button:hover { background: var(--accent) !important; color: #000 !important; }
.danger-btn > button { border-color: var(--accent2) !important; color: var(--accent2) !important; }
.danger-btn > button:hover { background: var(--accent2) !important; color: #fff !important; }
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
hr { border-color: var(--border) !important; margin: 1.5rem 0; }
.page-header {
    font-family: var(--mono);
    font-size: 1.4rem;
    color: var(--accent);
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 1.5rem;
}
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-family: var(--mono); font-weight: 700; letter-spacing: 0.05em; }
.badge-green  { background: #00e67622; color: #00e676; border: 1px solid #00e67655; }
.badge-red    { background: #ff4d6d22; color: #ff4d6d; border: 1px solid #ff4d6d55; }
.badge-yellow { background: #ffd60022; color: #ffd600; border: 1px solid #ffd60055; }
.badge-blue   { background: #00e5ff22; color: #00e5ff; border: 1px solid #00e5ff55; }
.zone-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
}
.zone-card h4 { margin: 0 0 4px 0; font-family: var(--mono); color: var(--accent); font-size: 0.95rem; }
.zone-card p  { margin: 0; font-size: 0.82rem; color: var(--muted); }
.info-box {
    background: #00e5ff0d; border: 1px solid #00e5ff33;
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.88rem; color: var(--text); margin-bottom: 16px;
}
.error-box {
    background: #ff4d6d0d; border: 1px solid #ff4d6d44;
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.88rem; color: var(--text); margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# API HELPERS
def api_get(path, params=None):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_post(path, body=None):
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_put(path, body=None):
    try:
        r = requests.put(f"{API}{path}", json=body, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_delete(path):
    try:
        r = requests.delete(f"{API}{path}", timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

def fetch_summary():
    return api_get("/summary") or {"events_today": 0, "high_alerts": 0, "zone_count": 0, "camera_count": 0}

def fetch_events(zone_id=None, detection_type=None, alert_level=None, limit=200):
    params = {"limit": limit}
    if zone_id:        params["zone_id"]        = zone_id
    if detection_type: params["detection_type"] = detection_type
    if alert_level:    params["alert_level"]    = alert_level
    return api_get("/events", params=params) or []

def fetch_zones(camera_id=None):
    params = {}
    if camera_id: params["camera_id"] = camera_id
    return api_get("/zones", params=params) or []

def fetch_cameras():
    return api_get("/cameras") or []

def insert_zone(name, alert_level, x1, y1, x2, y2, camera_id=None):
    return api_post("/zones", {"name": name, "alert_level": alert_level,
                               "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "camera_id": camera_id})

def delete_zone(zone_id):
    return api_delete(f"/zones/{zone_id}")

def insert_camera(name, stream_url, location):
    return api_post("/cameras", {"name": name, "stream_url": stream_url, "location": location})

def update_camera(camera_id, name, stream_url, location):
    return api_put(f"/cameras/{camera_id}", {"name": name, "stream_url": stream_url, "location": location})

def delete_camera(camera_id):
    return api_delete(f"/cameras/{camera_id}")

def login(username, password):
    return api_post("/auth/login", {"username": username, "password": password})

def has_users():
    result = api_get("/auth/has_users")
    return result.get("has_users", False) if result else False

def fetch_users():
    return api_get("/users") or []

def create_user(username, password, role):
    return api_post("/users", {"username": username, "password": password, "role": role})

def delete_user(user_id):
    return api_delete(f"/users/{user_id}")

def setup_admin(username, password):
    return api_post("/auth/setup", {"username": username, "password": password})

# API HEALTH CHECK
api_ok    = False
api_error = ""
try:
    health = requests.get(f"{API}/health", timeout=3)
    api_ok = health.status_code == 200
except Exception as e:
    api_error = str(e)

ALERT_BADGE = {
    "HIGH":   '<span class="badge badge-red">HIGH</span>',
    "MEDIUM": '<span class="badge badge-yellow">MEDIUM</span>',
    "LOW":    '<span class="badge badge-green">LOW</span>',
}

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 24px 0;">
        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:#00e5ff;letter-spacing:0.08em;">Sentinel</div>
        <div style="font-size:0.75rem;color:#6b7280;margin-top:4px;font-family:'Space Mono',monospace;">DSC 333 · Spring 2026</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["Live Feed", "Event History", "Smart Zones", "Cameras", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System Status**")

    # API status
    if api_ok:
        st.markdown('<span class="badge badge-green">● API CONNECTED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● API OFFLINE</span>', unsafe_allow_html=True)

    # Camera status
    if api_ok:
        cams = fetch_cameras()
        if cams:
            st.markdown(f'<span class="badge badge-green">● {len(cams)} Camera(s) Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">● No Cameras Added</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● Camera · API Offline</span>', unsafe_allow_html=True)

    # GCP Vision status
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and os.path.exists(creds_path):
        st.markdown('<span class="badge badge-green">● GCP Vision Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● GCP Vision Not Configured</span>', unsafe_allow_html=True)

    st.markdown("---")
    if api_ok:
        if st.button("🌱 Seed Sample Data"):
            result = api_post("/seed")
            if result and result.get("seeded"):
                st.success("Sample data seeded!")
                st.rerun()
            else:
                st.info("DB already has data.")

    st.markdown(f"<span style='font-size:0.75rem;color:#6b7280;'>Updated: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)

    if st.session_state.get("logged_in"):
        st.markdown("---")
        st.markdown(f"<span style='font-size:0.8rem;color:#6b7280;'>Logged in as <b style='color:#e8eaf0;'>{st.session_state.get('username','')}</b> ({st.session_state.get('role','')})</span>", unsafe_allow_html=True)
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# API OFFLINE GUARD
if not api_ok:
    st.markdown(f"""
    <div class="error-box">
        ⚠️ <b>Could not connect to the Sentinel API.</b><br>
        Make sure <code>sentinel_api.py</code> is running on <code>{API}</code>.<br>
        Run: <code>uvicorn sentinel_api:app --port 8000</code><br>
        <span style="color:#6b7280;font-size:0.82rem;">{api_error}</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# AUTH GATE
if not st.session_state.get("logged_in"):

    # First time setup — no users exist yet
    if not has_users():
        st.markdown('<div class="page-header">// FIRST TIME SETUP</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            👋 Welcome to Sentinel. No users exist yet — create your admin account to get started.
        </div>
        """, unsafe_allow_html=True)
        with st.form("setup_form"):
            su_user = st.text_input("Admin Username")
            su_pass = st.text_input("Password",        type="password")
            su_conf = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("🚀 Create Admin Account", use_container_width=True):
                if not su_user.strip() or not su_pass:
                    st.error("Username and password cannot be empty.")
                elif su_pass != su_conf:
                    st.error("Passwords do not match.")
                elif len(su_pass) < 5:
                    st.error("Password must be at least 5 characters.")
                else:
                    result = setup_admin(su_user.strip(), su_pass)
                    if result:
                        st.success("Admin account created! You can now log in.")
                        st.rerun()
                    else:
                        st.error("Setup failed. Make sure the API is running.")
        st.stop()

    # Normal login
    st.markdown('<div class="page-header">// LOGIN</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("🔐 Log In", use_container_width=True):
            if not username.strip() or not password:
                st.error("Please enter your username and password.")
            else:
                result = login(username.strip(), password)
                if result and result.get("success"):
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = result["username"]
                    st.session_state["role"]      = result["role"]
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    st.stop()

# PAGE: LIVE FEED
if page == "Live Feed":
    st.markdown('<div class="page-header">// LIVE FEED</div>', unsafe_allow_html=True)

    summary = fetch_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Events Today", summary["events_today"])
    c2.metric("High Alerts",  summary["high_alerts"])
    c3.metric("Active Zones", summary["zone_count"])

    st.markdown("---")
    feed_col, info_col = st.columns([3, 1])

    with feed_col:
        st.markdown("**Camera Feed**")
        cameras = fetch_cameras()
        if cameras:
            grid = st.columns(2)
            for i, cam in enumerate(cameras):
                with grid[i % 2]:
                    location_label = f" — {cam['location']}" if cam["location"] else ""
                    st.markdown(f"<span style='font-family:var(--mono);color:var(--accent);font-size:0.82rem;'>{cam['name']}{location_label}</span>", unsafe_allow_html=True)
                    st.image(cam["stream_url"], width='stretch')
        else:
            st.markdown("""
            <div style="background:#161922;border:1px solid #2a2d3a;border-radius:10px;
                        height:380px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        color:#6b7280;font-family:'Space Mono',monospace;
                        font-size:0.85rem;letter-spacing:0.05em;">
                <div style="font-size:3rem;margin-bottom:12px;">📷</div>
                <div>CAMERA FEED</div>
                <div style="font-size:0.7rem;margin-top:6px;color:#444;">No cameras configured yet</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Go to the Cameras page to add a stream.")

    with info_col:
        st.markdown("**Recent Events**")
        recent = fetch_events(limit=6)
        if recent:
            for row in recent:
                color = "#ff4d6d" if row["alert_level"] == "HIGH" else (
                        "#ffd600" if row["alert_level"] == "MEDIUM" else "#00e676")
                # detected_at comes as a string from the API
                ts_str     = row["detected_at"][11:19] if row["detected_at"] else "—"
                zone_label = row["zone"] or "Unknown Zone"
                st.markdown(f"""
                <div style="border-left:3px solid {color};padding:8px 12px;margin-bottom:8px;
                            background:#161922;border-radius:0 6px 6px 0;font-size:0.8rem;">
                    <div style="color:{color};font-family:'Space Mono',monospace;font-size:0.7rem;">{row['alert_level']}</div>
                    <div style="color:#e8eaf0;">{row['detection_type']}</div>
                    <div style="color:#6b7280;font-size:0.72rem;">{zone_label}</div>
                    <div style="color:#444;font-size:0.68rem;">{ts_str}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No events yet. Use 🌱 Seed Sample Data in the sidebar.")

# PAGE: EVENT HISTORY
elif page == "Event History":
    st.markdown('<div class="page-header">// EVENT HISTORY</div>', unsafe_allow_html=True)

    zones = fetch_zones()
    zone_options = {"All Zones": None} | {z["name"]: z["id"] for z in zones}

    f1, f2, f3 = st.columns(3)
    with f1:
        zone_sel  = st.selectbox("Filter by Zone",  list(zone_options.keys()))
    with f2:
        type_sel  = st.selectbox("Filter by Type",  ["All Types", "Person Detected", "Motion Only"])
    with f3:
        alert_sel = st.selectbox("Filter by Alert", ["All Levels", "HIGH", "MEDIUM", "LOW"])

    events = fetch_events(
        zone_id        = zone_options[zone_sel],
        detection_type = None if type_sel  == "All Types"  else type_sel,
        alert_level    = None if alert_sel == "All Levels" else alert_sel,
    )

    st.markdown("---")
    st.caption(f"Showing {len(events)} events")

    if events:
        df = pd.DataFrame(events)
        df = df.rename(columns={
            "id": "ID", "detected_at": "Timestamp", "zone": "Zone",
            "detection_type": "Type", "alert_level": "Alert", "snapshot_path": "Snapshot",
        })
        df["Snapshot"] = df["Snapshot"].apply(lambda x: "📷" if x and str(x) != "nan" else "—")
        st.dataframe(
            df[["ID", "Timestamp", "Zone", "Type", "Alert", "Snapshot"]],
            width='stretch',
            hide_index=True,
        )
        st.markdown("---")
        st.download_button("⬇ Download CSV", data=df.to_csv(index=False),
                           file_name="sentinel_events.csv", mime="text/csv")

        # Snapshot Viewer
        st.markdown("---")
        st.markdown("**Snapshot Viewer**")

        # Build list of events that actually have a snapshot file
        snapped = [
            row for row in events
            if row.get("snapshot_path") and os.path.exists(row["snapshot_path"])
        ]

        if snapped:
            options = {
                f"Event {row['id']} — {row['detected_at'][0:19]} — {row['zone'] or 'No Zone'}": row
                for row in snapped
            }
            selected_label = st.selectbox("Select an event to view snapshot", list(options.keys()))
            selected_event = options[selected_label]

            snap_col, info_col = st.columns([2, 1])
            with snap_col:
                st.image(selected_event["snapshot_path"], width='stretch')
            with info_col:
                alert = selected_event["alert_level"]
                badge = ALERT_BADGE.get(alert, "")
                st.markdown(f"""
                <div class="zone-card" style="margin-top:0;">
                    <h4>Event {selected_event['id']}</h4>
                    <p style="margin-top:6px;">{selected_event['detection_type']}</p>
                    <p style="margin-top:6px;">Alert: {badge}</p>
                    <p style="margin-top:6px;">Zone: <b style="color:#e8eaf0">{selected_event['zone'] or '—'}</b></p>
                    <p style="margin-top:6px;">Time: <b style="color:#e8eaf0">{selected_event['detected_at'][0:19]}</b></p>
                    <p style="margin-top:6px;font-size:0.75rem;word-break:break-all;">{selected_event['snapshot_path']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                No snapshots available yet. Snapshots are saved automatically when
                <code>sentinel_detect.py</code> detects a person.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No events match the selected filters.")

# PAGE: SMART ZONES
elif page == "Smart Zones":
    st.markdown('<div class="page-header">// SMART ZONES</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        📌 Zones define regions in the camera frame. Select a camera to preview its stream
        and configure zones for it. Coordinates are <b>normalized (0.0–1.0)</b> relative to frame width/height.
    </div>
    """, unsafe_allow_html=True)

    cameras = fetch_cameras()
    if not cameras:
        st.warning("No cameras configured yet. Go to the Cameras page to add one first.")
    else:
        cam_options       = {c["name"]: c["id"] for c in cameras}
        selected_cam_name = st.selectbox("Select Camera", list(cam_options.keys()))
        selected_cam_id   = cam_options[selected_cam_name]
        selected_cam_url  = next(c["stream_url"] for c in cameras if c["id"] == selected_cam_id)

        st.markdown("---")
        zones_col, form_col = st.columns([1.4, 1])

        with zones_col:
            st.markdown(f"**Preview — {selected_cam_name}**")
            try:
                st.image(selected_cam_url, width='stretch')
            except Exception:
                st.markdown("""
                <div style="background:#161922;border:1px solid #2a2d3a;border-radius:10px;
                            height:200px;display:flex;align-items:center;justify-content:center;
                            color:#6b7280;font-family:'Space Mono',monospace;font-size:0.8rem;">
                    Stream unavailable
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Zones for {selected_cam_name}**")
            zones = fetch_zones(camera_id=selected_cam_id)
            if zones:
                for z in zones:
                    badge  = ALERT_BADGE.get(z["alert_level"], "")
                    coords = f"({z['x1']:.2f}, {z['y1']:.2f}) → ({z['x2']:.2f}, {z['y2']:.2f})"
                    st.markdown(f"""
                    <div class="zone-card">
                        <h4>{z['name']}</h4>
                        <p>Coordinates: <code style="color:#00e5ff">{coords}</code></p>
                        <p style="margin-top:6px;">Alert Level: {badge} &nbsp;·&nbsp; Events: <b style="color:#e8eaf0">{z['event_count']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                    if st.button(f"🗑 Delete '{z['name']}'", key=f"del_{z['id']}"):
                        delete_zone(z["id"])
                        st.success(f"Zone '{z['name']}' deleted.")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption(f"No zones for {selected_cam_name} yet. Add one using the form.")

        with form_col:
            st.markdown("**Add New Zone**")

            zone_name = st.text_input("Zone Name", placeholder="e.g. Front Door")
            alert_lvl = st.selectbox("Alert Level", ["HIGH", "MEDIUM", "LOW"])

            st.markdown("**Bounding Box** *(0.0 – 1.0)*")
            x1 = st.slider("X1 (left)",   0.0, 1.0, 0.0, 0.01, key="zx1")
            y1 = st.slider("Y1 (top)",    0.0, 1.0, 0.0, 0.01, key="zy1")
            x2 = st.slider("X2 (right)",  0.0, 1.0, 0.5, 0.01, key="zx2")
            y2 = st.slider("Y2 (bottom)", 0.0, 1.0, 0.5, 0.01, key="zy2")

            # Zone Preview
            st.markdown("**Zone Preview**")
            if st.button("📸 Capture Frame", key="capture_preview"):
                frame_bytes = None
                try:
                    resp = requests.get(selected_cam_url, stream=True, timeout=5)
                    buf  = b""
                    for chunk in resp.iter_content(chunk_size=4096):
                        buf += chunk
                        s = buf.find(b"\xff\xd8")
                        e = buf.find(b"\xff\xd9")
                        if s != -1 and e != -1 and e > s:
                            frame_bytes = buf[s:e + 2]
                            break
                except Exception:
                    pass
                if frame_bytes:
                    st.session_state["preview_frame"] = frame_bytes
                else:
                    st.error("Could not grab frame. Is the stream running?")

            if "preview_frame" in st.session_state:
                frame_arr = np.frombuffer(st.session_state["preview_frame"], np.uint8)
                frame_bgr = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    h, w = frame_bgr.shape[:2]

                    # Draw existing zones in grey
                    for z in fetch_zones(camera_id=selected_cam_id):
                        px1 = int(z["x1"] * w); py1 = int(z["y1"] * h)
                        px2 = int(z["x2"] * w); py2 = int(z["y2"] * h)
                        cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (100, 100, 100), 1)
                        cv2.putText(frame_bgr, z["name"], (px1 + 4, py1 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                    # Draw new zone in cyan
                    if x2 > x1 and y2 > y1:
                        nx1 = int(x1 * w); ny1 = int(y1 * h)
                        nx2 = int(x2 * w); ny2 = int(y2 * h)
                        cv2.rectangle(frame_bgr, (nx1, ny1), (nx2, ny2), (0, 229, 255), 2)
                        label = zone_name.strip() if zone_name.strip() else "New Zone"
                        cv2.putText(frame_bgr, label, (nx1 + 4, ny1 + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

                    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.markdown("""
                <div style="background:#161922;border:1px solid #2a2d3a;border-radius:8px;
                            height:140px;display:flex;align-items:center;justify-content:center;
                            color:#6b7280;font-family:'Space Mono',monospace;font-size:0.75rem;">
                    Press 📸 Capture Frame to preview
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            if st.button("➕ Add Zone", key="add_zone_btn", width='stretch'):
                if not zone_name.strip():
                    st.error("Zone name cannot be empty.")
                elif x2 <= x1 or y2 <= y1:
                    st.error("X2 must be > X1 and Y2 must be > Y1.")
                else:
                    insert_zone(zone_name.strip(), alert_lvl, x1, y1, x2, y2, selected_cam_id)
                    st.session_state.pop("preview_frame", None)
                    st.success(f"Zone '{zone_name}' added to {selected_cam_name}!")
                    st.rerun()

# PAGE: CAMERAS
elif page == "Cameras":
    st.markdown('<div class="page-header">// CAMERAS</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        📷 Add camera streams here. Each stream URL should point to a running
        <code>sentinel_camera.py</code> instance. They will appear as a grid on the Live Feed page.
    </div>
    """, unsafe_allow_html=True)

    cameras_col, form_col = st.columns([1.4, 1])

    with cameras_col:
        st.markdown("**Registered Cameras**")
        cameras = fetch_cameras()
        if cameras:
            for cam in cameras:
                location_label = cam["location"] or "No location set"
                st.markdown(f"""
                <div class="zone-card">
                    <h4>{cam['name']}</h4>
                    <p>Location: <b style="color:#e8eaf0">{location_label}</b></p>
                    <p style="margin-top:4px;">URL: <code style="color:#00e5ff">{cam['stream_url']}</code></p>
                </div>
                """, unsafe_allow_html=True)

                edit_col, del_col = st.columns([1, 1])
                with edit_col:
                    if st.button(f"✏️ Edit '{cam['name']}'", key=f"editcam_{cam['id']}"):
                        st.session_state["editing_cam"] = cam["id"]
                with del_col:
                    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                    if st.button("🗑 Delete", key=f"delcam_{cam['id']}"):
                        delete_camera(cam["id"])
                        st.success(f"Camera '{cam['name']}' removed.")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                # Inline edit form
                if st.session_state.get("editing_cam") == cam["id"]:
                    # Parse existing URL back into IP and port for pre-filling
                    existing_url  = cam["stream_url"]
                    try:
                        stripped  = existing_url.replace("http://", "")
                        host_part = stripped.split("/")[0]
                        ex_ip, ex_port = host_part.rsplit(":", 1)
                    except Exception:
                        ex_ip, ex_port = "", "8080"

                    with st.form(f"edit_cam_form_{cam['id']}", clear_on_submit=True):
                        st.markdown("**Edit Camera**")
                        new_name     = st.text_input("Camera Name", value=cam["name"])
                        new_location = st.text_input("Location",    value=cam["location"] or "")
                        ip_col, port_col = st.columns([3, 1])
                        new_ip   = ip_col.text_input("IP Address", value=ex_ip)
                        new_port = port_col.text_input("Port",     value=ex_port)
                        s1, s2 = st.columns(2)
                        if s1.form_submit_button("💾 Save"):
                            if not new_name.strip() or not new_ip.strip():
                                st.error("Name and IP address cannot be empty.")
                            else:
                                new_url = f"http://{new_ip.strip()}:{new_port.strip()}/stream"
                                update_camera(cam["id"], new_name.strip(), new_url, new_location.strip() or None)
                                st.session_state.pop("editing_cam", None)
                                st.success(f"Camera updated! Stream: {new_url}")
                                st.rerun()
                        if s2.form_submit_button("✖ Cancel"):
                            st.session_state.pop("editing_cam", None)
                            st.rerun()
        else:
            st.caption("No cameras yet. Add one using the form.")

    with form_col:
        st.markdown("**Add New Camera**")
        with st.form("add_camera_form", clear_on_submit=True):
            cam_name     = st.text_input("Camera Name", placeholder="e.g. Front Door Cam")
            cam_location = st.text_input("Location",    placeholder="e.g. Front Porch")
            ip_col, port_col = st.columns([3, 1])
            cam_ip   = ip_col.text_input("IP Address", placeholder="192.168.1.10")
            cam_port = port_col.text_input("Port", value="8080")
            if st.form_submit_button("➕ Add Camera"):
                if not cam_name.strip():
                    st.error("Camera name cannot be empty.")
                elif not cam_ip.strip():
                    st.error("IP address cannot be empty.")
                else:
                    cam_url = f"http://{cam_ip.strip()}:{cam_port.strip()}/stream"
                    insert_camera(cam_name.strip(), cam_url, cam_location.strip() or None)
                    st.success(f"Camera '{cam_name}' added! Stream: {cam_url}")
                    st.rerun()

# PAGE: SETTINGS
elif page == "Settings":
    st.markdown('<div class="page-header">// SETTINGS</div>', unsafe_allow_html=True)

    # API & Database
    st.markdown("**API & Database**")
    st.markdown("""
    <div class="info-box">
        The dashboard connects to the Sentinel API at the URL set in your <code>.env</code> file
        (<code>API_BASE_URL</code>). The API manages all database operations.
    </div>
    """, unsafe_allow_html=True)
    s = fetch_summary()
    if s:
        st.markdown(f'<span class="badge badge-green">● API Connected</span> &nbsp; zones: <b>{s["zone_count"]}</b> &nbsp;·&nbsp; cameras: <b>{s["camera_count"]}</b> &nbsp;·&nbsp; events today: <b>{s["events_today"]}</b>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● API Offline</span>', unsafe_allow_html=True)

    st.markdown("")
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        st.link_button("📖 Swagger UI Docs", f"{API}/docs", width='stretch')
    with doc_col2:
        st.link_button("📄 ReDoc Docs", f"{API}/redoc", width='stretch')

    st.markdown("---")

    # Detection Settings
    st.markdown("**Detection Settings**")
    st.markdown("""
    <div class="info-box">
        These settings are saved to <code>settings.json</code> and read by
        <code>sentinel_detect.py</code> on each detection cycle. Changes take
        effect on the next capture without restarting the script.
    </div>
    """, unsafe_allow_html=True)

    # Load existing settings.json if it exists
    settings_path = "settings.json"
    saved_settings = {}
    if os.path.exists(settings_path):
        import json
        try:
            with open(settings_path) as f:
                saved_settings = json.load(f)
        except Exception:
            saved_settings = {}

    current_interval   = saved_settings.get("capture_interval", int(os.getenv("CAPTURE_INTERVAL", 10)))
    current_confidence = saved_settings.get("confidence_min",   float(os.getenv("CONFIDENCE_MIN", 0.55)))

    c1, c2 = st.columns(2)
    new_interval   = c1.number_input("Capture Interval (seconds)",
                                     min_value=1, max_value=300,
                                     value=current_interval,
                                     help="How often sentinel_detect.py grabs a frame per camera.")
    new_confidence = c2.slider("Detection Confidence (0.0 – 1.0)",
                               min_value=0.0, max_value=1.0,
                               value=float(current_confidence),
                               step=0.01,
                               help="Minimum GCP Vision confidence to count as a real detection.")

    if st.button("💾 Save Detection Settings"):
        import json
        with open(settings_path, "w") as f:
            json.dump({
                "capture_interval": new_interval,
                "confidence_min":   new_confidence,
            }, f, indent=2)
        st.success(f"Settings saved — interval: {new_interval}s, confidence: {new_confidence:.2f}")

    st.markdown("---")

    # GCP Credentials
    st.markdown("**GCP Credentials**")

    creds_path = os.path.join("secrets", "gcp_credentials.json")
    os.makedirs("secrets", exist_ok=True)

    # Set env var for this Streamlit process if the file exists
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(creds_path)

    if os.path.exists(creds_path):
        st.markdown(f'<span class="badge badge-green">● Credentials file found</span> &nbsp; <code>{creds_path}</code>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-red">● Not found</span> &nbsp; Expected at <code>{creds_path}</code>', unsafe_allow_html=True)

    uploaded_creds = st.file_uploader(
        "Upload GCP Service Account JSON",
        type=["json"],
        help="Download from GCP Console → IAM & Admin → Service Accounts → Keys. Any filename is fine — it will be saved as gcp_credentials.json inside the secrets/ folder."
    )

    if uploaded_creds is not None:
        import json
        # Validate that it looks like a GCP service account file before saving
        try:
            parsed = json.loads(uploaded_creds.getvalue())
            required_keys = {"type", "project_id", "private_key", "client_email"}
            if not required_keys.issubset(parsed.keys()):
                st.error("❌ This doesn't look like a valid GCP service account file. Make sure you downloaded the right JSON from GCP.")
            elif parsed.get("type") != "service_account":
                st.error(f"❌ Expected type 'service_account', got '{parsed.get('type')}'. Make sure you're uploading a Service Account key, not a different GCP credential type.")
            else:
                if st.button("💾 Save Credentials"):
                    with open(creds_path, "wb") as f:
                        f.write(uploaded_creds.getvalue())
                    st.success(f"✅ Credentials saved to `{creds_path}` (project: `{parsed.get('project_id')}`, account: `{parsed.get('client_email')}`)")
                    st.rerun()
        except json.JSONDecodeError:
            st.error("❌ Could not parse the file as JSON. Make sure you're uploading the correct file.")

    # Verify credentials button
    if os.path.exists(creds_path):
        if st.button("🔍 Verify GCP Vision Connection"):
            try:
                from google.cloud import vision
                # Must set the env var in this process before initialising the client
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(creds_path)
                client   = vision.ImageAnnotatorClient()
                # Generate a 1x1 white JPEG using cv2 and read it back as bytes
                test_img_path = "/tmp/sentinel_gcp_test.jpg"
                cv2.imwrite(test_img_path, np.ones((1, 1, 3), dtype=np.uint8) * 255)
                with open(test_img_path, "rb") as f:
                    test_img = f.read()
                image = vision.Image(content=test_img)
                client.object_localization(image=image)
                st.success("✅ GCP Vision API connected and working!")
            except Exception as ex:
                st.error(f"❌ GCP Vision test failed: {ex}")

    st.markdown("---")

    # User Management — admin only
    st.markdown("**User Management**")
    if st.session_state.get("role") != "admin":
        st.markdown("""
        <div class="info-box">
            🔒 User management is only available to admin users.
        </div>
        """, unsafe_allow_html=True)
    else:
        users = fetch_users()
        if users:
            for u in users:
                u_col, r_col, d_col = st.columns([2, 1, 1])
                u_col.markdown(f"<span style='font-family:var(--mono);color:#e8eaf0;'>{u['username']}</span>", unsafe_allow_html=True)
                r_col.markdown(f"<span class=\"badge {'badge-blue' if u['role'] == 'admin' else 'badge-green'}\">{u['role']}</span>", unsafe_allow_html=True)
                with d_col:
                    if u["username"] != st.session_state.get("username"):
                        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                        if st.button("🗑", key=f"del_user_{u['id']}"):
                            result = delete_user(u["id"])
                            if result:
                                st.success(f"User '{u['username']}' removed.")
                                st.rerun()
                            else:
                                st.error("Could not delete — cannot remove the last admin.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.caption("(you)")
        else:
            st.caption("No users found.")

        st.markdown("")
        st.markdown("**Add New User**")
        with st.form("add_user_form", clear_on_submit=True):
            new_uname = st.text_input("Username")
            new_upass = st.text_input("Password", type="password")
            new_urole = st.selectbox("Role", ["viewer", "admin"])
            if st.form_submit_button("➕ Add User"):
                if not new_uname.strip() or not new_upass:
                    st.error("Username and password cannot be empty.")
                elif len(new_upass) < 5:
                    st.error("Password must be at least 5 characters.")
                else:
                    result = create_user(new_uname.strip(), new_upass, new_urole)
                    if result:
                        st.success(f"User '{new_uname}' added as {new_urole}.")
                        st.rerun()
                    else:
                        st.error("Username already exists or API error.")

    st.markdown("---")
    st.markdown("**Danger Zone**")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear All Events", width='stretch'):
            api_delete("/events")
            st.success("All event records cleared from database.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Deletes all event rows from the database. Snapshot files are kept.")

    with d2:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear All Zones", width='stretch'):
            zones = fetch_zones()
            for z in zones:
                api_delete(f"/zones/{z['id']}")
            st.success(f"Deleted {len(zones)} zone(s).")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Deletes all configured zones. Events linked to them are kept.")

    with d3:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear All Snapshots", width='stretch'):
            deleted = 0
            if os.path.exists(SNAPSHOT_DIR):
                for fname in os.listdir(SNAPSHOT_DIR):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        try:
                            os.remove(os.path.join(SNAPSHOT_DIR, fname))
                            deleted += 1
                        except Exception:
                            pass
            st.success(f"Deleted {deleted} snapshot file(s) from `{SNAPSHOT_DIR}/`.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Deletes image files from disk. Event records in the database are kept.")

    st.markdown("")
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🗑 Clear Everything", width='stretch'):
        # Clear DB events
        api_delete("/events")
        # Clear snapshot files
        deleted = 0
        if os.path.exists(SNAPSHOT_DIR):
            for fname in os.listdir(SNAPSHOT_DIR):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        os.remove(os.path.join(SNAPSHOT_DIR, fname))
                        deleted += 1
                    except Exception:
                        pass
        st.success(f"All events cleared and {deleted} snapshot file(s) deleted.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Clears both the database event records and all snapshot files on disk.")