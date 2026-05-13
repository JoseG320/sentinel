# Sentinel
**DSC 333 · Spring 2026 · Jose and Aylin**

An AI-powered local home security system that uses camera streams to detect people in real time, checks detections against user-defined smart zones, and logs events to a database. Built with FastAPI, Streamlit, PostgreSQL, and the GCP Vision API.

---

## How It Works

Sentinel runs three services on your main machine:

- **`sentinel_api.py`** — FastAPI backend. The only service that talks to the database. Exposes REST endpoints for cameras, zones, events, and users.
- **`sentinel_detect.py`** — Detection pipeline. Grabs frames from registered camera streams on an interval, sends them to GCP Vision for person detection, checks zone overlap, and writes events to the database.
- **`sentinel.py`** — Streamlit dashboard. The user interface for monitoring live feeds, reviewing event history, managing zones and cameras, and configuring the system.

Camera streams are provided separately by `sentinel_camera.py` (see the camera repository), which can run on any device. Whether it would be a Raspberry Pi or a laptop. All on the same network.

```
Any device (RPI or laptop)          Main Computer
sentinel_camera.py ─────────────► sentinel_detect.py
  (MJPEG stream)                         │
                                   sentinel_api.py
                                         │
                                     PostgreSQL
                                         │
                                   sentinel.py (dashboard)
```

---

## Requirements

- Python 3.10+
- PostgreSQL (local or GCP Cloud SQL)
- A GCP project with the Vision API enabled
- A GCP service account JSON key file

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DSC333-Sentinel/sentinel.git
cd sentinel
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```dotenv
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sentinel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword

# API
API_BASE_URL=http://localhost:8000

# Camera
STREAM_PORT=8080

# Snapshot Directory
SNAPSHOT_DIR=snapshots
```

### 3. Run the start script

```bash
chmod +x start.sh
./start.sh
```

The script will:
- Create a Python virtual environment if one doesn't exist
- Install all dependencies from `requirements.txt`
- Check for your `.env` file
- Create the `secrets/` folder and `settings.json` if missing
- Start the API, detection pipeline, and dashboard

You may also just create the virtual enviroment via Visual Studio Code and still run the script. The script just makes the process easier.

### 4. First time setup

Once the startup script is running, the dashboard should automaticaly open the in your browser:
```
http://localhost:8501
```

You will be prompted to create an admin account on first launch. After that, all subsequent visits require login.

### 5. Upload GCP credentials

Go to **Settings → GCP Credentials** and upload your GCP service account JSON file. The file will be saved to `secrets/gcp_credentials.json` regardless of its original filename. You can verify the connection works using the **Verify GCP Vision Connection** button.

To get your gcp_credentials.json file, refer to this link for more information: https://developers.google.com/workspace/guides/create-credentials#create_credentials_for_a_service_account

When creating the account, make sure you are giving the service account the "Cloud Vision AI Service Agent" role.

### 6. Add a camera

Go to **Cameras → Add New Camera**. Enter the IP address and port of a machine running `sentinel_camera.py`. The stream URL is constructed automatically as `http://<ip>:<port>/stream`.

### 7. Configure smart zones

Go to **Smart Zones**, select a camera, click **Capture Frame**, and use the sliders to draw a bounding box on the frame. Each zone has its own alert level (HIGH, MEDIUM, LOW).

---

## Accessing from other devices

By default the dashboard is accessible from any device on the same WiFi network at:
```
http://<your-machine-ip>:8501
```

For remote access from outside your network, install [Tailscale](https://tailscale.com) on your machine and phone — no router configuration needed.

For Raspberry Pi cameras, you can install [PiTunnel](https://www.pitunnel.com/) to make the camera available on your network or outside of it.

---

## API Documentation

While the app is running, interactive API documentation is available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

Both are also linked from the **Settings** page in the dashboard.

---

## Detection Settings

Detection behaviour is controlled by `settings.json` (created automatically on first run):

```json
{
  "capture_interval": 10,
  "confidence_min": 0.55
}
```

- `capture_interval` — seconds between each detection pass per camera
- `confidence_min` — minimum GCP Vision confidence score (0.0–1.0) to count as a detection

These can be adjusted live from the **Settings** page without restarting the detection script.

---

## Project Structure

```
sentinel/
├── sentinel.py          # Streamlit dashboard
├── sentinel_api.py      # FastAPI backend
├── sentinel_detect.py   # GCP Vision detection pipeline
├── start.sh             # Startup script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your credentials (not committed)
├── settings.json        # Detection settings (auto-created)
├── secrets/             # GCP credentials folder (not committed)
│   └── gcp_credentials.json
└── snapshots/           # Saved detection snapshots (auto-created)
```

---

## User Roles

| Role    | Dashboard | Settings | User Management | Danger Zone |
|---------|-----------|----------|-----------------|-------------|
| Admin   | Yes        | Yes       | Yes              | Yes          |
| Viewer  | Yes        | Yes       | No              | No          |

Admins can add and remove users from the Settings page.

---

## Stopping the App

Press `Ctrl+C` in the terminal running `start.sh`. All three services will stop cleanly and the virtual environment will be deactivated.
