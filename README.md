# Human Detection and Tracking for AGVs
A multithreaded sensor fusion pipeline that combines radar (RD-03D) and AI camera data, fuses the inputs via a decision tree, and feeds the resulting coordinates into a Kalman filter for smooth, predicted position tracking.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [File Descriptions](#file-descriptions)
  - [Kalman Filter.py](#kalman-filterpy)
  - [Sensor Fusion.py](#sensor-fusionpy)
- [Dependencies](#dependencies)
  - [Raspberry Pi OS](#raspberry-pi-os)
  - [AI Camera](#ai-camera)
  - [RD-03D](#rd-03d)
- [Configuration](#configuration)

---

## Overview 
This project implements a real-time human detection and tracking system intended for use on Automated Guided Vehicles (AGVs). The system:

1. Runs two **concurrent threads** — one collecting positional data from an RD-03D radar, another processing detections from a Sony IMX500 AI camera
2. **Fuses the two data sources** by matching radar targets to camera detections using angular alignment, with defined fallback behaviour when one sensor is unavailable
3. Feeds the fused `(x, y)` coordinates into a **Kalman filter** for each of up to three tracked targets, outputting smoothed position predictions only once a target has been **confirmed** over multiple consecutive frames

---

## System Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Raspberry Pi 5                               │
│                                                                         │
│          ┌──────────────────────────────────────────────┐               │
│          │           Sensor Fusion Code System          │               │
│          └───────────────────┬──────────────────────────┘               │
│                              │                                          │
│              ┌───────────────┴───────────────┐                          │
│              │                               │                          │
│   ┌──────────▼──────────┐       ┌────────────▼────────┐                 │
│   │   Camera Software   │       │  UART Serial        │                 │
│   │  • Picamera 2       │       │  Interface          │                 │
│   │  • libcamera        │       └────────────┬────────┘                 │
│   └──────────┬──────────┘                    │                          │
│              │                               │                          │
│   ┌──────────▼──────────┐       ┌────────────▼────────┐                 │
│   │   Linux Drivers     │       │  Linux UART         │                 │
│   │  • IMX500 Driver    │       │  Drivers            │                 │
│   │  • CSI-2 Receiver   │       └────────────┬────────┘                 │
│   └──────────┬──────────┘                    │                          │
└──────────────┼───────────────────────────────┼──────────────────────────┘
               │                               │
    I2C:       │  MPI CSI-2:                   │ UART Serial
  (Camera      │  (Image and AI Tensor)        │
  Configs)     │                               │
               │                               │
┌──────────────┴──────────┐       ┌────────────┴────────────┐
│  Raspberry Pi AI Camera │       │   RD-03D mmWave Radar   │
│                         │       │                         │
│  Outputs:               │       │  Output:                │
│  • Image Frames         │       │  • Int 16 type binary   │
│  • AI Accelerator       │       │    containing all       │
│    Tensor               │       │    target information   │
└─────────────────────────┘       └─────────────────────────┘
```

---

## File Descriptions
### Kalman Filter.py
Contains two classes — `KalmanFilter` and `KalmanTracker` — that together implement a robust, gated Kalman tracking system for a single target.

#### `KalmanFilter`

A standard linear Kalman filter operating on a **4-dimensional state vector** representing position and velocity:

```
State vector x = [x_pos, y_pos, x_vel, y_vel]
```

| Matrix | Dimension | Description |
|---|---|---|
| `F` (State Transition) | 4×4 | Projects current state to next state; `dt` offsets are injected at runtime by `KalmanTracker` |
| `H` (Measurement) | 2×4 | Extracts only `[x, y]` from the state — velocity is unobserved |
| `P` (Covariance) | 4×4 | Initial uncertainty set to `200 × I` |
| `R` (Measurement Noise) | 2×2 | Set to `80 × I` — suppresses the effect of sudden measurement jumps |
| `Q` (Process Noise) | 4×4 | Set to `0.01 × I` — trusts the motion model closely between updates |

**`predict()`** — Advances the state estimate and covariance forward using `F` and `Q`.

**`update(measurement)`** — Takes a `[x, y]` measurement, computes the innovation, derives the Kalman gain `K`, and updates both the state estimate and uncertainty. Includes protection against singular matrix inversion.

#### `KalmanTracker`

A higher-level wrapper around `KalmanFilter` that adds:

- **Time-based `dt`** — Computes elapsed time between calls and injects it into `F[0,2]` and `F[1,3]` so velocity terms correctly propagate position
- **Gating** — After the startup phase, measurements more than **800 mm** from the predicted position are rejected to prevent track corruption from spurious detections. Rather than fully resetting the hit streak on a rejected frame, it is decremented by 1 so a single missed frame does not lose a confirmed track
- **Hit streak confirmation** — A target must accumulate **5 consecutive valid updates** (`threshold = 5`) before `is_confirmed` returns `True` and its smoothed position is trusted by the main system. During the startup phase (streak below threshold), gating is intentionally relaxed to allow large initial position jumps

| Method / Property | Description |
|---|---|
| `update(raw_x, raw_y)` | Runs predict → gate check → Kalman update; returns `(smooth_x, smooth_y)` |
| `is_confirmed` | `True` once hit streak ≥ 5 |
| `reset()` | Clears `initiated` flag and resets hit streak to 0 |

---
### Sensor Fusion.py
The main application file. Spawns two data collection threads, performs sensor fusion, and drives a `KalmanTracker` instance for each of up to three concurrent targets.

#### Shared State

| Variable | Type | Description |
|---|---|---|
| `t_radar` | `dict` (keys 1–3) | Each entry holds `valid`, `x`, `y`, `dist`, `angle`, `time` for one radar target |
| `t_camera` | `list` | List of detection dicts containing `label`, `angle`, `conf` for the current camera frame |
| `camera_lock` | `threading.Lock` | Protects `t_camera` from simultaneous read/write across threads |

#### Thread 1 — `radar_collect()`

Polls the RD-03D radar continuously. For each update cycle, reads up to three targets and populates `t_radar`. A target is only marked `valid` if its distance falls within **350 mm – 7000 mm**. The bearing of each target is computed as:

```
angle = degrees( atan( x / y ) )
```

The thread closes the radar connection safely via a `finally` block even if an exception occurs.

#### Thread 2 — `camera_collect()`

Runs the IMX500 detector and processes each frame. Camera constants:

| Constant | Value | Description |
|---|---|---|
| `WIDTH` | 640 px | Frame width |
| `FOV` | 75° | Horizontal field of view |
| `DEG_PER_PIXEL` | `75 / 640` | Angular resolution per pixel |

Only detections with **confidence > 0.5** are kept. For each qualifying detection the bounding box centre pixel is converted to a horizontal angle relative to the camera centre:

```
offset = center_x - 320
angle  = offset × DEG_PER_PIXEL
```

The resulting list of `{label, angle, conf}` dicts is written to `t_camera` under `camera_lock`.

#### `camera_radar_match(angle, detection_list)`

Attempts to find a camera detection whose angle is within **20°** of a given radar target's angle, filtering only for `"person"` labels. Returns the closest angular match, or `None` if no detection falls within the gate.

#### Sensor Fusion Logic (Main Loop)

Runs at ~10 Hz. For each of the three target slots the fusion logic follows three paths:

```
Radar data recent (< 0.5s old) AND valid?
│
├── YES → use radar y as depth
│         │
│         ├── Camera angle match found within 20°?
│         │   ├── YES → FUSED MODE
│         │   │         x = radar_y × tan(camera_angle)
│         │   │         y = radar_y
│         │   │         (camera angle used — more precise than radar angle)
│         │   │
│         │   └── NO  → RADAR FALLBACK
│         │             x = radar_x
│         │             y = radar_y
│         │
│         └── Store y as last_known_y[i]
│
└── NO  → CAMERA ONLY MODE (Tracker 1 only)
          If any camera detection exists:
            x = last_known_y[i] × tan(camera_angle)
            y = last_known_y[i]   ← depth memory from last radar reading
```

> **Note:** Camera-only fallback is only applied to **Tracker 1**. Trackers 2 and 3 require live radar data and are reset if the radar goes stale.

Once `(raw_x, raw_y)` is resolved it is passed to the corresponding `KalmanTracker`. If the tracker is confirmed (`hit_streak ≥ 5`) the smoothed output is printed to console, distinguishing between `"person"` and inanimate object labels. If no input is found for a slot that tracker is reset.

---

## Dependencies
### Raspberry Pi OS 
The first thing that you need to do is download the Raspberry Pi Imager to allow you to download the Raspberry Pi OS onto an SD card. This can be gotten at the website provided here: [Raspberry Pi Imager](https://www.raspberrypi.com/software/)

When installing the Raspberry Pi Os it is important to choose a version of the OS that is based on the **Bookworm version of Debian**. Bookworm is the codename for the Debian 12 distro and is required for certain libraries for the AI Camera to work correctly. The newest Raspberry Pi OS is based on the Trixie version of Debian which is Debian 13. To download the Bookworm OS go into the Legacy OS tab in the Raspberry Pi Imager and look for the 64-bit OS bookworm based OS. Once the OS is installed on the SD card and booted onto the Raspberry Pi, to check if the installed OS is the correct verion use the code.
```
cat /etc/os-release
```
Once this is done you should see an output that looks similar to the one seen below:
```
PRETTY_NAME="Raspberry Pi OS (64-bit)"
NAME="Raspberry Pi OS"
VERSION_ID="12"
VERSION="12 (bookworm)"
VERSION_CODENAME=bookworm
ID=debian
ID_LIKE=debian
```
If Debian 12 (bookworm) appears in either the `PRETTY_NAME` or `VERSION` section of the code then you have installed the correct version of the OS.

One this is done it is advised to run a quick update to all the libraries on the Pi to make sure everything is up to date. The following line of code updates the local list of packages on the Pi
```
sudo apt update
```
Running the next command upgrades all installed packages to the latest version.
```
sudo apt full-upgrade
```

### AI Camera 
The AI Camera used in this project is a Sony **IMX500** that requires specific packages to be downloaded in order for it to function properly. Run the following code to download all the required IMX500 packages:
```
sudo apt install imx500-all
```
After installing the packages, a reboot is needed before the installed packages can be used. Run the following code to reboot the system.
```
sudo reboot
```
Now that all the libraries are installed, it is best to run a quick test program to ensure that all files have been downloaded properly. The following code will turn on the connected AI camera and display a preview to test if the camera is working properly:
```
rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30
```
After the test has been performed and the camera is working properly the next thing to do is to download the Core Electronics AI_camera library that is used in the vision system. The library can be gotten from the attached link: [AI Camera Library](https://core-electronics.com.au/attachments/uploads/ai-camera-library-demo.zip). When this file is unzipped there are some demo code files as well as the ai_camera.py library file. It is important that this library file is stored in the **same working directory** as the code files that are using it otherwise the code will not run.

### RD-03D
For the radar sensor setup all that was required was to download the required rd03d library. The library is also made by Core Electronics and can be gotten at the attached link: [Rd03d Library](https://core-electronics.com.au/attachments/uploads/rpi_mmwave.zip). Similarly to the AI camera library this zip file will contain some demo code files as well as the required library. This library must also be stored in the **same working directory** as the code file in order for it to run properly.

One last change that may need to be made depends on the type of Raspberry Pi that is being used. If you are using a Raspberry Pi 4, a small edit needs to be made to the library. Open up the rd03d library file and look for the following line of code:
```
def __init__(self, uart_port='/dev/ttyAMA0', baudrate=256000, multi_mode=True):
```
This line initialises the UART communication and is done differently on the Pi 5 compared to older models. Replace it with the following line of code:
```
def __init__(self, uart_port='/dev/ttyS0', baudrate=256000, multi_mode=True):
```
## Configuration
The key tunable parameters across both files are summarised below:

| Parameter | File | Default | Description |
|---|---|---|---|
| `R` (Measurement Noise) | `Kalman Filter.py` | `80 × I` | Higher values make the filter trust measurements less and smooth more aggressively |
| `Q` (Process Noise) | `Kalman Filter.py` | `0.01 × I` | Higher values allow the filter to adapt faster to rapid motion |
| `P` (Initial Covariance) | `Kalman Filter.py` | `200 × I` | Higher values reflect greater uncertainty about the initial position |
| `threshold` (Hit Streak) | `Kalman Filter.py` | `5` frames | Consecutive valid frames required before a target is confirmed |
| Gating Distance | `Kalman Filter.py` | `800 mm` | Measurements further than this from the prediction are rejected post-confirmation |
| Radar Distance Range | `Sensor Fusion.py` | `350 – 5000 mm` | Targets outside this range are marked invalid |
| Radar Staleness Timeout | `Sensor Fusion.py` | `0.5 s` | Radar data older than this is treated as absent |
| Camera Confidence Threshold | `Sensor Fusion.py` | `0.5` | Detections below this confidence score are discarded |
| Angle Matching Gate | `Sensor Fusion.py` | `20°` | Maximum angular difference allowed when matching a radar target to a camera detection |
| Camera FOV | `Sensor Fusion.py` | `75°` | Must match the physical field of view of the lens in use |
| Loop / Thread Rate | `Sensor Fusion.py` | `10 Hz` | `time.sleep(0.1)` applied in both threads and the main loop |
