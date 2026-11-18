# toshot

Headless pipeline for **DJI Osmo 360** (`.OSV` / `.LRF`) and **Insta360** (`.insv` / `.lrv`) on **Linux and Windows**.

It reads dual-fisheye masters (and their camera proxies), stitches a gyro-stabilized equirectangular sphere, optionally tracks a subject with YOLO + ByteTrack, and exports a flat shot MP4. No DJI Studio / Insta360 Studio UI automation.

## Features

- **Formats:** `.osv`, `.lrf`, `.insv`, `.lrv` (case-insensitive)
- **Grouping:** `CAM_….OSV` + `CAM_….LRF`; Insta `_00_` / `_10_` pairs + `.lrv`
- **Stabilization:** `horizon` | `level` | `world` | `none`
- **Advanced stitch (default on):** DIS optical-flow parallax seam, selfie-stick / nadir mask, gyro rolling-shutter row warp
- **Equirect export:** maximum lens resolution (e.g. Osmo 7680×3840), 10-bit when the source is 10-bit, Google Spherical V1 metadata
- **AI shot:** proxy equirect → YOLO11 + ByteTrack → perspective reframe from the master
- **Batch:** overnight folder processing with per-capture logs

## Requirements

- Python **3.11+**
- **FFmpeg** and **ffprobe** on `PATH`, with:
  - HEVC decode
  - filters `remap`, `v360`, `sendcmd` (recommended; OpenCV fallback exists)
- Optional: NVIDIA GPU for YOLO (CUDA) and `hevc_nvenc` / `h264_nvenc`

### FFmpeg install hints

- **Linux:** distro package or [BtbN](https://github.com/BtbN/FFmpeg-Builds) static build
- **Windows:** [Gyan](https://www.gyan.dev/ffmpeg/builds/) or BtbN build; add `bin` to `PATH`

## Install

```bash
cd 360toShot
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e .
```

## GUI

```bash
# Windows
.\.venv\Scripts\Activate.ps1
toshot-gui
# or
toshot gui
```

Pick a file or folder, tick one or more tasks (**Inspect**, **Equirect**, **KI-Shot**), set options, click **Start**. A **folder** runs as batch for every capture with the selected exports (e.g. Shot+Batch or Equirect+Batch, or both). Progress appears in the log panel.

## CLI

```bash
# Metadata, lenses, IMU count
toshot inspect clip.OSV
toshot inspect VID_….insv

# Max-resolution equirect (stabilized + advanced seam)
toshot equirect clip.OSV -o sphere.mp4 --stab world
toshot equirect clip.OSV -o sphere_4k.mp4 --width 3840
toshot equirect clip.OSV -o sphere_fast.mp4 --fast   # soft blend only

# AI-tracked flat shot
toshot shot clip.OSV -o shot.mp4 --stab world --fov 90
toshot shot clip.insv -o shot.mp4 --class-name person

# Folder batch
toshot batch ./inbox --out ./out
toshot batch ./inbox --out ./out --mode equirect
toshot batch ./inbox --out ./out --equirect   # shot + equirect
```

### Advanced stitch

Enabled by default for `equirect` / `shot` / `batch`:

| Option | Effect |
|--------|--------|
| optical flow | DIS flow aligns the front lens to the rear in the overlap, then a DP seam carve |
| selfie mask | In the nadir band, prefer the less stick-like lens |
| rolling shutter | Per-row equirect warp from gyro yaw/pitch rate |

Disable individually with `--no-optical-flow`, `--no-selfie-mask`, `--no-rolling-shutter`, or all at once with `--fast` (uses the quicker FFmpeg remap path).

### Stabilization modes

| Mode | Behavior |
|------|----------|
| `horizon` | Level pitch/roll; yaw follows the camera |
| `level` | Level pitch/roll; world yaw locked to the first frame |
| `world` | Full orientation lock (default) |
| `none` | No gyro correction |

## Proxy vs master

If a matching `.LRF` / `.lrv` sits next to the master, **tracking** uses the proxy and **equirect / shot pixels** come from the master. A lone proxy is processed as its own capture at proxy resolution.

## Project layout

```
src/toshot/
  cli.py
  dji/          # OSV / LRF
  insta/        # INSV / LRV
  source/       # Capture model + grouping
  stitch/       # Remap LUTs + FFmpeg stitch
  track/        # YOLO + ByteTrack
  export/       # equirect, shot, batch
```

## Limits (v0.1)

- Advanced stitch is an OpenCV per-frame path (slower than `--fast`)
- Insta gyro parsing is best-effort across trailer variants
- Unknown layouts fail with a clear error instead of inventing pixels

## License

MIT
