from __future__ import annotations

import json
import logging
import struct
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from toshot.math3d import slerp_attitude, quat_to_euler_ypr
from toshot.preview.decode import DualLensDecoder, choose_preview_media
from toshot.preview.shaders import EQUIRECT_LOOK_SHADER, VIEWPORT_SHADER
from toshot.source.models import Capture, StabMode
from toshot.stitch.advanced import blend_advanced
from toshot.stitch.options import StitchOptions
from toshot.stitch.pipeline import _scale_lens
from toshot.stitch.remap import blend_weights, build_lens_maps, feather_px_for

log = logging.getLogger(__name__)

# #region agent log
_DBG = Path(r"d:\360toShot\.cursor\debug-06c6a9.log")


def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None, run_id: str = "pre") -> None:
    try:
        payload = {
            "sessionId": "06c6a9",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with _DBG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion


def _bgr_to_rgba(bgr: np.ndarray) -> np.ndarray:
    a = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, a], axis=2)


def _upload_rgba(device: wgpu.GPUDevice, texture: wgpu.GPUTexture, rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    device.queue.write_texture(
        {"texture": texture},
        rgba.tobytes(),
        {"bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )


def _upload_r32(device: wgpu.GPUDevice, texture: wgpu.GPUTexture, arr: np.ndarray) -> None:
    h, w = arr.shape[:2]
    data = np.ascontiguousarray(arr, dtype=np.float32)
    device.queue.write_texture(
        {"texture": texture},
        data.tobytes(),
        {"bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )


def run_preview(
    capture: Capture,
    *,
    stab: StabMode = StabMode.WORLD,
    width: int = 1280,
    height: int = 720,
    lens_edge: int | None = None,
    equirect_w: int = 2048,
    options: StitchOptions | None = None,
) -> None:
    """
    Interactive Vulkan (wgpu) stitch preview — like OSV Studio:
    live dual-lens stitch, mouse look, scroll FOV. Uses LRF/LRV when present.
    With advanced options: CPU optical-flow seam, then equirect look on GPU.
    """
    opts = options or StitchOptions()
    use_adv = opts.advanced
    path, kind = choose_preview_media(capture)
    log.info(
        "Vulkan-Vorschau: %s (%s) stab=%s advanced=%s (flow=%s selfie=%s rs=%s)",
        path.name,
        kind,
        stab.value,
        use_adv,
        opts.optical_flow,
        opts.selfie_mask,
        opts.rolling_shutter,
    )
    # #region agent log
    _dbg(
        "A",
        "viewer.py:run_preview:entry",
        "preview start",
        {
            "path": path.name,
            "kind": kind,
            "stab": stab.value,
            "advanced": use_adv,
            "optical_flow": opts.optical_flow,
        },
    )
    # #endregion

    rear = capture.calibration.rear
    front = capture.calibration.front or rear
    if rear is None:
        raise RuntimeError("Keine Linsenkalibrierung für Vorschau")

    # Preview lens resolution (proxy-friendly)
    if lens_edge is None:
        from toshot.preview.decode import _video_size

        pw, ph, sbs = _video_size(path)
        if kind == "proxy" and sbs and ph > 0:
            lens_edge = ph - (ph % 2)
        elif kind == "proxy" and pw > 0:
            lens_edge = min(pw, ph) - (min(pw, ph) % 2)
        else:
            lens_edge = 1536
    lens_edge = max(256, int(lens_edge) - (int(lens_edge) % 2))
    equirect_w = int(equirect_w) - (int(equirect_w) % 2)
    equirect_h = equirect_w // 2
    equirect_h -= equirect_h % 2

    rear_s = _scale_lens(rear, lens_edge, lens_edge)
    front_s = _scale_lens(front, lens_edge, lens_edge)
    log.info("Baue GPU-Remap-Maps %dx%d (lens %d) …", equirect_w, equirect_h, lens_edge)
    x0, y0, w0 = build_lens_maps(rear_s, equirect_w, equirect_h)
    x1, y1, w1 = build_lens_maps(front_s, equirect_w, equirect_h, ref_lens=rear_s)
    a, b = blend_weights(w0, w1, feather_px=feather_px_for(equirect_w))
    a = np.nan_to_num(a).astype(np.float32)
    b = np.nan_to_num(b).astype(np.float32)
    # #region agent log
    def _v_delta(ym, wm):
        col = int(np.argmax(wm.mean(axis=0)))
        ys = np.where(wm[:, col] > 0.5)[0]
        if len(ys) < 5:
            return None
        return float(ym[ys[-1], col] - ym[ys[0], col])

    mid = a[a.shape[0] // 2, :]
    soft = (mid > 0.05) & (mid < 0.95)
    soft_idx = np.where(soft)[0]
    grad = float(np.max(np.abs(np.diff(mid)))) if mid.size > 1 else 0.0
    _dbg(
        "R",
        "viewer.py:maps",
        "seam blend stats",
        {
            "rear_yaw": rear.yaw_deg,
            "front_yaw": front.yaw_deg,
            "overlap_both": float(((w0 > 0.3) & (w1 > 0.3)).mean()),
            "rear_pref": float((a > 0.7).mean()),
            "front_pref": float((a < 0.3).mean()),
            "rear_v_delta": _v_delta(y0, w0),
            "front_v_delta": _v_delta(y1, w1),
            "soft_cols": int(soft.sum()),
            "soft_span": int(soft_idx[-1] - soft_idx[0] + 1) if soft_idx.size else 0,
            "max_grad": grad,
            "a_mid_min": float(mid.min()) if mid.size else None,
            "a_mid_max": float(mid.max()) if mid.size else None,
            "ghost_band_50": int(((mid > 0.45) & (mid < 0.55)).sum()),
            "pose": "osvplat_rel_quat_theta_fix_yaw-1",
            "fov_cap_deg": 94.0,
            "blend_ring_deg": [87.0, 93.0],
        },
        run_id="post-fix",
    )
    # #endregion

    decoder = DualLensDecoder(capture, lens_edge, lens_edge)
    decoder.start()

    # Wait briefly for first frame
    frame = None
    for _ in range(100):
        frame = decoder.latest()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        # #region agent log
        _dbg("B", "viewer.py:run_preview:no_frame", "decoder produced no frame", {"path": path.name})
        # #endregion
        decoder.stop()
        raise RuntimeError(f"Kein Preview-Frame von {path.name}")

    # #region agent log
    _dbg(
        "B",
        "viewer.py:run_preview:first_frame",
        "got first frame",
        {"index": frame.index, "rear": list(frame.rear.shape), "front": list(frame.front.shape)},
    )
    # #endregion

    # present_method must be "screen" or "bitmap" — "auto" crashes get_context(wgpu)
    canvas = RenderCanvas(
        title=f"toshot preview · {capture.id} · {kind} · Vulkan",
        size=(width, height),
        update_mode="continuous",
        present_method="screen",
    )
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_limits={})
    log.info("GPU: %s", adapter.summary)
    # #region agent log
    _dbg("A", "viewer.py:run_preview:before_context", "creating wgpu context", {"present_method": "screen", "adapter": getattr(adapter, "summary", "")})
    # #endregion
    try:
        context = canvas.get_context("wgpu")
        fmt = context.get_preferred_format(adapter)
        context.configure(device=device, format=fmt, alpha_mode="opaque")
        # #region agent log
        _dbg("A", "viewer.py:run_preview:context_ok", "wgpu context configured", {"fmt": str(fmt)})
        # #endregion
    except Exception as exc:
        # #region agent log
        _dbg(
            "A",
            "viewer.py:run_preview:context_fail",
            "get_context/configure failed",
            {"error": str(exc), "tb": traceback.format_exc()[-1500:]},
            run_id="post-fix",
        )
        # #endregion
        decoder.stop()
        raise

    usage_tex = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    sampler = device.create_sampler(
        mag_filter="linear",
        min_filter="linear",
        address_mode_u="clamp-to-edge",
        address_mode_v="clamp-to-edge",
    )
    ubuf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    rear_tex = front_tex = None
    rear_x_tex = rear_y_tex = front_x_tex = front_y_tex = weight_tex = None
    eq_tex = None

    # Keep map names out of later locals — closures bind by name; shadowing
    # y0 with euler yaw made OpenCV remap see a float (CV assert).
    map_x0 = np.ascontiguousarray(x0, dtype=np.float32)
    map_y0 = np.ascontiguousarray(y0, dtype=np.float32)
    map_x1 = np.ascontiguousarray(x1, dtype=np.float32)
    map_y1 = np.ascontiguousarray(y1, dtype=np.float32)

    def _cpu_advanced_equirect(rear_bgr: np.ndarray, front_bgr: np.ndarray, t_s: float) -> np.ndarray:
        # #region agent log
        if not getattr(_cpu_advanced_equirect, "_logged", False):
            _cpu_advanced_equirect._logged = True  # type: ignore[attr-defined]
            _dbg(
                "H1",
                "viewer.py:_cpu_advanced_equirect",
                "remap map dtypes before cv2.remap",
                {
                    "x0_dtype": str(map_x0.dtype),
                    "y0_dtype": str(map_y0.dtype),
                    "x0_shape": list(map_x0.shape),
                    "y0_shape": list(map_y0.shape),
                    "y0_is_ndarray": isinstance(map_y0, np.ndarray),
                    "rear_shape": list(rear_bgr.shape),
                },
                run_id="post-fix",
            )
        # #endregion
        mapped0 = cv2.remap(
            rear_bgr, map_x0, map_y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mapped1 = cv2.remap(
            front_bgr, map_x1, map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return blend_advanced(
            mapped0,
            mapped1,
            a,
            b,
            options=opts,
            attitude=capture.attitude or [],
            t_s=t_s,
            fps=float(decoder.fps or capture.fps or 30.0),
        )

    if use_adv:
        eq_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="rgba8unorm", usage=usage_tex
        )
        eq0 = _cpu_advanced_equirect(frame.rear, frame.front, frame.t_s)
        _upload_rgba(device, eq_tex, _bgr_to_rgba(eq0))
        shader = device.create_shader_module(code=EQUIRECT_LOOK_SHADER)
    else:
        rear_tex = device.create_texture(
            size=(lens_edge, lens_edge, 1), format="rgba8unorm", usage=usage_tex
        )
        front_tex = device.create_texture(
            size=(lens_edge, lens_edge, 1), format="rgba8unorm", usage=usage_tex
        )
        rear_x_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="r32float", usage=usage_tex
        )
        rear_y_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="r32float", usage=usage_tex
        )
        front_x_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="r32float", usage=usage_tex
        )
        front_y_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="r32float", usage=usage_tex
        )
        weight_tex = device.create_texture(
            size=(equirect_w, equirect_h, 1), format="r32float", usage=usage_tex
        )
        _upload_r32(device, rear_x_tex, x0)
        _upload_r32(device, rear_y_tex, y0)
        _upload_r32(device, front_x_tex, x1)
        _upload_r32(device, front_y_tex, y1)
        _upload_r32(device, weight_tex, a)
        _upload_rgba(device, rear_tex, _bgr_to_rgba(frame.rear))
        _upload_rgba(device, front_tex, _bgr_to_rgba(frame.front))
        shader = device.create_shader_module(code=VIEWPORT_SHADER)

    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": fmt}],
        },
        primitive={"topology": "triangle-list"},
    )
    # #region agent log
    _dbg(
        "F",
        "viewer.py:bind_group:before",
        "creating bind group",
        {"advanced": use_adv},
    )
    # #endregion
    try:
        if use_adv:
            entries = [
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": 32}},
                {"binding": 1, "resource": sampler},
                {"binding": 2, "resource": eq_tex.create_view()},
            ]
        else:
            entries = [
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": 32}},
                {"binding": 1, "resource": sampler},
                {"binding": 2, "resource": rear_tex.create_view()},
                {"binding": 3, "resource": front_tex.create_view()},
                {"binding": 4, "resource": rear_x_tex.create_view()},
                {"binding": 5, "resource": rear_y_tex.create_view()},
                {"binding": 6, "resource": front_x_tex.create_view()},
                {"binding": 7, "resource": front_y_tex.create_view()},
                {"binding": 8, "resource": weight_tex.create_view()},
            ]
        bind = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=entries,
        )
        # #region agent log
        _dbg("F", "viewer.py:bind_group:ok", "bind group created", {"advanced": use_adv}, run_id="post-fix")
        # #endregion
    except Exception as exc:
        # #region agent log
        _dbg(
            "F",
            "viewer.py:bind_group:fail",
            "bind group failed",
            {"error": str(exc), "tb": traceback.format_exc()[-1200:]},
            run_id="post-fix",
        )
        # #endregion
        decoder.stop()
        raise

    state = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "fov": 90.0,
        "drag": False,
        "last": (0.0, 0.0),
        "gyro": stab != StabMode.NONE,
        "last_idx": -1,
        "stab": stab.value,
        "has_attitude": bool(capture.attitude),
        "look_events": 0,
        "gyro_samples": 0,
    }
    # #region agent log
    _dbg(
        "J",
        "viewer.py:state_init",
        "look/stab init",
        {
            "stab": stab.value,
            "gyro_on": state["gyro"],
            "has_attitude": state["has_attitude"],
            "n_attitude": len(capture.attitude or []),
        },
        run_id="orient",
    )
    if capture.attitude:
        q0 = slerp_attitude(capture.attitude, capture.attitude[0].t_s)
        att_yaw0, att_pitch0, att_roll0 = quat_to_euler_ypr(q0)
        _dbg(
            "J",
            "viewer.py:attitude0",
            "first attitude euler",
            {"yaw": att_yaw0, "pitch": att_pitch0, "roll": att_roll0, "q": list(q0)},
            run_id="orient",
        )
    # #endregion

    def on_down(ev):
        state["drag"] = True
        state["last"] = (ev["x"], ev["y"])

    def on_up(_ev):
        state["drag"] = False

    def on_move(ev):
        if not state["drag"]:
            return
        x, y = ev["x"], ev["y"]
        lx, ly = state["last"]
        dx = x - lx
        dy = y - ly
        # FPS look: drag right → yaw right; drag up → look up. No ±89 clamp so
        # the full sphere is reachable (pitch wraps through zenith/nadir).
        state["yaw"] -= dx * 0.2
        state["pitch"] -= dy * 0.2
        if state["pitch"] > 180.0:
            state["pitch"] -= 360.0
        elif state["pitch"] < -180.0:
            state["pitch"] += 360.0
        state["last"] = (x, y)
        # #region agent log
        state["look_events"] = int(state.get("look_events", 0)) + 1
        if state["look_events"] <= 8 or state["look_events"] % 30 == 0:
            _dbg(
                "M",
                "viewer.py:on_move",
                "mouse look",
                {
                    "dx": dx,
                    "dy": dy,
                    "yaw": state["yaw"],
                    "pitch": state["pitch"],
                    "pitch_clamped": False,
                    "uv_flip": False,
                    "n": state["look_events"],
                },
                run_id="post-fix",
            )
        # #endregion

    def on_wheel(ev):
        dy = ev.get("dy") or ev.get("delta_y") or 0
        state["fov"] = float(np.clip(state["fov"] + (float(dy) * 0.05), 30.0, 140.0))

    def on_key(ev):
        key = (ev.get("key") or "").lower()
        if key in ("escape", "q"):
            decoder.stop()
            canvas.close()
            try:
                loop.stop()
            except Exception:
                pass
        elif key in (" ", "space"):
            decoder.toggle_pause()
        elif key == "g":
            state["gyro"] = not state["gyro"]
            log.info("Gyro-Lock %s", "an" if state["gyro"] else "aus")
        elif key == "r":
            state["yaw"] = state["pitch"] = state["roll"] = 0.0
            state["fov"] = 90.0

    canvas.add_event_handler(on_down, "pointer_down")
    canvas.add_event_handler(on_up, "pointer_up")
    canvas.add_event_handler(on_move, "pointer_move")
    canvas.add_event_handler(on_wheel, "wheel")
    canvas.add_event_handler(on_key, "key_down")

    def on_close(_ev=None):
        decoder.stop()
        try:
            loop.stop()
        except Exception:
            pass

    canvas.add_event_handler(on_close, "close")

    def draw():
        try:
            fr = decoder.latest()
            if fr is not None and fr.index != state["last_idx"]:
                if use_adv:
                    eq = _cpu_advanced_equirect(fr.rear, fr.front, fr.t_s)
                    _upload_rgba(device, eq_tex, _bgr_to_rgba(eq))
                    # #region agent log
                    if state.get("_adv_n", 0) < 3:
                        state["_adv_n"] = state.get("_adv_n", 0) + 1
                        _dbg(
                            "ADV",
                            "viewer.py:draw:advanced",
                            "advanced equirect uploaded",
                            {"n": state["_adv_n"], "frame": fr.index, "shape": list(eq.shape)},
                            run_id="post-fix",
                        )
                    # #endregion
                else:
                    _upload_rgba(device, rear_tex, _bgr_to_rgba(fr.rear))
                    _upload_rgba(device, front_tex, _bgr_to_rgba(fr.front))
                state["last_idx"] = fr.index
                if state["gyro"] and capture.attitude and stab != StabMode.NONE:
                    q = slerp_attitude(capture.attitude, fr.t_s)
                    y, p, r = quat_to_euler_ypr(q)
                    if stab == StabMode.HORIZON:
                        state["roll"] = -r
                    elif stab == StabMode.LEVEL:
                        state["pitch"] = -p
                        state["roll"] = -r
                    elif stab == StabMode.WORLD:
                        state["roll"] = -r
                    # #region agent log
                    state["gyro_samples"] = int(state.get("gyro_samples", 0)) + 1
                    if state["gyro_samples"] <= 5 or state["gyro_samples"] % 60 == 0:
                        _dbg(
                            "J",
                            "viewer.py:gyro_apply",
                            "gyro euler applied",
                            {
                                "stab": stab.value,
                                "att_yaw": y,
                                "att_pitch": p,
                                "att_roll": r,
                                "state_yaw": state["yaw"],
                                "state_pitch": state["pitch"],
                                "state_roll": state["roll"],
                                "abs_roll_gt_90": abs(r) > 90.0,
                                "n": state["gyro_samples"],
                            },
                            run_id="orient",
                        )
                    # #endregion

            cw, ch = canvas.get_physical_size()
            aspect = float(cw) / float(max(ch, 1))
            yaw = state["yaw"]
            pitch = state["pitch"]
            roll = state["roll"] if state["gyro"] else 0.0
            # #region agent log
            if state.get("_orient_n", 0) < 5:
                state["_orient_n"] = state.get("_orient_n", 0) + 1
                _dbg(
                    "K",
                    "viewer.py:uniforms",
                    "gpu look uniforms",
                    {
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "fov": state["fov"],
                        "gyro": state["gyro"],
                        "uv_flip": False,
                        "pitch_unclamped": True,
                        "n": state["_orient_n"],
                    },
                    run_id="post-fix",
                )
            # #endregion
            device.queue.write_buffer(
                ubuf,
                0,
                struct.pack(
                    "ffffffff",
                    float(yaw),
                    float(pitch),
                    float(roll),
                    float(state["fov"]),
                    aspect,
                    float(equirect_w),
                    float(equirect_h),
                    0.0,
                ),
            )

            cur = context.get_current_texture()
            view = cur.create_view()
            enc = device.create_command_encoder()
            rp = enc.begin_render_pass(
                color_attachments=[
                    {
                        "view": view,
                        "resolve_target": None,
                        "clear_value": (0.02, 0.02, 0.03, 1.0),
                        "load_op": "clear",
                        "store_op": "store",
                    }
                ]
            )
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind)
            rp.draw(3)
            rp.end()
            device.queue.submit([enc.finish()])
            # #region agent log
            if state.get("_draw_n", 0) < 3:
                state["_draw_n"] = state.get("_draw_n", 0) + 1
                _dbg(
                    "C",
                    "viewer.py:draw",
                    "draw ok",
                    {"n": state["_draw_n"], "frame_idx": state["last_idx"], "size": [cw, ch]},
                    run_id="post-fix",
                )
            # #endregion
            canvas.request_draw()
        except Exception as exc:
            # #region agent log
            _dbg(
                "C",
                "viewer.py:draw:fail",
                "draw crashed",
                {"error": str(exc), "tb": traceback.format_exc()[-1500:]},
                run_id="post-fix",
            )
            # #endregion
            log.exception("Preview draw failed: %s", exc)
            decoder.stop()
            canvas.close()
            try:
                loop.stop()
            except Exception:
                pass

    canvas.request_draw(draw)
    log.info(
        "Steuerung: Ziehen=Look · Rad=FOV · Leertaste=Pause · G=Gyro · R=Reset · Esc=Ende"
    )
    # #region agent log
    _dbg("D", "viewer.py:run_preview:loop", "entering loop.run()", {})
    # #endregion
    try:
        loop.run()
        # #region agent log
        _dbg("D", "viewer.py:run_preview:loop_exit", "loop.run() returned", {}, run_id="post-fix")
        # #endregion
    finally:
        decoder.stop()
        log.info("Vorschau beendet")


def run_preview_path(path: str | Path, **kwargs) -> None:
    from toshot.source.group import load_capture
    from toshot.source.jobs import normalize_path

    cap = load_capture(normalize_path(path))
    run_preview(cap, **kwargs)
