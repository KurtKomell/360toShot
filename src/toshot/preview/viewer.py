from __future__ import annotations

import logging
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from toshot.math3d import Stabilizer
from toshot.preview.decode import DualLensDecoder, choose_preview_media
from toshot.preview.shaders import EQUIRECT_LOOK_SHADER, VIEWPORT_SHADER
from toshot.source.models import Capture, StabMode
from toshot.stitch.advanced import blend_advanced
from toshot.stitch.options import StitchOptions
from toshot.stitch.pipeline import _scale_lens
from toshot.stitch.remap import (
    blend_weights,
    build_lens_maps,
    feather_px_for,
    sphere_fov_for_pair,
)

log = logging.getLogger(__name__)

# 8 scalars + 3×vec4 stabilization rows (WGSL Uniforms)
_UNIFORM_BYTES = 80



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
    sf = sphere_fov_for_pair(rear_s, front_s)
    log.info(
        "Baue GPU-Remap-Maps %dx%d (lens %d)  Kugel=Rear-FOV %.2f°, Front %.2f° überlappt %.1f° …",
        equirect_w,
        equirect_h,
        lens_edge,
        sf.fov_rear,
        sf.fov_front,
        sf.overlap_deg,
    )
    x0, y0, w0 = build_lens_maps(
        rear_s,
        equirect_w,
        equirect_h,
        fov_deg=sf.fov_rear,
        blend_inner_deg=sf.rear_inner,
        blend_outer_deg=sf.rear_outer,
    )
    x1, y1, w1 = build_lens_maps(
        front_s,
        equirect_w,
        equirect_h,
        ref_lens=rear_s,
        fov_deg=sf.fov_front,
        blend_inner_deg=sf.front_inner,
        blend_outer_deg=sf.front_outer,
    )
    a, b = blend_weights(w0, w1, feather_px=feather_px_for(equirect_w))
    a = np.nan_to_num(a).astype(np.float32)
    b = np.nan_to_num(b).astype(np.float32)

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
        decoder.stop()
        raise RuntimeError(f"Kein Preview-Frame von {path.name}")


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
    try:
        context = canvas.get_context("wgpu")
        fmt = context.get_preferred_format(adapter)
        context.configure(device=device, format=fmt, alpha_mode="opaque")
    except Exception as exc:
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
        size=_UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    stabilizer = Stabilizer(capture.attitude or [], stab)
    eye3 = np.eye(3, dtype=np.float64)

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
    try:
        if use_adv:
            entries = [
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": _UNIFORM_BYTES}},
                {"binding": 1, "resource": sampler},
                {"binding": 2, "resource": eq_tex.create_view()},
            ]
        else:
            entries = [
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": _UNIFORM_BYTES}},
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
    except Exception as exc:
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
        "stab_m": stabilizer.matrix(frame.t_s),
    }

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
                else:
                    _upload_rgba(device, rear_tex, _bgr_to_rgba(fr.rear))
                    _upload_rgba(device, front_tex, _bgr_to_rgba(fr.front))
                state["last_idx"] = fr.index
                state["stab_m"] = stabilizer.matrix(fr.t_s)

            cw, ch = canvas.get_physical_size()
            aspect = float(cw) / float(max(ch, 1))
            s = state["stab_m"] if state["gyro"] else eye3
            device.queue.write_buffer(
                ubuf,
                0,
                struct.pack(
                    "8f12f",
                    float(state["yaw"]),
                    float(state["pitch"]),
                    float(state["roll"]),
                    float(state["fov"]),
                    aspect,
                    float(equirect_w),
                    float(equirect_h),
                    0.0,
                    *s[0], 0.0,
                    *s[1], 0.0,
                    *s[2], 0.0,
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
            canvas.request_draw()
        except Exception as exc:
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
    try:
        loop.run()
    finally:
        decoder.stop()
        log.info("Vorschau beendet")


def run_preview_path(path: str | Path, **kwargs) -> None:
    from toshot.source.group import load_capture
    from toshot.source.jobs import normalize_path

    cap = load_capture(normalize_path(path))
    run_preview(cap, **kwargs)
