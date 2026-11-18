from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from toshot import __version__
from toshot.export.batch import run_batch
from toshot.export.equirect import export_equirect
from toshot.export.shot import export_shot
from toshot.gui.log_bridge import StdRedirect, install_gui_logging
from toshot.source.group import resolve_path
from toshot.source.jobs import discover_jobs, job_for_path
from toshot.source.models import StabMode
from toshot.ffmpeg_bin import EncodeSettings, normalize_audio_bitrate, normalize_bitrate
from toshot.stitch.options import StitchOptions
from toshot.yolo_params import YOLO_CLASSES, YOLO_MODEL_LABELS, YOLO_MODELS

_CODEC_LABELS = {
    "Auto": "auto",
    "H.264": "h264",
    "HEVC": "hevc",
    "ProRes 422 HQ": "prores",
    "ProRes 4444": "prores4444",
}
_RATE_LABELS = {
    "Qualität (CRF)": "crf",
    "Bitrate": "bitrate",
}

SUPPORTED = "*.osv *.OSV *.lrf *.LRF *.insv *.INSV *.lrv *.LRV"

# Atmosphere: charcoal + copper (not purple / cream-serif defaults)
COLORS = {
    "bg": "#12151a",
    "panel": "#1a1f27",
    "panel2": "#222933",
    "text": "#e8ebe6",
    "muted": "#8b938a",
    "accent": "#c47a3a",
    "accent_hover": "#d4894a",
    "line": "#2e3642",
    "ok": "#6fbf8a",
    "err": "#d96b5c",
}


class ToshotApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"toshot  ·  {__version__}")
        self.geometry("1080x820")
        self.minsize(920, 640)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=COLORS["bg"])

        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._log_q: queue.Queue[str] = queue.Queue()
        self._log_handler = install_gui_logging(self._log_q)
        self._input = ctk.StringVar(value="")
        self._output = ctk.StringVar(value=str(Path.cwd() / "out"))
        self._do_inspect = ctk.BooleanVar(value=False)
        self._do_equirect = ctk.BooleanVar(value=True)
        self._do_shot = ctk.BooleanVar(value=False)
        self._stab = ctk.StringVar(value="world")
        self._fov = ctk.StringVar(value="90")
        self._width = ctk.StringVar(value="")
        self._class = ctk.StringVar(value="person")
        self._yolo_model = ctk.StringVar(value=YOLO_MODELS[0])
        # Advanced stitch on (DJI-like optical flow + seam carve); uncheck for speed
        self._fast = ctk.BooleanVar(value=False)
        self._optical = ctk.BooleanVar(value=True)
        self._selfie = ctk.BooleanVar(value=True)
        self._rs = ctk.BooleanVar(value=True)
        self._codec = ctk.StringVar(value="Auto")
        self._rate = ctk.StringVar(value="Qualität (CRF)")
        self._crf = ctk.StringVar(value="18")
        self._bitrate = ctk.StringVar(value="")
        self._preset = ctk.StringVar(value="medium")
        self._depth = ctk.StringVar(value="auto")
        self._audio = ctk.StringVar(value="192")
        self._status = ctk.StringVar(value="Bereit")

        self._build()
        self.after(50, self._drain_log)
        logging.getLogger("toshot.gui").info("GUI bereit — Log zeigt DEBUG+ inkl. FFmpeg")

    def _build(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=28, pady=24)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Brand header — one composition
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        ctk.CTkLabel(
            header,
            text="toshot",
            font=ctk.CTkFont(family="Segoe UI Semibold", size=36),
            text_color=COLORS["text"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="OSV · LRF · INSV · LRV  →  Equirect & KI-Shot",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=COLORS["muted"],
        ).pack(anchor="w", pady=(2, 0))

        left = ctk.CTkScrollableFrame(
            root,
            fg_color=COLORS["panel"],
            corner_radius=14,
            scrollbar_button_color=COLORS["line"],
            scrollbar_button_hover_color=COLORS["accent"],
        )
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        right = ctk.CTkFrame(root, fg_color=COLORS["panel"], corner_radius=14)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        self._build_controls(left)
        self._build_log(right)

        foot = ctk.CTkFrame(root, fg_color="transparent")
        foot.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ctk.CTkLabel(
            foot,
            textvariable=self._status,
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(side="left")
        self._run_btn = ctk.CTkButton(
            foot,
            text="Start",
            width=140,
            height=40,
            corner_radius=8,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#111",
            font=ctk.CTkFont(family="Segoe UI Semibold", size=15),
            command=self._on_run,
        )
        self._run_btn.pack(side="right")
        ctk.CTkButton(
            foot,
            text="Abbrechen",
            width=110,
            height=40,
            corner_radius=8,
            fg_color=COLORS["panel2"],
            hover_color=COLORS["line"],
            command=self._on_cancel,
        ).pack(side="right", padx=(0, 10))

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        pad = {"padx": 18, "pady": 6}
        ctk.CTkLabel(
            parent,
            text="Quelle",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(16, 0))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", **pad)
        ctk.CTkEntry(
            row,
            textvariable=self._input,
            height=36,
            fg_color=COLORS["panel2"],
            border_color=COLORS["line"],
            placeholder_text="Datei oder Ordner…",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Datei", width=70, command=self._pick_file).pack(
            side="left", padx=(8, 0)
        )
        ctk.CTkButton(row, text="Ordner", width=70, command=self._pick_folder).pack(
            side="left", padx=(6, 0)
        )

        ctk.CTkLabel(
            parent, text="Ausgabe", font=ctk.CTkFont(size=13), text_color=COLORS["muted"]
        ).pack(anchor="w", padx=18, pady=(10, 0))
        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", **pad)
        ctk.CTkEntry(
            row2,
            textvariable=self._output,
            height=36,
            fg_color=COLORS["panel2"],
            border_color=COLORS["line"],
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row2, text="…", width=40, command=self._pick_out).pack(
            side="left", padx=(8, 0)
        )

        ctk.CTkLabel(
            parent,
            text="Aufgaben (mehrere gleichzeitig)",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(10, 0))
        ctk.CTkLabel(
            parent,
            text="Ordner = Batch für alle Captures · Datei = einzelner Clip",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 4))
        tasks = ctk.CTkFrame(parent, fg_color="transparent")
        tasks.pack(fill="x", padx=18, pady=4)
        for text, var in (
            ("Inspect", self._do_inspect),
            ("Equirect", self._do_equirect),
            ("KI-Shot", self._do_shot),
        ):
            ctk.CTkCheckBox(
                tasks,
                text=text,
                variable=var,
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["line"],
                text_color=COLORS["text"],
            ).pack(side="left", padx=(0, 16))

        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(8, 4))
        grid.grid_columnconfigure((0, 1), weight=1)

        self._labeled(grid, 0, 0, "Stabilisierung", self._stab, ["world", "horizon", "level", "none"])
        self._entry(grid, 0, 1, "FOV (Shot)", self._fov)
        self._entry(grid, 1, 0, "Equirect-Breite (leer = max)", self._width)
        self._labeled(grid, 1, 1, "YOLO-Klasse", self._class, YOLO_CLASSES)
        self._labeled(
            grid,
            2,
            0,
            "YOLO-Modell",
            self._yolo_model,
            YOLO_MODELS,
        )
        # Show friendly labels in a second menu row note
        ctk.CTkLabel(
            grid,
            text="  ".join(YOLO_MODEL_LABELS[m] for m in YOLO_MODELS[:3]) + " …",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=10),
            anchor="w",
        ).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=4)

        ctk.CTkLabel(
            parent,
            text="Encoder",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(12, 0))
        enc = ctk.CTkFrame(parent, fg_color="transparent")
        enc.pack(fill="x", padx=18, pady=(4, 4))
        enc.grid_columnconfigure((0, 1), weight=1)
        self._labeled(enc, 0, 0, "Codec", self._codec, list(_CODEC_LABELS))
        self._labeled(enc, 0, 1, "Steuerung", self._rate, list(_RATE_LABELS))
        self._entry(enc, 1, 0, "CRF (0–51, kleiner = besser)", self._crf)
        self._entry(enc, 1, 1, "Bitrate (Mbit/s)", self._bitrate, placeholder="80")
        self._labeled(enc, 2, 0, "Preset", self._preset, ["veryfast", "fast", "medium", "slow"])
        self._labeled(enc, 2, 1, "Farbtiefe", self._depth, ["auto", "8", "10"])
        self._entry(enc, 3, 0, "Audio (kbit/s)", self._audio)
        ctk.CTkLabel(
            parent,
            text="Bitrate gilt bei Steuerung „Bitrate“. ProRes nutzt das Profil.",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 2))

        ctk.CTkLabel(
            parent,
            text="Stitch",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(12, 0))
        ctk.CTkLabel(
            parent,
            text="Equirect stitcht wie die Vorschau. Die Haken gelten für den KI-Shot.",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 2))
        checks = ctk.CTkFrame(parent, fg_color="transparent")
        checks.pack(fill="x", padx=18, pady=6)
        for text, var in (
            ("Optischer Fluss / Naht", self._optical),
            ("Selfie-Stick-Maske", self._selfie),
            ("Rolling Shutter", self._rs),
            ("Schnell (--fast)", self._fast),
        ):
            ctk.CTkCheckBox(
                checks,
                text=text,
                variable=var,
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["line"],
                text_color=COLORS["text"],
            ).pack(anchor="w", pady=3)

        ctk.CTkButton(
            parent,
            text="Vulkan-Vorschau (Live-Stitch)",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=self._open_preview,
        ).pack(fill="x", padx=18, pady=(16, 6))
        ctk.CTkButton(
            parent,
            text="Beide Linsen roh (ohne Stitch)",
            fg_color=COLORS["panel2"],
            hover_color=COLORS["line"],
            command=self._open_raw_lenses,
        ).pack(fill="x", padx=18, pady=(0, 6))
        ctk.CTkButton(
            parent,
            text="Captures im Ordner listen",
            fg_color=COLORS["panel2"],
            hover_color=COLORS["line"],
            command=self._list_captures,
        ).pack(fill="x", padx=18, pady=(0, 18))

    def _labeled(
        self,
        grid: ctk.CTkFrame,
        r: int,
        c: int,
        label: str,
        var: ctk.StringVar,
        values: list[str],
    ) -> None:
        cell = ctk.CTkFrame(grid, fg_color="transparent")
        cell.grid(row=r, column=c, sticky="ew", padx=(0, 8), pady=4)
        ctk.CTkLabel(cell, text=label, text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).pack(
            anchor="w"
        )
        ctk.CTkOptionMenu(
            cell,
            variable=var,
            values=values,
            fg_color=COLORS["panel2"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
        ).pack(fill="x", pady=(2, 0))

    def _entry(
        self,
        grid: ctk.CTkFrame,
        r: int,
        c: int,
        label: str,
        var: ctk.StringVar,
        placeholder: str = "",
    ) -> None:
        cell = ctk.CTkFrame(grid, fg_color="transparent")
        cell.grid(row=r, column=c, sticky="ew", padx=(0, 8), pady=4)
        ctk.CTkLabel(cell, text=label, text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).pack(
            anchor="w"
        )
        ctk.CTkEntry(
            cell,
            textvariable=var,
            height=32,
            fg_color=COLORS["panel2"],
            border_color=COLORS["line"],
            placeholder_text=placeholder,
        ).pack(fill="x", pady=(2, 0))

    def _encode_settings(self) -> EncodeSettings:
        codec = _CODEC_LABELS.get(self._codec.get())
        if codec is None:
            raise ValueError(f"Unbekannter Codec: {self._codec.get()}")
        mode = _RATE_LABELS.get(self._rate.get())
        if mode is None:
            raise ValueError(f"Unbekannte Steuerung: {self._rate.get()}")
        try:
            crf = int((self._crf.get() or "18").strip())
        except ValueError as exc:
            raise ValueError("CRF muss eine ganze Zahl zwischen 0 und 51 sein") from exc
        if not 0 <= crf <= 51:
            raise ValueError("CRF muss zwischen 0 und 51 liegen")
        bitrate = None
        if mode == "bitrate":
            raw = self._bitrate.get().strip()
            if not raw:
                raise ValueError("Bitte eine Video-Bitrate in Mbit/s angeben")
            bitrate = normalize_bitrate(raw)
        depth_txt = self._depth.get()
        if depth_txt not in {"auto", "8", "10"}:
            raise ValueError("Farbtiefe muss auto, 8 oder 10 sein")
        return EncodeSettings(
            codec=codec,
            crf=crf,
            bitrate=bitrate,
            preset=self._preset.get() or "medium",
            audio_bitrate=normalize_audio_bitrate(self._audio.get().strip() or "192"),
            bit_depth=None if depth_txt == "auto" else int(depth_txt),
        )

    def _build_log(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="Log",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(16, 6))
        self._log = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Cascadia Mono", size=12),
            fg_color=COLORS["panel2"],
            text_color=COLORS["text"],
            border_color=COLORS["line"],
            border_width=1,
            wrap="word",
        )
        self._log.pack(fill="both", expand=True, padx=18, pady=(0, 18))

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Capture wählen",
            filetypes=[
                ("360 Captures", SUPPORTED),
                ("Alle Dateien", "*.*"),
            ],
        )
        if path:
            self._input.set(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="Ordner wählen")
        if path:
            self._input.set(path)
            self._status.set("Ordner gewählt → Start verarbeitet alle Captures (Batch)")

    def _pick_out(self) -> None:
        path = filedialog.askdirectory(title="Ausgabeordner")
        if path:
            self._output.set(path)

    def _preview_source(self) -> Path | None:
        src = self._input.get().strip()
        if not src:
            messagebox.showinfo("toshot", "Bitte eine Capture-Datei wählen (nicht nur Ordner).")
            return None
        path = resolve_path(src)
        if path.is_dir():
            messagebox.showinfo(
                "toshot",
                "Vorschau braucht eine einzelne .OSV/.insv-Datei.",
            )
            return None
        return path

    def _spawn_preview(self, cmd: list[str], status: str) -> None:
        self._status.set(status)
        try:
            import os

            env = os.environ.copy()
            src_root = str(Path(__file__).resolve().parents[2])  # .../src
            env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.Popen(cmd, env=env)
        except Exception as exc:
            messagebox.showerror("toshot", f"Vorschau fehlgeschlagen:\n{exc}")

    def _open_preview(self) -> None:
        """Launch Vulkan live stitch in a separate process (GLFW needs own loop)."""
        path = self._preview_source()
        if path is None:
            return
        stab = self._stab.get() or "world"
        cmd = [
            sys.executable,
            "-m",
            "toshot.preview",
            str(path),
            "--stab",
            stab,
            "-v",
        ]
        if self._fast.get():
            cmd.append("--fast")
        else:
            if not self._optical.get():
                cmd.append("--no-optical-flow")
            if not self._selfie.get():
                cmd.append("--no-selfie-mask")
            if not self._rs.get():
                cmd.append("--no-rolling-shutter")
        mode = "fast" if self._fast.get() else "advanced"
        self._append_log(
            f"▶ Vulkan-Vorschau  {path.name}  stab={stab}  stitch={mode} "
            f"(flow={self._optical.get()} selfie={self._selfie.get()} rs={self._rs.get()})"
        )
        self._spawn_preview(cmd, "Vorschau gestartet…")

    def _open_raw_lenses(self) -> None:
        """Show rear|front fisheye without any stitching."""
        path = self._preview_source()
        if path is None:
            return
        cmd = [
            sys.executable,
            "-m",
            "toshot.preview",
            str(path),
            "--raw-lenses",
            "-v",
        ]
        self._append_log(f"▶ Roh-Linsen (ohne Stitch)  {path.name}")
        self._spawn_preview(cmd, "Roh-Linsen gestartet…")

    def _stitch_opts(self) -> StitchOptions:
        return StitchOptions.from_flags(
            optical_flow=self._optical.get(),
            selfie_mask=self._selfie.get(),
            rolling_shutter=self._rs.get(),
            fast=self._fast.get(),
        )

    def _append_log(self, line: str) -> None:
        # Exact lines as queued (already split); keep full text
        self._log.insert("end", line + "\n")
        self._log.see("end")

    def _drain_log(self) -> None:
        try:
            # Drain a burst so heavy FFmpeg output stays responsive
            for _ in range(200):
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        self.after(50, self._drain_log)

    def _list_captures(self) -> None:
        src = self._input.get().strip()
        if not src:
            messagebox.showinfo("toshot", "Bitte zuerst einen Ordner wählen.")
            return
        path = Path(src)
        folder = path if path.is_dir() else path.parent
        try:
            jobs = discover_jobs(folder)
        except Exception as exc:
            messagebox.showerror("toshot", str(exc))
            return
        self._append_log(f"— {len(jobs)} Capture(s) in {folder} (nur Dateinamen, kein Parse)")
        for j in jobs:
            self._append_log(
                f"  {j.id}  [{j.kind}]  master={j.master.name}  "
                f"proxy={j.proxy.name if j.proxy else '-'}"
            )

    def _on_cancel(self) -> None:
        self._stop.set()
        self._status.set("Abbruch angefordert…")

    def _batch_mode(self) -> str | None:
        """Map checkboxes to run_batch mode string."""
        eq = self._do_equirect.get()
        shot = self._do_shot.get()
        if eq and shot:
            return "both"
        if eq:
            return "equirect"
        if shot:
            return "shot"
        return None

    def _on_run(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("toshot", "Es läuft bereits ein Job.")
            return
        src = self._input.get().strip()
        if not src:
            messagebox.showinfo("toshot", "Bitte Datei oder Ordner wählen.")
            return
        if not (
            self._do_inspect.get() or self._do_equirect.get() or self._do_shot.get()
        ):
            messagebox.showinfo(
                "toshot",
                "Mindestens eine Aufgabe wählen: Inspect, Equirect und/oder KI-Shot.",
            )
            return
        try:
            encode = self._encode_settings()
        except ValueError as exc:
            messagebox.showinfo("toshot", str(exc))
            return
        self._stop.clear()
        self._run_btn.configure(state="disabled")
        self._status.set("Läuft…")
        tasks = []
        if self._do_inspect.get():
            tasks.append("inspect")
        if self._do_equirect.get():
            tasks.append("equirect")
        if self._do_shot.get():
            tasks.append("shot")
        self._append_log(
            f"▶ Start  src={src}  tasks={'+'.join(tasks)}  "
            f"fast={self._fast.get()}  {encode.summary()}  out={self._output.get()}"
        )
        self._worker = threading.Thread(target=self._run_job, args=(src, encode), daemon=True)
        self._worker.start()

    def _log_inspect(self, cap) -> None:
        lines = [
            f"id={cap.id}",
            f"family={cap.family.value}",
            f"layout={cap.layout.value}",
            f"master={cap.master_paths}",
            f"proxy={cap.proxy_path}",
            f"fps={cap.fps:.3f}",
            f"duration={cap.duration_s:.3f}s",
            f"bit_depth={cap.bit_depth}",
            f"equirect_max={cap.equirect_size()}",
            f"attitude={len(cap.attitude)}",
            f"model={cap.calibration.model}",
            *cap.notes,
        ]
        for line in lines:
            self._log_q.put(line)

    def _export_one(
        self,
        cap,
        out_dir: Path,
        stab: StabMode,
        opts: StitchOptions,
        width,
        fov,
        encode: EncodeSettings,
    ) -> list[Path]:
        written: list[Path] = []
        if self._do_equirect.get():
            dest = out_dir / f"{cap.id}_equirect.mp4"
            export_equirect(cap, dest, stab=stab, width=width, options=opts, encode=encode)
            self._log_q.put(f"wrote {dest}")
            written.append(dest)
        if self._do_shot.get():
            dest = out_dir / f"{cap.id}_shot.mp4"
            export_shot(
                cap,
                dest,
                stab=stab,
                fov=fov,
                class_name=self._class.get() or "person",
                model_name=self._yolo_model.get() or "yolo11n.pt",
                options=opts,
                encode=encode,
            )
            self._log_q.put(f"wrote {dest}")
            written.append(dest)
        return written

    def _run_job(self, src: str, encode: EncodeSettings) -> None:
        log = logging.getLogger("toshot.gui")
        # Mirror stdout/stderr (ultralytics prints, tracebacks) into the same panel
        with StdRedirect(self._log_q):
            try:
                stab = StabMode(self._stab.get())
                opts = self._stitch_opts()
                out_dir = Path(self._output.get().strip() or "out")
                out_dir.mkdir(parents=True, exist_ok=True)
                width = None
                wtxt = self._width.get().strip()
                if wtxt:
                    width = int(wtxt)
                fov = float(self._fov.get() or 90)
                path = resolve_path(src)
                batch_mode = self._batch_mode()
                do_inspect = self._do_inspect.get()
                log.info("Pfad aufgelöst: %s (isdir=%s)", path, path.is_dir())
                log.info(
                    "Stitch: advanced=%s optical=%s selfie=%s rs=%s",
                    opts.advanced,
                    opts.optical_flow,
                    opts.selfie_mask,
                    opts.rolling_shutter,
                )

                # Folder → batch all captures with selected exports
                if path.is_dir():
                    log.info("Suche Captures im Ordner (schnell, ohne Parse)…")
                    jobs = discover_jobs(path)
                    if not jobs:
                        self._status_safe("Keine .OSV/.LRF/.insv/.lrv im Ordner")
                        return
                    if do_inspect:
                        for i, job in enumerate(jobs, 1):
                            if self._stop.is_set():
                                break
                            log.info("— inspect [%d/%d] %s", i, len(jobs), job.id)
                            self._log_inspect(job.load())
                    if batch_mode is None:
                        self._status_safe(
                            "Inspect fertig" if do_inspect else "Nichts zu exportieren"
                        )
                        return
                    log.info("— batch mode=%s in %s", batch_mode, path)
                    written = run_batch(
                        path,
                        out_dir,
                        mode=batch_mode,
                        stab=stab,
                        fov=fov,
                        class_name=self._class.get() or "person",
                        model_name=self._yolo_model.get() or "yolo11n.pt",
                        equirect_width=width,
                        options=opts,
                        encode=encode,
                    )
                    for p in written:
                        log.info("wrote %s", p)
                    self._status_safe(f"Batch fertig ({len(written)} Dateien)")
                    return

                # Single file → only this capture (+ sibling proxy), not whole folder
                log.info("Lese einzelnes Capture (ffprobe / Metadaten)…")
                job = job_for_path(path)
                log.info("Job: master=%s proxy=%s", job.master, job.proxy)
                cap = job.load()
                log.info(
                    "Capture ok: %s  %s/%s  %.1fs  streams=%d",
                    cap.id,
                    cap.family.value,
                    cap.layout.value,
                    cap.duration_s,
                    len(cap.video_streams),
                )
                if do_inspect:
                    self._log_inspect(cap)
                written = self._export_one(cap, out_dir, stab, opts, width, fov, encode)
                if written:
                    self._status_safe(f"Fertig ({len(written)} Datei(en))")
                elif do_inspect:
                    self._status_safe("Inspect fertig")
                else:
                    self._status_safe("Nichts geschrieben")
            except Exception:
                log.exception("Job fehlgeschlagen")
                self._log_q.put(traceback.format_exc())
                self._status_safe("Fehler — siehe Log")
            finally:
                self.after(0, lambda: self._run_btn.configure(state="normal"))

    def _status_safe(self, text: str) -> None:
        self.after(0, lambda: self._status.set(text))


def run_gui() -> None:
    app = ToshotApp()
    app.mainloop()
