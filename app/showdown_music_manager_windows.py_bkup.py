"""
DiRT Showdown Music Manager (Beginner-Friendly, All-in-One)
===========================================================

This tool helps you replace DiRT Showdown's built-in music (WIM files) with your own custom tracks.

It does FOUR big jobs:

1) Read the game's DIC packs:
   - A .DIC file is like an "audio dictionary" that lists lots of audio streams.
   - Each stream often points at a container file, like .WIM or .WIP.

2) Use musicPlayer.xml (if present):
   - This file tells us which stream IDs are FE / RACE / REPLAY / etc.
   - It's useful for filtering so you only replace certain types of music.

3) Convert your custom music to the game's format:
   - DiRT uses 8-bit u-Law (mulaw) audio in these WIM containers (very old-school).
   - We use ffmpeg to convert.
   - Optional: loudness matching (LUFS) so volume feels similar to original.

4) Inject:
   - For WIM streams only (music). We don't inject WIP by default.
   - We always back up the original DIC/WIM files before overwriting them.

Requirements
------------
- Windows 10/11
- Python 3.10+ (works in 3.13+)
- pip install rich
- ffmpeg.exe bundled under ./bin/ffmpeg/ (or available on PATH)
- vgmstream CLI bundled under ./bin/vgmstream-win/ (vgmstream-cli.exe) (or available on PATH)

Notes
-----
- This script is cautious:
  - It makes backups under: showdown_backups/
  - It writes logs under: showdown_outputs/logs/
  - It writes plans under: showdown_outputs/plans/

"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- PyInstaller / Windows path rooting -------------------------------------
# When running as a PyInstaller-built EXE, sys.executable points to the EXE.
# When running from source (typically in app\), we want the project root.
def get_app_root() -> Path:
    # Running as PyInstaller EXE
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent

        # Newer PyInstaller sometimes nests data in _internal
        internal_dir = exe_dir / "_internal"
        if internal_dir.exists():
            return internal_dir

        return exe_dir

    # Running from source: app/script.py
    return Path(__file__).resolve().parent.parent

APP_ROOT = get_app_root()

DATA_ROOT = (APP_ROOT.parent if (getattr(sys, 'frozen', False) and APP_ROOT.name.lower() == '_internal') else APP_ROOT)
# ---------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Any, Set

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()


APP_VERSION = "0.0.2"

# ==============================
# UX helpers (prevent cls wiping important output)
# ==============================

def clear_screen() -> None:
    """Clear terminal (Windows-only).

    Users can disable screen clears by setting environment variable:
      SHOWDOWN_NO_CLS=1
    """
    try:
        if os.environ.get("SHOWDOWN_NO_CLS", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        # Rich clear is smoother than os.system("cls") and works in CMD.
        console.clear()
    except Exception:
        try:
            os.system("cls")
        except Exception:
            pass


def pause(message: str = "Press Enter to continue...") -> None:
    """Pause so the user can read the screen."""
    try:
        input(f"\n{message}")
    except Exception:
        try:
            Prompt.ask(message, default="")
        except Exception:
            pass


def render_header(title: str, game_dir: str | None = None, xml_status: str | None = None) -> None:
    """Render a consistent header panel."""
    t = Text()
    t.append(f"{title}\n", style="bold cyan")
    t.append(f"Version: {APP_VERSION}\n", style="dim")
    if game_dir is not None:
        t.append("\nGame Folder: ", style="bold")
        t.append(f"{game_dir or 'Not set'}\n")
    if xml_status is not None:
        t.append("musicPlayer.xml: ", style="bold")
        t.append(xml_status)
    console.print(Panel(t, expand=False, border_style="cyan"))


def render_section(title: str) -> None:
    console.print(Rule(f"[bold]{title}[/bold]"))

# ==============================
# File/Folder constants
# ==============================

# Tool executables (Windows-only release)
# We prefer bundled binaries under ./bin/ but will fall back to PATH.
DEFAULT_VGMSTREAM_EXE_NAMES = ["vgmstream-cli.exe", "test.exe"]
DEFAULT_FFMPEG_EXE_NAMES = ["ffmpeg.exe", "ffmpeg"]
DEFAULT_FFPROBE_EXE_NAMES = ["ffprobe.exe", "ffprobe"]

CONFIG_FILE = DATA_ROOT / "showdown_manager_config.json"

PROFILES_JSON = DATA_ROOT / "showdown_profiles.json"
PROFILES_PY_TXT = Path("showdown_profiles.py.txt")

BACKUP_ROOT = DATA_ROOT / "backups"
OUTPUT_ROOT = DATA_ROOT / "output"
LOGS_DIR = OUTPUT_ROOT / "logs"
PLANS_DIR = OUTPUT_ROOT / "plans"

# Default loudness targets (these are safe defaults).
DEFAULT_TP = -1.5   # True Peak limit (dBTP)
DEFAULT_LRA = 11.0  # Loudness range target
DEFAULT_SCAN_FAST_STREAMS = 12  # when using "fast" scan mode

# What to do if a custom audio file fails conversion/analyze:
# - strict: abort immediately
# - skip: skip that track
# - fallback: try loudnorm, then fallback to basic conversion, else skip
DEFAULT_BAD_AUDIO_POLICY = "fallback"

# Hard cap to avoid going crazy with threads.
DEFAULT_MAX_WORKERS_CAP = 4

TEMP_MULAW_NAME = "_temp_out.mulaw"
MULAW_SILENCE_BYTE = b"\xFF"

# Accepted custom music file extensions
AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"
}


# ==============================
# Windows-only helpers
# ==============================

def is_windows() -> bool:
    return True

def get_script_dir() -> Path:
    return Path(__file__).resolve().parent

def bin_dir() -> Path:
    return APP_ROOT / "bin"

def resolve_exe(candidates: List[str]) -> Optional[Path]:
    """Locate an executable (Windows-only).

    `candidates` may contain:
      - relative paths (e.g. r"bin\ffmpeg\ffmpeg.exe")
      - bare exe names (e.g. "ffmpeg.exe") to fall back to PATH.

    Search order:
      1) <script_dir>/<relative candidate>
      2) PATH lookup for bare names
    """
    base = APP_ROOT
    for c in candidates:
        p = (base / c)
        if p.exists():
            return p

    for c in candidates:
        name = Path(c).name
        hit = shutil.which(name)
        if hit:
            return Path(hit)

    return None


def resolve_vgmstream_exe() -> Path:
    candidates = [
        r"bin\vgmstream-win\vgmstream-cli.exe",
        r"bin\vgmstream-win\test.exe",
        r"vgmstream-cli.exe",
        r"test.exe",
        r"vgmstream-cli",
        r"test",
    ]
    p = resolve_exe(candidates)
    if not p:
        raise FileNotFoundError(
            "vgmstream CLI not found. Expected ./bin/vgmstream-win/vgmstream-cli.exe (or on PATH)."
        )
    return p


def resolve_ffmpeg_exe() -> str:
    candidates = [
        r"bin\ffmpeg\ffmpeg.exe",
        r"ffmpeg.exe",
        r"ffmpeg",
    ]
    p = resolve_exe(candidates)
    if not p:
        raise FileNotFoundError(
            "ffmpeg not found. Expected ./bin/ffmpeg/ffmpeg.exe (or on PATH)."
        )
    return str(p)


def resolve_ffprobe_exe() -> str:
    candidates = [
        r"bin\ffmpeg\ffprobe.exe",
        r"ffprobe.exe",
        r"ffprobe",
    ]
    p = resolve_exe(candidates)
    if not p:
        raise FileNotFoundError(
            "ffprobe not found. Expected ./bin/ffmpeg/ffprobe.exe (or on PATH)."
        )
    return str(p)




# ==============================
# About / third-party notices viewer
# ==============================

ABOUT_GLOBS = [
    "LICENSE*",
    "COPYING*",
    "NOTICE*",
    "THIRD_PARTY_NOTICES*",
    "README*",
]

def _collect_about_files() -> List[Path]:
    """Collect license/readme/notice files from project root and bin subfolders."""
    root = get_script_dir().parent if (get_script_dir().name.lower() == "app") else get_script_dir()
    paths: List[Path] = []

    def add_matches(base: Path) -> None:
        for g in ABOUT_GLOBS:
            for p in base.glob(g):
                if p.is_file():
                    paths.append(p)

    # Project root
    add_matches(root)

    # Known third-party folders
    add_matches(root / "bin" / "ffmpeg")
    add_matches(root / "bin" / "vgmstream-win")

    # De-dup while preserving order
    seen: Set[str] = set()
    out: List[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def show_about_menu() -> None:
    """Interactive viewer for bundled README/licence/notice files."""
    root = get_script_dir().parent if (get_script_dir().name.lower() == "app") else get_script_dir()
    files = _collect_about_files()

    while True:
        clear_screen()
        console.print(Panel.fit(
            "[bold]About / Licences / Readmes[/bold]\n"
            "View the bundled third-party notices and licence/readme files.\n"
            f"[dim]Root:[/dim] {root}",
            border_style="cyan"
        ))

        if not files:
            console.print("[yellow]No README/LICENCE/NOTICE files found.[/yellow]")
            Prompt.ask("Press Enter to return", default="")
            return

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("#", style="bold", width=4)
        table.add_column("File")
        table.add_column("Location", style="dim")
        for i, p in enumerate(files, start=1):
            rel = p
            try:
                rel = p.relative_to(root)
            except Exception:
                pass
            table.add_row(str(i), p.name, str(rel))
        table.add_row("0", "Return", "")
        console.print(table)

        sel = Prompt.ask("Open which file?", default="0")
        if sel.strip() == "0":
            return
        if not sel.isdigit():
            continue
        idx = int(sel)
        if idx < 1 or idx > len(files):
            continue

        p = files[idx - 1]
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            console.print(f"[red]Failed to read {p}: {e}[/red]")
            Prompt.ask("Press Enter", default="")
            continue

        clear_screen()
        console.print(Panel.fit(f"[bold]{p.name}[/bold]\n[dim]{p}[/dim]", border_style="green"))
        # Use Rich pager for long text
        with console.pager(styles=True):
            console.print(content)
        Prompt.ask("Press Enter to return", default="")

# ==============================
# Time helpers (timezone-aware)
# ==============================

def utc_now() -> datetime:
    """Timezone-aware UTC time (no datetime.utcnow() deprecation warnings)."""
    return datetime.now(timezone.utc)


def utc_iso_seconds() -> str:
    """Human readable UTC timestamp, seconds precision, includes +00:00."""
    return utc_now().isoformat(timespec="seconds")


def utc_stamp_compact() -> str:
    """Compact timestamp for filenames, ends with Z."""
    return utc_now().strftime("%Y%m%d_%H%M%SZ")


# ==============================
# Logging
# ==============================

class Logger:
    """Tiny file logger to help debug without spamming console."""
    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self._append(f"=== Session start {utc_iso_seconds()} ===\n")

    def _append(self, s: str) -> None:
        with self.logfile.open("a", encoding="utf-8") as f:
            f.write(s)

    def info(self, msg: str) -> None:
        self._append(f"[INFO] {msg}\n")

    def warn(self, msg: str) -> None:
        self._append(f"[WARN] {msg}\n")

    def error(self, msg: str) -> None:
        self._append(f"[ERROR] {msg}\n")


def new_session_logger() -> Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOGS_DIR / f"session_{utc_stamp_compact()}.log"
    return Logger(logfile)


# ==============================
# Data classes
# ==============================

@dataclass(frozen=True)
class DicScanSummary:
    stream_count: int
    containers: Dict[str, int]          # counts by extension (WIM/WIP/etc)
    wim_stream_indexes: List[int]       # stream indices pointing to WIM
    wim_filenames: List[str]            # corresponding WIM file names


@dataclass(frozen=True)
class WimStreamEntry:
    dic_path: Path
    stream_index: int
    wim_filename: str
    samples: int
    channels: int
    sample_rate: int
    target_bytes: int


@dataclass(frozen=True)
class Profile:
    I: float
    TP: float
    LRA: float
    method: str
    streams_analyzed: int
    created_utc: str
    injectable: bool
    containers: Dict[str, int]


@dataclass
class Plan:
    created_utc: str
    game_root: str
    dic_key: str
    dic_path: str
    injectable: bool
    profile_used: Optional[Dict[str, float]]
    mapping_mode: str
    mapping_seed: int
    bad_audio_policy: str
    music_folder: str
    total_wim_streams: int
    injection_filter_mode: str
    injection_filter_count: int
    assignments: List[Dict[str, Any]]
    notes: List[str]


# ==============================
# Subprocess helpers
# ==============================

def run_capture(cmd: List[str]) -> Tuple[str, str]:
    """Run subprocess and capture stdout/stderr. Raises if command fails."""
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout, p.stderr


def safe_run_capture(cmd: List[str]) -> Tuple[int, str, str]:
    """Run subprocess and capture output without raising."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def run(cmd: List[str]) -> None:
    """Run subprocess and raise if it fails."""
    subprocess.run(cmd, check=True)


def check_dependencies(vgmstream_exe: Path, ffmpeg_exe: str, log: Logger) -> None:
    """
    Make sure:
    - vgmstream exists
    - ffmpeg works
    """
    if not vgmstream_exe.exists():
        raise FileNotFoundError(
            f"Missing vgmstream CLI: {vgmstream_exe.resolve()}\n"
            f"Put vgmstream CLI next to this script (test.exe) or under ./bin/ (vgmstream-cli.exe)."
        )

    rc, out, err = safe_run_capture([ffmpeg_exe, "-version"])
    if rc != 0:
        raise RuntimeError("ffmpeg not found or not working. Install ffmpeg and ensure it's in PATH.")
    log.info("ffmpeg OK")
    log.info(f"ffmpeg -version first line: {(out.splitlines() or [''])[0]}")

    # Not all vgmstream builds support -V reliably; warn only.
    rc, out, err = safe_run_capture([str(vgmstream_exe), "-V"])
    if rc == 0:
        log.info(f"vgmstream -V ok")
    else:
        log.warn("vgmstream -V failed (non-fatal)")


# ==============================
# Simple GUI folder picker (tkinter)
# ==============================

def pick_folder_dialog(title: str, initial: Optional[str] = None) -> Optional[str]:
    """Open a folder picker window. If it fails, we fallback to manual input."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        kwargs = {"title": title}
        if initial and Path(initial).exists():
            kwargs["initialdir"] = initial
        path = filedialog.askdirectory(**kwargs)
        root.destroy()
        return path or None
    except Exception:
        return None


# ==============================
# Config (settings + cache)
# ==============================

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def get_mapping_seed(cfg: dict, dic_key: str) -> int:
    seeds = cfg.setdefault("mapping_seeds", {})
    if dic_key not in seeds:
        seeds[dic_key] = random.randint(1, 2_000_000_000)
        save_config(cfg)
    return int(seeds[dic_key])


def reroll_mapping_seed(cfg: dict, dic_key: str) -> int:
    cfg.setdefault("mapping_seeds", {})
    cfg["mapping_seeds"][dic_key] = random.randint(1, 2_000_000_000)
    save_config(cfg)
    return int(cfg["mapping_seeds"][dic_key])


def get_bad_audio_policy(cfg: dict) -> str:
    pol = cfg.get("bad_audio_policy", DEFAULT_BAD_AUDIO_POLICY)
    if pol not in ("strict", "skip", "fallback"):
        pol = DEFAULT_BAD_AUDIO_POLICY
    return pol


def set_bad_audio_policy(cfg: dict, pol: str) -> None:
    cfg["bad_audio_policy"] = pol
    save_config(cfg)


# ---- DIC scan cache (massive UX improvement) ----

def dic_cache_key(dic_path: Path) -> str:
    return str(dic_path).replace("\\", "/")


def dic_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def get_scan_cache(cfg: dict) -> dict:
    return cfg.setdefault("dic_scan_cache", {})


# ==============================
# Backups
# ==============================

def safe_relpath(path: Path, game_root: Path) -> Path:
    """Return path relative to game root if possible, else just filename."""
    try:
        return path.resolve().relative_to(game_root.resolve())
    except Exception:
        return Path(path.name)


def backup_file_once(src: Path, game_root: Path, log: Logger) -> Optional[Path]:
    """
    Copy src -> showdown_backups/<relative path> if it isn't backed up yet.
    This guarantees we never overwrite originals without a backup.
    """
    if not src.exists():
        log.warn(f"Backup requested but source missing: {src}")
        return None
    rel = safe_relpath(src, game_root)
    dest = BACKUP_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
        log.info(f"Backed up: {src} -> {dest}")
    return dest


def backup_presence_check(files: List[Path], game_root: Path) -> Tuple[int, int]:
    have = 0
    miss = 0
    for f in files:
        rel = safe_relpath(f, game_root)
        b = BACKUP_ROOT / rel
        if b.exists():
            have += 1
        else:
            miss += 1
    return have, miss


def restore_all_from_backups(game_root: Path, log: Logger) -> int:
    """Restore every backed-up file back into the game folder."""
    if not BACKUP_ROOT.exists():
        raise FileNotFoundError(f"No backups folder found: {BACKUP_ROOT.resolve()}")
    restored = 0
    for src in BACKUP_ROOT.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(BACKUP_ROOT)
        dest = game_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        restored += 1
    log.info(f"Restored {restored} files from backups into game root")
    return restored


# ==============================
# DIC scanning (vgmstream)
# ==============================

def find_dic_files(game_root: Path) -> List[Path]:
    return sorted(game_root.rglob("*.dic"))


def vgmstream_meta_text(vgmstream_exe: Path, dic_path: Path) -> str:
    out, _ = run_capture([str(vgmstream_exe), "-m", str(dic_path)])
    return out


def parse_stream_count(meta_text: str) -> int:
    for line in meta_text.splitlines():
        if "stream count:" in line:
            return int(line.split("stream count:")[1].strip())
    raise RuntimeError("Could not find 'stream count' in vgmstream metadata.")


def vgmstream_info_json(vgmstream_exe: Path, dic_path: Path, stream_index: int) -> dict:
    """
    Ask vgmstream for JSON info about one stream index inside a DIC.
    This is the call that can be slow if repeated many times.
    """
    out, _ = run_capture([str(vgmstream_exe), "-I", "-s", str(stream_index), str(dic_path)])
    return json.loads(out)


def extract_container_and_file(stream_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    vgmstream stream name often looks like:
      "01_fe_2ch.WIM/01_fe_2ch"
    We care about:
      - extension (WIM/WIP/etc)
      - the actual container filename (like "01_fe_2ch.WIM")
    """
    if not stream_name:
        return None, None
    head = stream_name.split("/", 1)[0]
    ext = Path(head).suffix.upper().lstrip(".")
    if ext:
        return ext, head
    return None, head


def scan_dic_summary(vgmstream_exe: Path, dic_path: Path, show_progress: bool = True) -> DicScanSummary:
    """
    Reads the DIC and finds:
    - total stream count
    - what containers appear (WIM/WIP/etc)
    - which streams point to WIM files
    """
    meta = vgmstream_meta_text(vgmstream_exe, dic_path)
    stream_count = parse_stream_count(meta)

    containers: Dict[str, int] = {}
    wim_indexes: List[int] = []
    wim_files: List[str] = []

    # Show progress so the user sees activity (even if it ends up being fast).
    progress_ctx = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) if show_progress else None

    if progress_ctx:
        with progress_ctx as progress:
            task = progress.add_task(f"Scanning streams in {dic_path.name}...", total=stream_count)
            for s in range(1, stream_count + 1):
                info = vgmstream_info_json(vgmstream_exe, dic_path, s)
                stream_info = info.get("streamInfo", {})
                name = stream_info.get("name", "")
                ext, head = extract_container_and_file(name)

                if ext:
                    containers[ext] = containers.get(ext, 0) + 1

                if ext == "WIM" and head and head.upper().endswith(".WIM"):
                    wim_indexes.append(s)
                    wim_files.append(head)

                progress.advance(task)
    else:
        for s in range(1, stream_count + 1):
            info = vgmstream_info_json(vgmstream_exe, dic_path, s)
            stream_info = info.get("streamInfo", {})
            name = stream_info.get("name", "")
            ext, head = extract_container_and_file(name)
            if ext:
                containers[ext] = containers.get(ext, 0) + 1
            if ext == "WIM" and head and head.upper().endswith(".WIM"):
                wim_indexes.append(s)
                wim_files.append(head)

    return DicScanSummary(stream_count, containers, wim_indexes, wim_files)


def enumerate_wim_entries(vgmstream_exe: Path, dic_path: Path, summary: DicScanSummary, show_progress: bool = True) -> List[WimStreamEntry]:
    """
    For each WIM stream, read metadata like samples/channels.
    We need this to know how many bytes each WIM file should contain.
    """
    entries: List[WimStreamEntry] = []
    total = len(summary.wim_stream_indexes)

    progress_ctx = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) if show_progress else None

    if progress_ctx:
        with progress_ctx as progress:
            task = progress.add_task(f"Reading WIM metadata ({dic_path.name})...", total=total)
            for s, wim_name in zip(summary.wim_stream_indexes, summary.wim_filenames):
                info = vgmstream_info_json(vgmstream_exe, dic_path, s)
                sample_rate = int(info.get("sampleRate", info.get("sample_rate", 0)) or 0)
                channels = int(info.get("channels", 0))
                samples = int(info.get("numberOfSamples", info.get("num_samples", 0)) or 0)
                target_bytes = samples * channels
                entries.append(WimStreamEntry(dic_path, s, wim_name, samples, channels, sample_rate, target_bytes))
                progress.advance(task)
    else:
        for s, wim_name in zip(summary.wim_stream_indexes, summary.wim_filenames):
            info = vgmstream_info_json(vgmstream_exe, dic_path, s)
            sample_rate = int(info.get("sampleRate", info.get("sample_rate", 0)) or 0)
            channels = int(info.get("channels", 0))
            samples = int(info.get("numberOfSamples", info.get("num_samples", 0)) or 0)
            target_bytes = samples * channels
            entries.append(WimStreamEntry(dic_path, s, wim_name, samples, channels, sample_rate, target_bytes))

    return entries


# ==============================
# musicPlayer.xml support
# ==============================

def find_musicplayer_xml(game_root: Path) -> Optional[Path]:
    p = game_root / "audio" / "musicPlayer.xml"
    return p if p.exists() else None


def escape_bare_ampersands(xml_text: str) -> str:
    """Fix raw '&' that break XML parsing (artist names, etc.)."""
    return re.sub(
        r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)',
        '&amp;',
        xml_text
    )


def strip_non_xml_banner_comments(xml_text: str) -> str:
    """
    Codemasters sometimes uses invalid 'banner comments' like:
      <!----------------------- ... ----------------------->
    These are NOT valid XML comments and will break parsers.
    """
    return re.sub(r'<!-{3,}[\s\S]*?-{3,}>', '', xml_text)


def parse_musicplayer_streams(xml_path: Path, log: Logger) -> Set[str]:
    """
    Return stream IDs from musicPlayer.xml.

    We try to parse as XML after cleaning banner comments and ampersands.
    If XML parsing still fails, we fall back to regex extraction.
    """
    import xml.etree.ElementTree as ET

    raw = xml_path.read_text(encoding="utf-8", errors="replace")
    cleaned = strip_non_xml_banner_comments(raw)
    cleaned = escape_bare_ampersands(cleaned)

    streams: Set[str] = set()

    try:
        root = ET.fromstring(cleaned)
        for elem in root.iter():
            if "stream" in elem.attrib:
                s = (elem.attrib.get("stream") or "").strip()
                if s:
                    streams.add(s.lstrip("_"))
        log.info(f"musicPlayer.xml streams found (XML parse): {len(streams)}")
        return streams
    except Exception as e:
        log.warn(f"Failed to parse musicPlayer.xml after cleaning, using regex fallback: {e}")

    # Fallback: pull out stream="..."
    for m in re.finditer(r'\bstream\s*=\s*"([^"]+)"', cleaned):
        s = m.group(1).strip()
        if s:
            streams.add(s.lstrip("_"))

    log.info(f"musicPlayer.xml streams found (regex fallback): {len(streams)}")
    return streams


def wim_filename_from_stream_id(stream_id: str) -> str:
    return f"{stream_id}.WIM"


def categorize_stream_id(stream_id: str) -> str:
    s = stream_id.lower()
    if "_fe_" in s:
        return "FE"
    if "_race_" in s:
        return "RACE"
    if "_replay_" in s:
        return "REPLAY"
    return "OTHER"


def build_xml_wim_filters(stream_ids: Set[str]) -> Dict[str, Set[str]]:
    """
    Build sets of WIM filenames based on stream types.
    Example output:
      XML_RACE -> {"01_race_2ch.WIM", ...}
    """
    all_wims = {wim_filename_from_stream_id(s) for s in stream_ids}
    fe = {wim_filename_from_stream_id(s) for s in stream_ids if categorize_stream_id(s) == "FE"}
    race = {wim_filename_from_stream_id(s) for s in stream_ids if categorize_stream_id(s) == "RACE"}
    replay = {wim_filename_from_stream_id(s) for s in stream_ids if categorize_stream_id(s) == "REPLAY"}
    other = {wim_filename_from_stream_id(s) for s in stream_ids if categorize_stream_id(s) == "OTHER"}
    return {
        "XML_ALL": all_wims,
        "XML_FE": fe,
        "XML_RACE": race,
        "XML_REPLAY": replay,
        "XML_OTHER": other
    }


# ==============================
# Profiles (loudness targets)
# ==============================

def load_profiles(path: Path) -> Dict[str, Profile]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, Profile] = {}
        for k, v in raw.items():
            out[k] = Profile(
                I=float(v["I"]),
                TP=float(v["TP"]),
                LRA=float(v["LRA"]),
                method=str(v.get("method", "unknown")),
                streams_analyzed=int(v.get("streams_analyzed", 0)),
                created_utc=str(v.get("created_utc", "")),
                injectable=bool(v.get("injectable", False)),
                containers=dict(v.get("containers", {})),
            )
        return out
    except Exception:
        return {}


def save_profiles(profiles: Dict[str, Profile], path: Path) -> None:
    data = {
        k: {
            "I": p.I,
            "TP": p.TP,
            "LRA": p.LRA,
            "method": p.method,
            "streams_analyzed": p.streams_analyzed,
            "created_utc": p.created_utc,
            "injectable": p.injectable,
            "containers": p.containers,
        }
        for k, p in sorted(profiles.items(), key=lambda x: x[0])
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_profiles_as_python_dict(profiles: Dict[str, Profile], out_path: Path) -> None:
    lines = []
    lines.append("# Auto-generated from showdown_profiles.json")
    lines.append("# Key: DIC path relative to game root")
    lines.append("PROFILES = {")
    for k, p in sorted(profiles.items(), key=lambda x: x[0]):
        lines.append(
            f'  {k!r}: {{"I": {p.I:.3f}, "TP": {p.TP:.2f}, "LRA": {p.LRA:.1f}}}, '
            f'# injectable={p.injectable} containers={p.containers}'
        )
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile_key_for_dic(dic_path: Path, game_root: Path) -> str:
    return str(safe_relpath(dic_path, game_root)).replace("\\", "/")


def vgmstream_decode_wav(vgmstream_exe: Path, dic_path: Path, stream_index: int, out_wav: Path) -> None:
    run([str(vgmstream_exe), "-s", str(stream_index), "-o", str(out_wav), str(dic_path)])


def ffmpeg_loudnorm_analyze(ffmpeg_exe: str, wav_path: Path, I: float, TP: float, LRA: float) -> dict:
    """
    Ask ffmpeg's loudnorm filter to analyze loudness.
    We use this to learn the game's typical volume so we can match it.
    """
    cmd = [
        ffmpeg_exe, "-hide_banner", "-i", str(wav_path),
        "-filter:a", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
        "-f", "null", "NUL"
    ]
    _, err = run_capture(cmd)
    m = re.search(r"\{[\s\S]*\}", err)
    if not m:
        raise RuntimeError("Could not parse loudnorm JSON from ffmpeg output.")
    return json.loads(m.group(0))


def compute_profile_for_dic(
    vgmstream_exe: Path,
    ffmpeg_exe: str,
    dic_path: Path,
    summary: DicScanSummary,
    tp: float,
    lra: float,
    streams_to_analyze: Optional[int],
) -> Profile:
    """
    Build a profile by measuring some of the original streams:
    - Analyze each stream -> get input_i (LUFS)
    - Take the median of those LUFS values -> robust "typical loudness"
    """
    stream_count = summary.stream_count
    analyze_n = stream_count if streams_to_analyze is None else max(1, min(streams_to_analyze, stream_count))
    placeholder_I = -23.0
    input_is: List[float] = []

    with tempfile.TemporaryDirectory(prefix="showdown_scan_") as tmp:
        tmpdir = Path(tmp)
        for s in range(1, analyze_n + 1):
            wav_path = tmpdir / f"scan_{s:04d}.wav"
            vgmstream_decode_wav(vgmstream_exe, dic_path, s, wav_path)
            j = ffmpeg_loudnorm_analyze(ffmpeg_exe, wav_path, placeholder_I, tp, lra)
            input_is.append(float(j["input_i"]))

    target_I = float(statistics.median(input_is))
    injectable = summary.containers.get("WIM", 0) > 0
    return Profile(
        I=target_I,
        TP=tp,
        LRA=lra,
        method="median_of_originals",
        streams_analyzed=analyze_n,
        created_utc=utc_iso_seconds(),
        injectable=injectable,
        containers=summary.containers,
    )


def autotune_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(DEFAULT_MAX_WORKERS_CAP, max(2, cpu // 2)))


def advanced_full_scan_build_profiles(
    game_root: Path,
    vgmstream_exe: Path,
    ffmpeg_exe: str,
    tp: float,
    lra: float,
    fast_streams: int,
    accurate: bool,
    workers: int,
    log: Logger
) -> Dict[str, Profile]:
    """
    Scan every DIC in the game folder:
    - Determine containers + WIM presence
    - Build loudness profile (fast or accurate)
    """
    clear_screen()
    dic_files = find_dic_files(game_root)
    if not dic_files:
        raise RuntimeError("No .dic files found under game root.")

    streams_to_analyze = None if accurate else fast_streams
    workers = max(1, min(workers, 8))

    log.info(f"Advanced scan start: count={len(dic_files)} mode={'accurate' if accurate else 'fast'} workers={workers}")

    def worker(dic_path: Path) -> Tuple[str, Optional[Profile], str]:
        key = profile_key_for_dic(dic_path, game_root)
        try:
            summ = scan_dic_summary(vgmstream_exe, dic_path, show_progress=False)
            prof = compute_profile_for_dic(vgmstream_exe, ffmpeg_exe, dic_path, summ, tp, lra, streams_to_analyze)
            return key, prof, ""
        except Exception as e:
            return key, None, str(e)

    profiles: Dict[str, Profile] = {}
    failures: List[Tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Full scan: building profiles...", total=len(dic_files))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, d) for d in dic_files]
            for fut in as_completed(futs):
                key, prof, err = fut.result()
                if prof is not None:
                    profiles[key] = prof
                else:
                    failures.append((key, err))
                    log.warn(f"Scan error: {key}: {err}")
                progress.advance(task)

    injectable = sum(1 for p in profiles.values() if p.injectable)
    console.print(f"[green]Profiles built:[/green] {len(profiles)}  (Injectable: {injectable})")
    if failures:
        console.print(f"[yellow]Failures:[/yellow] {len(failures)} (showing first 10)")
        for k, e in failures[:10]:
            console.print(f"  - {k}: {e}")

    log.info(f"Advanced scan done: built={len(profiles)} injectable={injectable} failures={len(failures)}")
    return profiles


# ==============================
# Custom music mapping
# ==============================

def is_audio_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS


def build_song_list(source_folder: Path) -> List[Path]:
    """
    Gather the user's music files.
    We also ignore common junk like desktop.ini (even if it slips in).
    """
    songs = []
    for p in sorted(source_folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.lower() == "desktop.ini":
            continue
        if is_audio_file(p):
            songs.append(p)

    if not songs:
        raise RuntimeError(
            f"No audio files found in: {source_folder}\n"
            f"Accepted extensions: {', '.join(sorted(AUDIO_EXTS))}"
        )
    return songs


def pick_mapping_mode() -> str:
    clear_screen()
    console.print(Panel(
        "Mapping decides how your custom songs are assigned to the game's WIM streams.\n\n"
        "1) Sequential: alphabetical order (A->Z)\n"
        "2) Shuffle: random order BUT repeatable using a seed\n"
        "3) Round-robin: shuffled list, wraps around if you have fewer songs than streams\n",
        title="Mapping Mode", border_style="cyan"
    ))
    choice = Prompt.ask("Choose", default="2")
    return {"1": "sequential", "2": "shuffle", "3": "roundrobin"}.get(choice, "shuffle")


def build_seeded_mapping(songs: List[Path], mode: str, seed: int) -> List[Path]:
    if mode == "sequential":
        return songs
    rng = random.Random(seed)
    shuffled = songs[:]
    rng.shuffle(shuffled)
    return shuffled


def assign_song_for_stream(mapped_songs: List[Path], stream_index: int) -> Path:
    return mapped_songs[(stream_index - 1) % len(mapped_songs)]


# ==============================
# Conversion / injection helpers
# ==============================

def fit_bytes(raw: bytes, target_len: int) -> bytes:
    """Trim or pad audio bytes to EXACTLY match the original file size."""
    if len(raw) > target_len:
        return raw[:target_len]
    if len(raw) < target_len:
        return raw + (MULAW_SILENCE_BYTE * (target_len - len(raw)))
    return raw


def ffmpeg_simple_mulaw(ffmpeg_exe: str, src_audio: Path, out_mulaw: Path) -> Tuple[bool, str]:
    """Convert audio to 48kHz stereo mulaw with no loudness matching."""
    cmd = [
        ffmpeg_exe, "-hide_banner", "-y", "-i", str(src_audio),
        "-ar", "48000", "-ac", "2",
        "-f", "mulaw", str(out_mulaw)
    ]
    rc, _, err = safe_run_capture(cmd)
    return rc == 0, err.strip()[-400:]


def ffmpeg_two_pass_mulaw(ffmpeg_exe: str, src_audio: Path, out_mulaw: Path, prof: Profile) -> Tuple[bool, str]:
    """
    Two-pass loudnorm:
    Pass 1: analyze loudness
    Pass 2: apply correction to reach target loudness
    """
    devnull = "NUL"
    cmd1 = [
        ffmpeg_exe, "-hide_banner", "-y", "-i", str(src_audio),
        "-filter:a", f"loudnorm=I={prof.I}:TP={prof.TP}:LRA={prof.LRA}:print_format=json",
        "-f", "null", devnull
    ]
    rc, _, err = safe_run_capture(cmd1)
    if rc != 0:
        return False, err.strip()[-600:]

    m = re.search(r"\{[\s\S]*\}", err)
    if not m:
        return False, "Could not parse loudnorm JSON."

    j = json.loads(m.group(0))
    filt = (
        f"loudnorm=I={prof.I}:TP={prof.TP}:LRA={prof.LRA}"
        f":measured_I={j['input_i']}"
        f":measured_TP={j['input_tp']}"
        f":measured_LRA={j['input_lra']}"
        f":measured_thresh={j['input_thresh']}"
        f":offset={j['target_offset']}"
        f":linear=true:print_format=summary"
    )

    cmd2 = [
        ffmpeg_exe, "-hide_banner", "-y", "-i", str(src_audio),
        "-filter:a", filt,
        "-ar", "48000", "-ac", "2",
        "-f", "mulaw", str(out_mulaw)
    ]
    rc2, _, err2 = safe_run_capture(cmd2)
    return rc2 == 0, err2.strip()[-600:]


def convert_song_to_mulaw(
    ffmpeg_exe: str,
    song: Path,
    out_mulaw: Path,
    prof: Optional[Profile],
    bad_policy: str,
    log: Logger
) -> bool:
    """
    Convert one custom song into game format.
    - If a profile is available, try loudness matching.
    - If conversion fails:
        strict   -> abort
        skip     -> skip this song
        fallback -> try basic conversion
    """
    if prof is None:
        ok, msg = ffmpeg_simple_mulaw(ffmpeg_exe, song, out_mulaw)
        if ok:
            return True
        log.warn(f"Convert failed (no profile): {song.name} :: {msg}")
        if bad_policy == "strict":
            raise RuntimeError(f"ffmpeg convert failed for {song.name}: {msg}")
        return False

    ok, msg = ffmpeg_two_pass_mulaw(ffmpeg_exe, song, out_mulaw, prof)
    if ok:
        return True

    log.warn(f"Loudnorm convert failed: {song.name} :: {msg}")

    if bad_policy == "strict":
        raise RuntimeError(f"ffmpeg loudnorm failed for {song.name}: {msg}")

    if bad_policy == "skip":
        return False

    ok2, msg2 = ffmpeg_simple_mulaw(ffmpeg_exe, song, out_mulaw)
    if ok2:
        log.info(f"Fallback convert succeeded for: {song.name}")
        return True

    log.warn(f"Fallback convert failed: {song.name} :: {msg2}")
    return False


def cleanup_temp_artifacts(script_dir: Path, log: Logger) -> None:
    """Remove temporary mulaw file if it exists."""
    p = script_dir / TEMP_MULAW_NAME
    if p.exists():
        try:
            p.unlink()
            log.info(f"Removed temp artifact: {p}")
        except Exception:
            log.warn(f"Failed to remove temp artifact: {p}")


# ==============================
# Injection filter UI
# ==============================

def choose_injection_filter_mode(xml_filters: Optional[Dict[str, Set[str]]]) -> Tuple[str, Optional[Set[str]]]:
    """
    This chooses WHICH WIM streams are eligible for replacement inside the selected DIC.

    - DIC_ALL: replace all WIM streams found in this DIC
    - XML_*: only replace streams that appear in musicPlayer.xml (FE/RACE/REPLAY/etc)
    """
    clear_screen()
    console.print(Panel(
        "Injection Filter (Important!)\n\n"
        "You already chose a DIC pack (like win_mus_race).\n"
        "Now this filter decides WHICH WIM tracks inside that pack will be replaced.\n\n"
        "If you're unsure, choose:\n"
        "  1) All WIM streams in this DIC\n",
        title="Injection Filter Explained", border_style="cyan"
    ))

    if not xml_filters or not xml_filters.get("XML_ALL"):
        console.print("1) All WIM streams in this DIC (default)\n")
        Prompt.ask("Choose", choices=["1"], default="1")
        return "DIC_ALL", None

    console.print("1) All WIM streams in this DIC (default)")
    console.print("2) Only streams referenced by musicPlayer.xml (ALL)")
    console.print("3) Only FE streams (menu/FE) from musicPlayer.xml")
    console.print("4) Only RACE streams from musicPlayer.xml")
    console.print("5) Only REPLAY streams from musicPlayer.xml")
    console.print("6) Only OTHER streams from musicPlayer.xml")
    c = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5", "6"], default="1")

    if c == "1":
        return "DIC_ALL", None
    if c == "2":
        return "XML_ALL", xml_filters["XML_ALL"]
    if c == "3":
        return "XML_FE", xml_filters["XML_FE"]
    if c == "4":
        return "XML_RACE", xml_filters["XML_RACE"]
    if c == "5":
        return "XML_REPLAY", xml_filters["XML_REPLAY"]
    return "XML_OTHER", xml_filters["XML_OTHER"]


def filter_wim_entries_by_allowed_names(entries: List[WimStreamEntry], allowed_wims: Optional[Set[str]]) -> List[WimStreamEntry]:
    if allowed_wims is None:
        return entries
    allow_upper = {a.upper() for a in allowed_wims}
    return [e for e in entries if e.wim_filename.upper() in allow_upper]


# ==============================
# Plan export
# ==============================

def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_ROOT.mkdir(exist_ok=True)


def write_plan(plan: Plan) -> Tuple[Path, Path]:
    safe_tag = plan.dic_key.replace("/", "_").replace("\\", "_").replace(":", "")
    json_path = PLANS_DIR / f"plan_{safe_tag}_{utc_stamp_compact()}.json"
    txt_path = PLANS_DIR / f"plan_{safe_tag}_{utc_stamp_compact()}.txt"

    json_path.write_text(json.dumps(plan.__dict__, indent=2), encoding="utf-8")

    lines = []
    lines.append("DiRT Showdown Music Manager - Plan")
    lines.append("=" * 40)
    lines.append(f"Created UTC: {plan.created_utc}")
    lines.append(f"Game root:   {plan.game_root}")
    lines.append(f"DIC key:     {plan.dic_key}")
    lines.append(f"DIC path:    {plan.dic_path}")
    lines.append(f"Injectable:  {plan.injectable}")
    lines.append(f"Profile:     {plan.profile_used if plan.profile_used else 'None (no loudness match)'}")
    lines.append(f"Mapping:     {plan.mapping_mode}  seed={plan.mapping_seed}")
    lines.append(f"Bad policy:  {plan.bad_audio_policy}")
    lines.append(f"Music dir:   {plan.music_folder}")
    lines.append(f"Filter:      {plan.injection_filter_mode} (streams={plan.injection_filter_count})")
    lines.append("")
    if plan.notes:
        lines.append("NOTES:")
        for n in plan.notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("Assignments (stream -> WIM -> song):")
    for a in plan.assignments[:200]:
        lines.append(f"- s{a['stream_index']:03d}  {a['wim_filename']}  <=  {a['song_name']}")
    if len(plan.assignments) > 200:
        lines.append(f"... ({len(plan.assignments)-200} more)")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, txt_path


# ==============================
# ZIP builder
# ==============================

def make_zip_of_modified_wims(modified_wims: List[Path], output_root: Path, game_root: Path, dic_key: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    tag = dic_key.replace("/", "_").replace("\\", "_").replace(":", "")
    zip_path = output_root / f"showdown_pack_{tag}_{utc_stamp_compact()}.zip"

    uniq: List[Path] = []
    seen = set()
    for p in modified_wims:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)

    install_bat = (
        "@echo off\n"
        "setlocal\n"
        "echo DiRT Showdown Custom Music Pack Installer\n"
        "echo.\n"
        "echo Extract this ZIP into your DiRT Showdown folder (the one with 'audio').\n"
        "echo Overwrite when prompted.\n"
        "echo.\n"
        "pause\n"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for wim in uniq:
            rel = safe_relpath(wim, game_root)
            z.write(wim, arcname=str(rel).replace("\\", "/"))
        z.writestr("install.bat", install_bat)
        z.writestr(
            "README.txt",
            "DiRT Showdown - Custom Music Pack (WIM only)\n\n"
            f"Created (UTC): {utc_iso_seconds()}\n\n"
            "Install:\n"
            "1) Open your DiRT Showdown ROOT folder (the one containing 'audio').\n"
            "2) Extract this ZIP into that folder.\n"
            "3) Overwrite when prompted.\n"
        )

    return zip_path


# ==============================
# WAV cleanup
# ==============================

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def find_wavs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    wavs = [p for p in base.rglob("*.wav") if p.is_file()]
    wavs += [p for p in base.rglob("*.WAV") if p.is_file()]
    uniq = []
    seen = set()
    for p in wavs:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq



def find_wavs_excluding(base: Path, exclude_dirs: List[Path]) -> List[Path]:
    """Find *.wav under base, excluding any paths that live under exclude_dirs."""
    base = base.resolve()
    ex = []
    for d in exclude_dirs:
        if d:
            try:
                ex.append(d.resolve())
            except Exception:
                pass

    out: List[Path] = []
    for p in find_wavs(base):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        skip = False
        for d in ex:
            try:
                if rp == d or d in rp.parents:
                    skip = True
                    break
            except Exception:
                pass
        if not skip:
            out.append(p)
    return out


def cleanup_recent_wavs(
    targets: List[Path],
    since_dt: datetime,
    exclude_dirs: List[Path],
    log: Optional["Logger"] = None
) -> int:
    """Delete WAV files modified since since_dt under targets, excluding exclude_dirs."""
    since_epoch = since_dt.timestamp()
    deleted = 0
    for t in targets:
        if not t or not t.exists():
            continue
        for wav in find_wavs_excluding(t, exclude_dirs):
            try:
                if wav.stat().st_mtime >= since_epoch:
                    wav.unlink(missing_ok=True)
                    deleted += 1
            except Exception as e:
                if log:
                    log.warn(f"WAV cleanup failed: {wav} :: {e}")
    if log:
        log.info(f"Auto WAV cleanup: deleted={deleted}")
    return deleted


def cleanup_wavs(game_root: Path, script_dir: Path, log: Logger) -> None:
    clear_screen()
    console.print(Panel(
        "WAV Cleanup\n\n"
        "Sometimes tools create WAV files during scanning/decoding.\n"
        "You said: there should be NO WAV files left behind.\n\n"
        "Options:\n"
        "1) Delete ALL *.wav under game root + tools folder\n"
        "2) Delete only WAVs newer than N days (safer)\n",
        title="Advanced: WAV Cleanup", border_style="yellow"
    ))
    mode = Prompt.ask("Choose", choices=["1", "2"], default="2")
    days = 0
    cutoff = None
    if mode == "2":
        days = IntPrompt.ask("Delete WAVs newer than how many days?", default=7)
        cutoff = datetime.now() - timedelta(days=days)

    targets = [game_root, script_dir]
    wavs: List[Path] = []
    for t in targets:
        wavs.extend(find_wavs(t))

    if cutoff:
        wavs = [p for p in wavs if datetime.fromtimestamp(p.stat().st_mtime) >= cutoff]

    if not wavs:
        console.print("[green]No WAV files found for deletion.[/green]")
        return

    total = sum((p.stat().st_size for p in wavs), 0)
    table = Table(title="WAV Deletion Preview", box=box.SIMPLE)
    table.add_column("Files", style="bold cyan")
    table.add_column("Total size", style="magenta")
    table.add_row(str(len(wavs)), human_bytes(total))
    console.print(table)

    console.print("[dim]Showing up to 20 WAV paths:[/dim]")
    for p in wavs[:20]:
        console.print(f"  {p}")

    warn = "Delete ALL wavs" if mode == "1" else f"Delete wavs newer than {days} day(s)"
    if not Confirm.ask(f"{warn} now?", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Deleting WAV files...", total=len(wavs))
        deleted = 0
        failed = 0
        for p in wavs:
            try:
                p.unlink()
                deleted += 1
            except Exception:
                failed += 1
            progress.advance(task)

    console.print(f"[green]Deleted:[/green] {deleted}  [yellow]Failed:[/yellow] {failed}")
    log.info(f"WAV cleanup: deleted={deleted} failed={failed} mode={mode} cutoff_days={days if cutoff else 'ALL'}")


# ==============================
# UI helpers
# ==============================

def pick_game_root(cfg: dict) -> Path:
    clear_screen()
    console.print(Panel(
        "Select your DiRT Showdown install folder.\n"
        "This folder should contain an 'audio' folder inside it.\n",
        title="Game Folder", border_style="cyan"
    ))
    last = cfg.get("last_game_root")
    picked = pick_folder_dialog("Select DiRT Showdown install folder", initial=last)
    if not picked:
        picked = Prompt.ask("Folder picker cancelled. Type the game folder path")
    root = Path(picked).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    cfg["last_game_root"] = str(root)
    save_config(cfg)
    return root


def pick_music_folder(cfg: dict) -> Path:
    clear_screen()
    console.print(Panel(
        "Select your custom music folder.\n"
        "Put ONLY your music files in this folder (mp3/wav/flac/etc).\n",
        title="Custom Music Folder", border_style="cyan"
    ))
    last = cfg.get("last_music_folder")
    picked = pick_folder_dialog("Select custom music folder", initial=last)
    if not picked:
        picked = Prompt.ask("Folder picker cancelled. Type the music folder path")
    folder = Path(picked).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    cfg["last_music_folder"] = str(folder)
    save_config(cfg)
    return folder


def prompt_defaults_tp_lra() -> Tuple[float, float]:
    console.print(f"[dim]Defaults: TP={DEFAULT_TP} dBTP | LRA={DEFAULT_LRA}[/dim]")
    tp_in = Prompt.ask("True peak limit (dBTP) [Enter=default]", default="")
    lra_in = Prompt.ask("Target LRA [Enter=default]", default="")
    tp = float(tp_in) if tp_in.strip() else DEFAULT_TP
    lra = float(lra_in) if lra_in.strip() else DEFAULT_LRA
    return tp, lra


def list_dic_choices(dic_files: List[Path]) -> None:
    table = Table(title="Found .DIC packs", box=box.SIMPLE, show_lines=True)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("DIC file", style="white")
    table.add_column("Folder", style="dim")
    for i, d in enumerate(dic_files, 1):
        table.add_row(str(i), d.name, str(d.parent))
    console.print(table)


def choose_dic(dic_files: List[Path]) -> Path:
    list_dic_choices(dic_files)
    idx = IntPrompt.ask("Choose DIC number", default=1)
    if idx < 1 or idx > len(dic_files):
        raise ValueError("Invalid selection")
    return dic_files[idx - 1]


def show_wim_stream_preview(entries: List[WimStreamEntry], limit: int = 16) -> None:
    table = Table(title=f"WIM Streams (showing first {min(limit, len(entries))})",
                  box=box.SIMPLE, show_lines=False)
    table.add_column("Stream", style="bold cyan", width=6)
    table.add_column("WIM", style="white")
    table.add_column("Est. length", style="magenta", width=10)
    table.add_column("Bytes", style="green", justify="right")
    for e in entries[:limit]:
        sr = e.sample_rate or 48000
        seconds = e.samples / sr if sr else 0
        table.add_row(str(e.stream_index), e.wim_filename, f"{seconds:0.1f}s", f"{e.target_bytes:,}")
    console.print(table)


# ==============================
# Injection core (WIM-only)
# ==============================

def inject_wim_only(
    game_root: Path,
    vgmstream_exe: Path,
    ffmpeg_exe: str,
    dic_path: Path,
    dic_key: str,
    prof: Optional[Profile],
    music_folder: Path,
    mapping_mode: str,
    mapping_seed: int,
    bad_audio_policy: str,
    dry_run: bool,
    injection_filter_mode: str,
    allowed_wims: Optional[Set[str]],
    log: Logger
) -> Tuple[List[Path], Plan]:
    """
    This is the main "do the injection" function.
    It will:
      1) Backup the DIC
      2) Find WIM streams and WIM files
      3) Convert your music to mulaw
      4) Overwrite each WIM file with converted bytes (after padding/trimming)
    """
    clear_screen()
    log.info(f"Inject start: {dic_key}")

    # Always back up the DIC
    backup_file_once(dic_path, game_root, log)

    # Summarize streams
    summ = scan_dic_summary(vgmstream_exe, dic_path, show_progress=True)
    if summ.containers.get("WIM", 0) <= 0:
        raise RuntimeError(f"Non-injectable pack (no WIM streams). Containers: {summ.containers}")

    # Build entries for WIM streams
    all_entries = enumerate_wim_entries(vgmstream_exe, dic_path, summ, show_progress=True)
    entries = filter_wim_entries_by_allowed_names(all_entries, allowed_wims)
    if not entries:
        raise RuntimeError(
            f"No WIM entries matched your filter ({injection_filter_mode}).\n"
            f"Total WIM entries in DIC: {len(all_entries)}"
        )

    # Gather custom songs
    songs = build_song_list(music_folder)
    mapped_songs = build_seeded_mapping(songs, mapping_mode, mapping_seed)

    # Build an "injection plan" (useful for debugging and replayability)
    assignments: List[Dict[str, Any]] = []
    wim_dir = dic_path.parent

    for e in entries:
        wim_path = wim_dir / e.wim_filename
        song = assign_song_for_stream(mapped_songs, e.stream_index)
        assignments.append({
            "stream_index": e.stream_index,
            "wim_filename": e.wim_filename,
            "wim_path": str(wim_path),
            "song": str(song),
            "song_name": song.name,
            "target_bytes": e.target_bytes,
            "sample_rate": e.sample_rate,
            "channels": e.channels,
            "samples": e.samples,
        })

    plan = Plan(
        created_utc=utc_iso_seconds(),
        game_root=str(game_root),
        dic_key=dic_key,
        dic_path=str(dic_path),
        injectable=True,
        profile_used=None if prof is None else {"I": prof.I, "TP": prof.TP, "LRA": prof.LRA},
        mapping_mode=mapping_mode,
        mapping_seed=mapping_seed,
        bad_audio_policy=bad_audio_policy,
        music_folder=str(music_folder),
        total_wim_streams=len(entries),
        injection_filter_mode=injection_filter_mode,
        injection_filter_count=len(entries),
        assignments=assignments,
        notes=[]
    )

    # Backup notice
    wim_paths = [wim_dir / e.wim_filename for e in entries]
    have_b, miss_b = backup_presence_check([dic_path] + wim_paths, game_root)
    if miss_b > 0:
        plan.notes.append(f"NOTE: {miss_b} files had no backups yet; they will be backed up before overwrite.")
    log.info(f"Backup presence check: have={have_b} missing={miss_b}")

    modified: List[Path] = []
    script_dir = RUNTIME_ROOT
    temp_mulaw = script_dir / TEMP_MULAW_NAME

    op_start = datetime.now()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Injecting WIM files...", total=len(entries))

            for e in entries:
                wim_path = wim_dir / e.wim_filename
                song = assign_song_for_stream(mapped_songs, e.stream_index)

                progress.update(task, description=f"{wim_path.name} <= {song.name}")

                if not wim_path.exists():
                    msg = f"Missing WIM (skipped): {wim_path}"
                    console.print(f"[yellow]WARN[/yellow] {msg}")
                    log.warn(msg)
                    progress.advance(task)
                    continue

                if dry_run:
                    modified.append(wim_path)
                    progress.advance(task)
                    continue

                # Backup before overwrite
                backup_file_once(wim_path, game_root, log)

                # Convert
                ok = convert_song_to_mulaw(ffmpeg_exe, song, temp_mulaw, prof, bad_audio_policy, log)
                if not ok:
                    msg = f"Skipped (bad audio): {song.name}"
                    console.print(f"[yellow]SKIP[/yellow] {msg}")
                    log.warn(msg)
                    progress.advance(task)
                    continue

                # Read bytes, trim/pad to exact size, then write to WIM
                raw = temp_mulaw.read_bytes()
                fixed = fit_bytes(raw, e.target_bytes)
                wim_path.write_bytes(fixed)

                modified.append(wim_path)
                progress.advance(task)

    finally:
        cleanup_temp_artifacts(script_dir, log)
        # Auto-clean WAVs created during this injection (safe: excludes user music/output/backups)
        try:
            cleanup_recent_wavs(
                targets=[game_root, RUNTIME_ROOT, bin_dir()],
                since_dt=op_start,
                exclude_dirs=[music_folder, OUTPUT_ROOT, BACKUP_ROOT],
                log=log,
            )
        except Exception as _e:
            log.warn(f"Auto WAV cleanup failed (non-fatal): {_e}")

    log.info(f"Inject done: modified={len(modified)}")
    return modified, plan


# ==============================
# Main program
# ==============================


def main() -> None:
    ensure_dirs()
    log = new_session_logger()

    session_start = datetime.now()
    current_game_root: Optional[Path] = None
    current_music_folder: Optional[Path] = None
    clear_screen()
    script_dir = Path(__file__).resolve().parent
    cleanup_temp_artifacts(script_dir, log)

    cfg = load_config()
    vgmstream_exe = resolve_vgmstream_exe()
    ffmpeg_exe = resolve_ffmpeg_exe()

    console.print(Panel(
        "DiRT Showdown Music Manager\n"
        "Beginner-friendly edition.\n"
        "Includes clear-screen, caching, progress bars, XML filters.\n",
        title="Showdown Music Manager", border_style="green"
    ))

    try:
        check_dependencies(vgmstream_exe, ffmpeg_exe, log)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        log.error(str(e))
        return

    console.print("\n[green]Startup checks complete.[/green]")
    pause("Press Enter to select DiRT Showdown Install Folder...")

    # Choose and VALIDATE game root
    while True:
        try:
            game_root = pick_game_root(cfg)
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            log.error(str(e))
            return

        audio_dir = game_root / "audio"
        dic_files = find_dic_files(game_root)

        if not audio_dir.exists():
            console.print(Panel(
                "[red]Invalid Folder Selected[/red]\n\n"
                "This does not appear to be a DiRT Showdown folder.\n"
                "The folder must contain an 'audio' directory.\n",
                title="Folder Validation Failed",
                border_style="red"
            ))

        elif not dic_files:
            console.print(Panel(
                "[red]Invalid Folder Selected[/red]\n\n"
                "No .dic files were found in this folder.\n"
                "This is not a valid DiRT Showdown install directory.\n",
                title="Folder Validation Failed",
                border_style="red"
            ))

        else:
            break  # valid folder

        choice = Prompt.ask(
            "Try selecting the folder again or Exit?",
            choices=["retry", "exit"],
            default="retry"
        )
        if choice == "exit":
            console.print("[yellow]Exiting program.[/yellow]")
            return


    current_game_root = game_root

    # Load musicPlayer.xml if present
    xml_path = find_musicplayer_xml(game_root)
    xml_filters: Optional[Dict[str, Set[str]]] = None
    if xml_path:
        streams = parse_musicplayer_streams(xml_path, log)
        xml_filters = build_xml_wim_filters(streams)
        console.print(Panel(
            f"Found: {xml_path}\n"
            f"Streams in XML: {len(streams)}\n"
            f"WIM candidates from XML: {len(xml_filters['XML_ALL'])}",
            title="musicPlayer.xml loaded", border_style="cyan"
        ))
    else:
        console.print(Panel(
            "musicPlayer.xml not found at <game_root>/audio/musicPlayer.xml\n"
            "XML filtering will be disabled.",
            title="musicPlayer.xml not found", border_style="yellow"
        ))
    pause()

    dic_files = find_dic_files(game_root)
    if not dic_files:
        console.print("[red]No .dic files found under game root.[/red]")
        log.error("No .dic files found")
        return

    profiles = load_profiles(PROFILES_JSON)
    bad_policy = get_bad_audio_policy(cfg)
    workers_default = autotune_workers()

    # Main menu loop
    while True:
        clear_screen()
        xml_status = (
            f"Loaded ({len(xml_filters.get('XML_ALL', set()))} streams)"
            if (xml_filters is not None and isinstance(xml_filters, dict) and len(xml_filters.get('XML_ALL', set())) > 0)
            else "Not loaded"
        )
        render_header("DiRT Showdown Music Manager", game_dir=str(game_root), xml_status=xml_status)
        console.print("1) Inject (choose one DIC) [WIM only]")
        console.print("2) Re-roll mapping for a DIC (new random assignment)")
        console.print("3) Full Scan & Build Profiles (all DICs)")
        console.print("4) WAV Cleanup (delete leftover .wav files)")
        console.print("5) Restore ALL files from backups")
        console.print("6) Settings (bad audio policy)")
        console.print("7) Reload musicPlayer.xml")
        console.print("8) About / Licences / Readmes")
        console.print("9) Exit")
        choice = Prompt.ask("Choose", default="1")

        if choice == "9":
            # On exit, clean any WAVs created during this session (safe)
            if current_game_root is not None:
                try:
                    cleanup_recent_wavs(
                        targets=[current_game_root, RUNTIME_ROOT, bin_dir()],
                        since_dt=session_start,
                        exclude_dirs=[current_music_folder or Path(""), OUTPUT_ROOT, BACKUP_ROOT],
                        log=log,
                    )
                except Exception as _e:
                    log.warn(f"Exit WAV cleanup failed (non-fatal): {_e}")
            break

        if choice == "8":
            show_about_menu()
            continue

        if choice == "7":
            clear_screen()
            xml_path = find_musicplayer_xml(game_root)
            if xml_path:
                streams = parse_musicplayer_streams(xml_path, log)
                xml_filters = build_xml_wim_filters(streams)
                console.print(f"[green]Reloaded XML.[/green] Streams={len(streams)} WIM candidates={len(xml_filters['XML_ALL'])}")
            else:
                xml_filters = None
                console.print("[yellow]musicPlayer.xml still not found.[/yellow]")
            Prompt.ask("Press Enter to continue", default="")
            continue

        if choice == "6":
            clear_screen()
            console.print(Panel(
                "Bad audio policy controls what happens if ffmpeg fails on a file:\n\n"
                "strict   = stop immediately (best for debugging)\n"
                "skip     = skip that file and continue\n"
                "fallback = try loudness match, then basic conversion if needed (recommended)\n",
                title="Settings", border_style="cyan"
            ))
            pol = Prompt.ask("Choose policy", choices=["strict", "skip", "fallback"], default=bad_policy)
            set_bad_audio_policy(cfg, pol)
            bad_policy = pol
            console.print(f"[green]Saved bad audio policy:[/green] {bad_policy}")
            log.info(f"Set bad audio policy: {bad_policy}")
            Prompt.ask("Press Enter to continue", default="")
            continue

        if choice == "5":
            clear_screen()
            if Confirm.ask("Restore ALL files from showdown_backups into the game folder?", default=False):
                try:
                    restored = restore_all_from_backups(game_root, log)
                    console.print(f"[green]Restored {restored} file(s).[/green]")
                except Exception as e:
                    console.print(f"[red]{e}[/red]")
                    log.error(f"Restore failed: {e}")
            Prompt.ask("Press Enter to continue", default="")
            continue

        if choice == "4":
            cleanup_wavs(game_root, script_dir, log)
            Prompt.ask("Press Enter to continue", default="")
            continue

        if choice == "3":
            clear_screen()
            tp, lra = prompt_defaults_tp_lra()
            mode = Prompt.ask("Scan mode", choices=["fast", "accurate"], default="fast")
            accurate = (mode == "accurate")
            fast_streams = IntPrompt.ask("Fast mode: streams per DIC", default=DEFAULT_SCAN_FAST_STREAMS) if not accurate else DEFAULT_SCAN_FAST_STREAMS
            workers = IntPrompt.ask("Parallel workers", default=workers_default)

            console.print(Panel(
                "Full scan builds profiles for ALL DICs.\n\n"
                "Tip: If you've already injected custom music, profiles will measure your modded audio.\n"
                "If you want the ORIGINAL profile, restore from backups first.\n",
                title="Before Scan", border_style="yellow"
            ))

            if Confirm.ask("Restore from backups BEFORE scanning? (recommended)", default=False):
                try:
                    restored = restore_all_from_backups(game_root, log)
                    console.print(f"[green]Restored {restored} file(s) from backups.[/green]")
                except Exception as e:
                    console.print(f"[red]Restore failed:[/red] {e}")
                    log.error(f"Restore failed before scan: {e}")
                    if not Confirm.ask("Continue scan anyway?", default=False):
                        continue

            try:
                new_profiles = advanced_full_scan_build_profiles(
                    game_root=game_root,
                    vgmstream_exe=vgmstream_exe,
                    ffmpeg_exe=ffmpeg_exe,
                    tp=tp,
                    lra=lra,
                    fast_streams=fast_streams,
                    accurate=accurate,
                    workers=workers,
                    log=log
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled scan (Ctrl+C).[/yellow]")
                log.warn("User cancelled scan")
                Prompt.ask("Press Enter to continue", default="")
                continue
            except Exception as e:
                console.print(f"[red]Scan failed:[/red] {e}")
                log.error(f"Scan failed: {e}")
                Prompt.ask("Press Enter to continue", default="")
                continue

            profiles.update(new_profiles)
            save_profiles(profiles, PROFILES_JSON)
            export_profiles_as_python_dict(profiles, PROFILES_PY_TXT)

            console.print(f"[green]Saved profiles:[/green] {PROFILES_JSON.resolve()}")
            console.print(f"[green]Python export:[/green] {PROFILES_PY_TXT.resolve()}")
            Prompt.ask("Press Enter to continue", default="")
            continue

        if choice == "2":
            clear_screen()
            dic = choose_dic(dic_files)
            dic_key = profile_key_for_dic(dic, game_root)
            new_seed = reroll_mapping_seed(cfg, dic_key)
            console.print(f"[green]Re-rolled mapping seed for:[/green] {dic_key}\nNew seed: {new_seed}")
            log.info(f"Rerolled seed: {dic_key} => {new_seed}")
            Prompt.ask("Press Enter to continue", default="")
            continue

        # ============= Inject =============
        clear_screen()
        dic = choose_dic(dic_files)
        dic_key = profile_key_for_dic(dic, game_root)

        # Use cached DIC summary if possible (fast UX)
        cache = get_scan_cache(cfg)
        fp = dic_fingerprint(dic)
        ckey = dic_cache_key(dic)

        if ckey in cache and cache[ckey].get("fp") == fp:
            summ = DicScanSummary(
                stream_count=cache[ckey]["stream_count"],
                containers=cache[ckey]["containers"],
                wim_stream_indexes=cache[ckey]["wim_indexes"],
                wim_filenames=cache[ckey]["wim_files"]
            )
            console.print("[dim]Using cached DIC scan result (fast).[/dim]")
        else:
            clear_screen()
            console.print(Panel(
                "Reading the DIC pack...\n"
                "This can take a little time because vgmstream is being asked about each stream.\n"
                "You'll see a progress bar.\n",
                title="Working", border_style="cyan"
            ))
            summ = scan_dic_summary(vgmstream_exe, dic, show_progress=True)
            cache[ckey] = {
                "fp": fp,
                "stream_count": summ.stream_count,
                "containers": summ.containers,
                "wim_indexes": summ.wim_stream_indexes,
                "wim_files": summ.wim_filenames,
            }
            save_config(cfg)

        clear_screen()
        console.print(Panel(
            f"DIC: {dic_key}\n"
            f"Stream count: {summ.stream_count}\n"
            f"Containers: {summ.containers}\n"
            f"WIM streams: {len(summ.wim_stream_indexes)}\n",
            title="Pack Summary", border_style="cyan"
        ))

        if summ.containers.get("WIM", 0) <= 0:
            console.print("[yellow]This pack has no WIM streams, so we won't inject into it.[/yellow]")
            Prompt.ask("Press Enter to continue", default="")
            continue

        # Build WIM entries (show progress)
        entries_all = enumerate_wim_entries(vgmstream_exe, dic, summ, show_progress=True)
        clear_screen()
        show_wim_stream_preview(entries_all, limit=16)

        # Profile choice
        prof = profiles.get(dic_key)
        if prof:
            console.print(Panel(
                f"Profile found for this DIC:\n"
                f"Target LUFS (I): {prof.I:.3f}\n"
                f"TP: {prof.TP:.2f}  LRA: {prof.LRA:.1f}\n\n"
                "Using a profile helps your custom music match the game's volume.\n",
                title="Profile", border_style="cyan"
            ))
            if not Confirm.ask("Use this profile for loudness matching?", default=True):
                prof = None
        else:
            console.print(Panel(
                "No profile found for this DIC.\n\n"
                "You can still inject, but volume may be inconsistent.\n"
                "If you want perfect matching, run:\n"
                "  Full Scan & Build Profiles\n",
                title="No Profile", border_style="yellow"
            ))
            prof = None

        # Filter
        injection_filter_mode, allowed_wims = choose_injection_filter_mode(xml_filters)
        entries = filter_wim_entries_by_allowed_names(entries_all, allowed_wims)

        clear_screen()
        console.print(f"[green]Eligible WIM streams after filter:[/green] {len(entries)} / {len(entries_all)}")
        if not entries:
            console.print("[yellow]No streams matched that filter. Choose a different filter.[/yellow]")
            Prompt.ask("Press Enter to continue", default="")
            continue

        mapping_mode = pick_mapping_mode()
        seed = get_mapping_seed(cfg, dic_key)
        console.print(f"[dim]Mapping seed for this DIC:[/dim] {seed}")

        if Confirm.ask("Re-roll mapping seed now (fresh random assignment)?", default=False):
            seed = reroll_mapping_seed(cfg, dic_key)
            console.print(f"[green]New seed:[/green] {seed}")
            log.info(f"Rerolled seed before inject: {dic_key} => {seed}")

        music_folder = pick_music_folder(cfg)

        current_music_folder = music_folder
        dry = Confirm.ask("Dry-run (no files written; exports plan)?", default=False)

        clear_screen()
        console.print(Panel(
            f"DIC: {dic_key}\n"
            f"Streams to inject: {len(entries)}\n"
            f"Filter mode: {injection_filter_mode}\n"
            f"Loudness match: {'YES' if prof else 'NO'}\n"
            f"Bad policy: {bad_policy}\n"
            f"Mapping: {mapping_mode} (seed={seed})\n",
            title="Inject Summary", border_style="cyan"
        ))
        if not Confirm.ask("Proceed?", default=True):
            continue

        modified, plan = inject_wim_only(
            game_root=game_root,
            vgmstream_exe=vgmstream_exe,
            ffmpeg_exe=ffmpeg_exe,
            dic_path=dic,
            dic_key=dic_key,
            prof=prof,
            music_folder=music_folder,
            mapping_mode=mapping_mode,
            mapping_seed=seed,
            bad_audio_policy=bad_policy,
            dry_run=dry,
            injection_filter_mode=injection_filter_mode,
            allowed_wims=allowed_wims,
            log=log
        )

        json_plan, txt_plan = write_plan(plan)
        console.print(f"[green]Plan saved:[/green] {txt_plan.resolve()}")
        console.print(f"[green]Plan JSON:[/green] {json_plan.resolve()}")

        if dry:
            console.print(f"[yellow]Dry-run complete.[/yellow] Would modify {len(modified)} WIM file(s).")
            Prompt.ask("Press Enter to continue", default="")
            continue

        console.print(f"[green]Injection complete.[/green] Modified {len(modified)} WIM file(s).")

        if modified and Confirm.ask("Create a ZIP pack of the modified WIM files?", default=True):
            zip_path = make_zip_of_modified_wims(modified, OUTPUT_ROOT, game_root, dic_key=dic_key)
            console.print(f"[green]ZIP created:[/green] {zip_path.resolve()}")

        Prompt.ask("Press Enter to continue", default="")

    cleanup_temp_artifacts(Path(__file__).resolve().parent, log)
    log.info("Session end")
    console.print(Panel.fit("[bold]All done.[/bold]\nThanks for using Showdown Music Manager.", border_style="green"))
    pause("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        try:
            # Print to console if Rich is available
            console.print("\n[bold red]FATAL ERROR[/bold red]")
            console.print(traceback.format_exc())
        except Exception:
            # Fallback if console isn't available
            print("\nFATAL ERROR")
            print(traceback.format_exc())

        # Also write a crash log next to the script/EXE so you never lose it
        try:
            from pathlib import Path
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            crash_path = Path.cwd() / f"crash_{ts}.log"
            crash_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"\nCrash log written to: {crash_path}")
        except Exception:
            pass

        input("\nPress Enter to exit...")
        raise