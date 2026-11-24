"""ctypes wrapper for speech.cpp C API (speech.dll / libspeech).

Uses speech_dialogue_start / speech_dialogue_push / speech_dialogue_end to run
two simultaneous Haru-4B conversation sessions on a single loaded model instance
without duplicating VRAM usage.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# speech_status enum values
SPEECH_OK = 0
SPEECH_CANCELLED = 1

# speech_device_kind enum values
SPEECH_DEVICE_CPU = 0
SPEECH_DEVICE_GPU = 1
SPEECH_DEVICE_IGPU = 2

# Callback signature:
# int on_frame(const float * samples, size_t n_samples, const char * text, void * user_data)
SPEECH_DIALOGUE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_void_p,
)


class SpeechError(RuntimeError):
    """Raised when a speech.cpp C API call returns a negative status."""

    def __init__(self, status: int, status_name: str, option: Optional[str], message: str):
        self.status = status
        self.status_name = status_name
        self.option = option
        self.message = message
        opt_str = f" (option={option})" if option else ""
        super().__init__(f"speech.cpp error [{status_name}{opt_str}]: {message}")


@dataclass
class DeviceInfo:
    index: int
    name: str
    description: str
    kind: str  # "cpu", "gpu", "igpu"
    total_bytes: int
    free_bytes: int


def find_speech_exe() -> Optional[Path]:
    """Locate speech.exe from PATH or default Windows install location."""
    found = shutil.which("speech")
    if found:
        return Path(found)
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        candidate = Path(local_app) / "Programs" / "speech.cpp" / "current" / "speech.exe"
        if candidate.exists():
            return candidate
    return None


def find_speech_dll(explicit_path: str = "") -> Path:
    """Locate speech.dll on Windows."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Configured speech_dll not found: {p}")

    exe = find_speech_exe()
    if exe:
        dll = exe.parent / "speech.dll"
        if dll.exists():
            return dll

    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        candidate = Path(local_app) / "Programs" / "speech.cpp" / "current" / "speech.dll"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "speech.dll not found. Install speech.cpp or set model.speech_dll in config.yaml."
    )


def resolve_model_path(model_spec: str) -> Path:
    """Resolve a catalog model name (e.g. 'haru-4b:q8_0') or file path to a local GGUF path.

    Calls `speech pull <model_spec>` which prints the cached path (and downloads if missing).
    """
    p = Path(model_spec)
    if p.exists() and p.is_file():
        return p.resolve()

    exe = find_speech_exe()
    if not exe:
        raise FileNotFoundError("speech.exe not found in PATH or default install directory.")

    proc = subprocess.run(
        [str(exe), "pull", model_spec],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`speech pull {model_spec}` failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"`speech pull {model_spec}` returned no path.")
    resolved = Path(lines[-1])
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved model path does not exist: {resolved}")
    return resolved


class SpeechLibrary:
    """Thin, safe ctypes wrapper around speech.dll."""

    def __init__(self, dll_path: str = ""):
        self.dll_path = find_speech_dll(dll_path)
        dll_dir = str(self.dll_path.parent)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
        self._lib = ctypes.CDLL(str(self.dll_path))
        self._bind_functions()
        self._option_cache: dict[str, int] = {}

    def _bind_functions(self) -> None:
        lib = self._lib

        # Version & errors
        lib.speech_version.argtypes = []
        lib.speech_version.restype = ctypes.c_char_p
        lib.speech_api_version_major.argtypes = []
        lib.speech_api_version_major.restype = ctypes.c_int
        lib.speech_api_version_minor.argtypes = []
        lib.speech_api_version_minor.restype = ctypes.c_int

        lib.speech_status_name.argtypes = [ctypes.c_int]
        lib.speech_status_name.restype = ctypes.c_char_p
        lib.speech_last_error.argtypes = []
        lib.speech_last_error.restype = ctypes.c_char_p
        lib.speech_last_error_option.argtypes = []
        lib.speech_last_error_option.restype = ctypes.c_char_p

        # Devices
        lib.speech_device_count.argtypes = []
        lib.speech_device_count.restype = ctypes.c_size_t
        lib.speech_device_name.argtypes = [ctypes.c_size_t]
        lib.speech_device_name.restype = ctypes.c_char_p
        lib.speech_device_description.argtypes = [ctypes.c_size_t]
        lib.speech_device_description.restype = ctypes.c_char_p
        lib.speech_device_get_kind.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        lib.speech_device_get_kind.restype = ctypes.c_int
        lib.speech_device_memory.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.speech_device_memory.restype = ctypes.c_int

        # Options
        lib.speech_option_from_name.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.speech_option_from_name.restype = ctypes.c_int

        # Load params
        lib.speech_load_params_new.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.speech_load_params_new.restype = ctypes.c_int
        lib.speech_load_params_free.argtypes = [ctypes.c_void_p]
        lib.speech_load_params_free.restype = None
        lib.speech_load_params_set_device.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.speech_load_params_set_device.restype = ctypes.c_int
        lib.speech_load_params_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.speech_load_params_set_threads.restype = ctypes.c_int
        lib.speech_load_params_set_warmup.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.speech_load_params_set_warmup.restype = ctypes.c_int

        # Model
        lib.speech_model_load.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.speech_model_load.restype = ctypes.c_int
        lib.speech_model_free.argtypes = [ctypes.c_void_p]
        lib.speech_model_free.restype = None

        lib.speech_model_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.speech_model_get_info.restype = ctypes.c_int
        lib.speech_model_info_free.argtypes = [ctypes.c_void_p]
        lib.speech_model_info_free.restype = None
        lib.speech_model_info_sample_rate.argtypes = [ctypes.c_void_p]
        lib.speech_model_info_sample_rate.restype = ctypes.c_int
        lib.speech_model_info_frame_samples.argtypes = [ctypes.c_void_p]
        lib.speech_model_info_frame_samples.restype = ctypes.c_int
        lib.speech_model_info_device.argtypes = [ctypes.c_void_p]
        lib.speech_model_info_device.restype = ctypes.c_char_p
        lib.speech_model_info_json.argtypes = [ctypes.c_void_p]
        lib.speech_model_info_json.restype = ctypes.c_char_p

        # Voices
        lib.speech_voice_add.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.speech_voice_add.restype = ctypes.c_int
        lib.speech_voice_make.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        lib.speech_voice_make.restype = ctypes.c_int

        # Request
        lib.speech_request_new.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.speech_request_new.restype = ctypes.c_int
        lib.speech_request_free.argtypes = [ctypes.c_void_p]
        lib.speech_request_free.restype = None
        lib.speech_request_set_string.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
        lib.speech_request_set_string.restype = ctypes.c_int
        lib.speech_request_set_int.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
        lib.speech_request_set_int.restype = ctypes.c_int
        lib.speech_request_set_float.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        lib.speech_request_set_float.restype = ctypes.c_int
        lib.speech_request_set_bool.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.speech_request_set_bool.restype = ctypes.c_int

        # Dialogue
        lib.speech_dialogue_start.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            SPEECH_DIALOGUE_CALLBACK,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.speech_dialogue_start.restype = ctypes.c_int
        lib.speech_dialogue_free.argtypes = [ctypes.c_void_p]
        lib.speech_dialogue_free.restype = None
        lib.speech_dialogue_push.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.speech_dialogue_push.restype = ctypes.c_int
        lib.speech_dialogue_end.argtypes = [ctypes.c_void_p]
        lib.speech_dialogue_end.restype = ctypes.c_int

    def _check(self, status: int) -> int:
        if status < 0:
            sname_b = self._lib.speech_status_name(status)
            sname = sname_b.decode("utf-8", errors="replace") if sname_b else str(status)
            err_b = self._lib.speech_last_error()
            err = err_b.decode("utf-8", errors="replace") if err_b else "unknown error"
            opt_b = self._lib.speech_last_error_option()
            opt = opt_b.decode("utf-8", errors="replace") if opt_b else None
            raise SpeechError(status, sname, opt, err)
        return status

    def version(self) -> str:
        v = self._lib.speech_version()
        return v.decode("utf-8") if v else "unknown"

    def api_version(self) -> tuple[int, int]:
        return (
            int(self._lib.speech_api_version_major()),
            int(self._lib.speech_api_version_minor()),
        )

    def list_devices(self) -> list[DeviceInfo]:
        count = int(self._lib.speech_device_count())
        kind_map = {
            SPEECH_DEVICE_CPU: "cpu",
            SPEECH_DEVICE_GPU: "gpu",
            SPEECH_DEVICE_IGPU: "igpu",
        }
        devices: list[DeviceInfo] = []
        for i in range(count):
            name = (self._lib.speech_device_name(i) or b"").decode("utf-8", errors="replace")
            desc = (self._lib.speech_device_description(i) or b"").decode("utf-8", errors="replace")
            kind_val = ctypes.c_int(0)
            self._check(self._lib.speech_device_get_kind(i, ctypes.byref(kind_val)))
            total = ctypes.c_uint64(0)
            free = ctypes.c_uint64(0)
            self._check(self._lib.speech_device_memory(i, ctypes.byref(total), ctypes.byref(free)))
            devices.append(
                DeviceInfo(
                    index=i,
                    name=name,
                    description=desc,
                    kind=kind_map.get(kind_val.value, "unknown"),
                    total_bytes=int(total.value),
                    free_bytes=int(free.value),
                )
            )
        return devices

    def option_id(self, name: str) -> int:
        if name not in self._option_cache:
            opt = ctypes.c_int(0)
            self._check(self._lib.speech_option_from_name(name.encode("utf-8"), ctypes.byref(opt)))
            self._option_cache[name] = int(opt.value)
        return self._option_cache[name]

    def load_model(
        self,
        model_path: str | Path,
        device: str = "auto",
        threads: int = 0,
        warmup: bool = True,
    ) -> "SpeechModel":
        params = ctypes.c_void_p()
        self._check(self._lib.speech_load_params_new(ctypes.byref(params)))
        try:
            if device:
                self._check(
                    self._lib.speech_load_params_set_device(params, device.encode("utf-8"))
                )
            if threads and threads > 0:
                self._check(self._lib.speech_load_params_set_threads(params, int(threads)))
            self._check(self._lib.speech_load_params_set_warmup(params, 1 if warmup else 0))

            model_ptr = ctypes.c_void_p()
            self._check(
                self._lib.speech_model_load(
                    str(model_path).encode("utf-8"),
                    params,
                    ctypes.byref(model_ptr),
                )
            )
            return SpeechModel(self, model_ptr)
        finally:
            self._lib.speech_load_params_free(params)


class SpeechModel:
    """Managed speech_model handle. Always freed cleanly before exit."""

    def __init__(self, library: SpeechLibrary, ptr: ctypes.c_void_p):
        self._lib = library
        self._ptr = ptr
        self._closed = False
        self._sample_rate = 24000
        self._frame_samples = 3200
        self._device = "unknown"
        self._read_info()
        atexit.register(self.close)

    def _read_info(self) -> None:
        info = ctypes.c_void_p()
        self._lib._check(self._lib._lib.speech_model_get_info(self._ptr, ctypes.byref(info)))
        try:
            self._sample_rate = int(self._lib._lib.speech_model_info_sample_rate(info))
            self._frame_samples = int(self._lib._lib.speech_model_info_frame_samples(info))
            dev_b = self._lib._lib.speech_model_info_device(info)
            self._device = dev_b.decode("utf-8", errors="replace") if dev_b else "unknown"
        finally:
            self._lib._lib.speech_model_info_free(info)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def device(self) -> str:
        return self._device

    def add_voice(self, name: str, path: str | Path) -> None:
        self._lib._check(
            self._lib._lib.speech_voice_add(
                self._ptr,
                name.encode("utf-8"),
                str(path).encode("utf-8"),
            )
        )

    def start_dialogue(
        self,
        options: dict[str, object],
        input_sample_rate: Optional[int] = None,
    ) -> "SpeechDialogueSession":
        sr = input_sample_rate or self._sample_rate
        req = ctypes.c_void_p()
        self._lib._check(self._lib._lib.speech_request_new(self._ptr, ctypes.byref(req)))
        try:
            for key, val in options.items():
                if val is None:
                    continue
                opt = self._lib.option_id(key)
                if isinstance(val, bool):
                    self._lib._check(
                        self._lib._lib.speech_request_set_bool(req, opt, 1 if val else 0)
                    )
                elif isinstance(val, int):
                    self._lib._check(self._lib._lib.speech_request_set_int(req, opt, int(val)))
                elif isinstance(val, float):
                    self._lib._check(self._lib._lib.speech_request_set_float(req, opt, float(val)))
                elif isinstance(val, str):
                    self._lib._check(
                        self._lib._lib.speech_request_set_string(req, opt, val.encode("utf-8"))
                    )
                else:
                    raise TypeError(f"Unsupported option type for {key}: {type(val)}")

            return SpeechDialogueSession(self._lib, req, sr)
        finally:
            # Per speech.h: speech_dialogue_start copies the options, so request can be freed.
            self._lib._lib.speech_request_free(req)

    def close(self) -> None:
        if not self._closed and self._ptr:
            self._closed = True
            self._lib._lib.speech_model_free(self._ptr)
            self._ptr = ctypes.c_void_p()


class SpeechDialogueSession:
    """One active Haru-4B conversation session (holds its own KV cache)."""

    def __init__(self, library: SpeechLibrary, request_ptr: ctypes.c_void_p, sample_rate: int):
        self._lib = library
        self._ptr = ctypes.c_void_p()
        self._closed = False
        self._ended = False
        self._pending_frames: list[tuple[np.ndarray, str]] = []

        # Keep callback reference alive for the lifetime of this session!
        def _on_frame(
            samples_ptr: ctypes.POINTER(ctypes.c_float),
            n_samples: int,
            text_ptr: bytes,
            _user_data: ctypes.c_void_p,
        ) -> int:
            arr = np.ctypeslib.as_array(samples_ptr, shape=(n_samples,)).copy()
            text = text_ptr.decode("utf-8", errors="replace") if text_ptr else ""
            self._pending_frames.append((arr, text))
            return 0

        self._callback = SPEECH_DIALOGUE_CALLBACK(_on_frame)
        self._lib._check(
            self._lib._lib.speech_dialogue_start(
                request_ptr,
                int(sample_rate),
                self._callback,
                None,
                ctypes.byref(self._ptr),
            )
        )

    def pop_frames(self) -> list[tuple[np.ndarray, str]]:
        frames = self._pending_frames
        self._pending_frames = []
        return frames

    def push_frame(self, samples: np.ndarray) -> list[tuple[np.ndarray, str]]:
        if self._closed or self._ended:
            raise RuntimeError("Dialogue session is already ended/closed.")
        arr = np.ascontiguousarray(samples, dtype=np.float32)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib._check(self._lib._lib.speech_dialogue_push(self._ptr, ptr, arr.size))
        return self.pop_frames()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ptr:
            if not self._ended:
                self._ended = True
                try:
                    self._lib._lib.speech_dialogue_end(self._ptr)
                except Exception:
                    pass
            self._lib._lib.speech_dialogue_free(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> "SpeechDialogueSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

