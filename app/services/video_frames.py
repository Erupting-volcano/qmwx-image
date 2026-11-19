"""Turn one video sweep into evenly spaced still frames; no external binaries needed."""

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import BinaryIO

import av
from PIL import Image


DEFAULT_FPS = 1.5
MAX_FRAME_SIDE = 2048
JPEG_QUALITY = 95


class VideoFrameError(ValueError):
    """A safe explanation for a video that cannot be turned into frames."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class ExtractedFrame:
    filename: str
    stream: BytesIO
    timestamp: float


def _jpeg_bytes(image: Image.Image) -> BytesIO:
    if max(image.size) > MAX_FRAME_SIDE:
        image = image.copy()
        image.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE), Image.LANCZOS)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buffer.seek(0)
    return buffer


def extract_frames(source: Path | BinaryIO, *, max_frames: int, max_seconds: float,
                   max_pixels: int, fps: float = DEFAULT_FPS) -> list[ExtractedFrame]:
    """Decode once and keep the frame closest to each evenly spaced target moment."""
    if max_frames < 1:
        raise ValueError("抽帧数量上限必须为正整数")
    try:
        container = av.open(source)
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoFrameError("invalid_video", "无法读取该视频文件，请确认格式受支持。") from exc

    try:
        if not container.streams.video:
            raise VideoFrameError("video_stream_missing", "文件中没有视频画面，无法抽帧。")
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if duration <= 0:
            raise VideoFrameError("video_duration_unknown", "无法确定视频时长，请换一个文件或格式。")
        if duration > max_seconds:
            raise VideoFrameError(
                "video_too_long",
                f"视频时长 {duration:.1f} 秒，超过 {max_seconds:.0f} 秒上限；请分段拍摄或先裁剪。",
            )
        count = max(2, min(max_frames, math.ceil(duration * fps)))
        targets = [(index + 0.5) * duration / count for index in range(count)]
        if stream.codec_context.width * stream.codec_context.height > max_pixels:
            raise VideoFrameError("video_resolution_too_large", "视频分辨率超过安全上限，请降低分辨率后重试。")

        frames: list[ExtractedFrame] = []
        next_target = 0
        for frame in container.decode(stream):
            timestamp = float(frame.time or 0.0)
            if timestamp + 1e-6 < targets[next_target]:
                continue
            image = frame.to_image()
            if image.width * image.height > max_pixels:
                raise VideoFrameError("video_resolution_too_large", "视频帧分辨率超过安全上限。")
            seconds = round(timestamp, 1)
            frames.append(ExtractedFrame(
                filename=f"frame-{len(frames):03d}-t{seconds:05.1f}s.jpg",
                stream=_jpeg_bytes(image),
                timestamp=timestamp,
            ))
            next_target += 1
            if next_target >= len(targets):
                break
        if not frames:
            raise VideoFrameError("video_empty", "视频中没有可解码的画面。")
        return frames
    except av.FFmpegError as exc:
        raise VideoFrameError("video_decode_failed", "视频解码失败，文件可能损坏。") from exc
    finally:
        container.close()
