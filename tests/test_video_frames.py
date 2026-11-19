"""Video sweeps become evenly spaced frames, then follow the normal image path."""

from io import BytesIO

import pytest
from PIL import Image

from app.services.video_frames import VideoFrameError, extract_frames
from video_helpers import make_video


def test_frames_are_evenly_spaced_jpegs():
    frames = extract_frames(make_video(4.0), max_frames=6, max_seconds=60, max_pixels=40_000_000)
    assert 2 <= len(frames) <= 6
    timestamps = [frame.timestamp for frame in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[-1] > timestamps[0]
    for frame in frames:
        assert frame.filename.startswith("frame-") and frame.filename.endswith(".jpg")
        image = Image.open(frame.stream)
        assert image.format == "JPEG"
        assert image.size == (160, 120)


def test_frame_count_follows_the_requested_rate():
    frames = extract_frames(make_video(2.0), max_frames=20, max_seconds=60, max_pixels=40_000_000, fps=1.5)
    assert len(frames) == 3


def test_frame_count_is_capped_by_max_frames():
    frames = extract_frames(make_video(6.0), max_frames=2, max_seconds=60, max_pixels=40_000_000)
    assert len(frames) == 2


def test_long_video_is_rejected():
    with pytest.raises(VideoFrameError) as info:
        extract_frames(make_video(3.0), max_frames=5, max_seconds=1, max_pixels=1_000_000)
    assert info.value.code == "video_too_long"


def test_non_video_input_is_rejected():
    with pytest.raises(VideoFrameError) as info:
        extract_frames(BytesIO(b"definitely not a video"), max_frames=5, max_seconds=10,
                       max_pixels=1_000_000)
    assert info.value.code == "invalid_video"


def test_tall_frames_are_downscaled_to_a_safe_side():
    frames = extract_frames(make_video(1.0, size=(2400, 1200)), max_frames=2, max_seconds=60,
                            max_pixels=40_000_000)
    image = Image.open(frames[0].stream)
    assert max(image.size) <= 2048
