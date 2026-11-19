"""Shared helper: build a tiny H.264 video in memory for frame-extraction tests."""

from io import BytesIO

import av
from PIL import Image


def make_video(seconds=2.0, fps=10, size=(160, 120), total_frames=None) -> BytesIO:
    buffer = BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
    count = total_frames if total_frames is not None else int(seconds * fps)
    for index in range(count):
        image = Image.new("RGB", size, (index * 7 % 256, 80, 160))
        for packet in stream.encode(av.VideoFrame.from_image(image)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    buffer.seek(0)
    return buffer
