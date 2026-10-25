from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


def image_bytes(image_format="PNG", size=(32, 24), color=(30, 110, 170)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def png_bytes():
    return image_bytes()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def client_factory(tmp_path):
    clients = []

    def factory(*, processors=None, **overrides):
        config = Settings(data_dir=tmp_path / f"data-{len(clients)}", **overrides)
        test_client = TestClient(create_app(config, processors=processors))
        test_client.__enter__()
        clients.append(test_client)
        return test_client, config

    yield factory
    for test_client in reversed(clients):
        test_client.__exit__(None, None, None)
