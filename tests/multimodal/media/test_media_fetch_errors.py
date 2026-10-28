# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A media URL that cannot be fetched is a client error (400), not a 500."""

import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from PIL import Image

from vllm.connections import HTTPConnection
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO

# Presigned object-storage URLs carry credentials in the query string.
SECRET_QUERY = "?X-Amz-Credential=AKIDEXAMPLE&X-Amz-Signature=deadbeefsecret"


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    hits: dict[str, int] = {}

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        _Handler.hits[path] = _Handler.hits.get(path, 0) + 1
        if path == "/ok.png":
            body = _png_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/slow.png":
            time.sleep(3)
            self.send_response(200)
            self.end_headers()
        elif path == "/garbage.png":
            body = b"not an image"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/custom-reason.png":
            # A third-party host controls its reason phrase.
            self.send_response(
                403, "Denied see https://leak.example/?X-Amz-Signature=x"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:  # /403.png -> 403, /503.png -> 503, ...
            self.send_response(int(path.strip("/").split(".")[0]))
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _one_attempt(monkeypatch):
    monkeypatch.setenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "1")


async def _load(url: str, use_async: bool):
    # A fresh connection per call: the global one keeps an aiohttp session
    # bound to the event loop it was created on.
    connector = MediaConnector(connection=HTTPConnection(reuse_client=False))
    if use_async:
        return await connector.load_from_url_async(url, ImageMediaIO(), fetch_timeout=1)
    return connector.load_from_url(url, ImageMediaIO(), fetch_timeout=1)


def _assert_client_error(exc: BaseException, reason: str) -> None:
    assert isinstance(exc, VLLMValidationError)
    message = str(exc)
    assert message.startswith("Failed to fetch media from "), message
    assert reason in message, message
    # The presigned query string must never be echoed back or logged.
    assert "X-Amz" not in message and "secret" not in message, message
    err = create_error_response(exc)
    assert err.error.code == HTTPStatus.BAD_REQUEST
    assert err.error.type == "BadRequestError"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (403, "HTTP 403 Forbidden"),
        (404, "HTTP 404 Not Found"),
        (429, "HTTP 429 Too Many Requests"),
        (500, "HTTP 500 Internal Server Error"),
        (503, "HTTP 503 Service Unavailable"),
    ],
)
async def test_http_error_status_is_bad_request(server, use_async, status, reason):
    with pytest.raises(VLLMValidationError) as exc_info:
        await _load(f"{server}/{status}.png{SECRET_QUERY}", use_async)
    _assert_client_error(exc_info.value, reason)
    assert "127.0.0.1" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_timeout_is_bad_request(server, use_async):
    with pytest.raises(VLLMValidationError) as exc_info:
        await _load(f"{server}/slow.png{SECRET_QUERY}", use_async)
    _assert_client_error(exc_info.value, "timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_unresolvable_host_is_bad_request(use_async):
    with pytest.raises(VLLMValidationError) as exc_info:
        await _load(
            f"http://media.does-not-exist.invalid/a.png{SECRET_QUERY}", use_async
        )
    _assert_client_error(exc_info.value, "could not connect")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_5xx_is_still_retried_before_mapping(server, monkeypatch, use_async):
    monkeypatch.setenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "2")
    path = "/502.png"
    before = _Handler.hits.get(path, 0)
    with pytest.raises(VLLMValidationError) as exc_info:
        await _load(f"{server}{path}", use_async)
    _assert_client_error(exc_info.value, "HTTP 502 Bad Gateway")
    assert _Handler.hits[path] - before == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_successful_fetch_unchanged(server, use_async):
    image = await _load(f"{server}/ok.png", use_async)
    assert image.size == (4, 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_decode_errors_are_not_wrapped(server, use_async):
    # Only the download is mapped; a fetched body that is not an image keeps
    # its own error.
    with pytest.raises(Exception) as exc_info:
        await _load(f"{server}/garbage.png", use_async)
    assert not str(exc_info.value).startswith("Failed to fetch media")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_host_reason_phrase_is_not_echoed(server, use_async):
    with pytest.raises(VLLMValidationError) as exc_info:
        await _load(f"{server}/custom-reason.png", use_async)
    _assert_client_error(exc_info.value, "HTTP 403 Forbidden")
    assert "leak.example" not in str(exc_info.value)
