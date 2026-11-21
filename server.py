
import os
import uuid
import json
import base64
import shutil
import asyncio
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CART3R_API_KEY = os.getenv("CART3R_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/cart3r-data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_FRAMES = max(1, min(int(os.getenv("MAX_FRAMES", "8")), 12))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "180"))

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1-mini")
TRANSCRIPTION_MODEL = os.getenv(
    "TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")

if not CART3R_API_KEY:
    raise RuntimeError("CART3R_API_KEY is missing.")

DATA_DIR.mkdir(parents=True, exist_ok=True)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
mcp = FastMCP("CART3R TikTok Vision")


# --------------------------------------------------
# Authentication middleware
# --------------------------------------------------

class BearerAuthMiddleware:
    """Protect every endpoint except the public health check."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        if path == "/health":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode(
            "utf-8", errors="ignore"
        )

        expected = f"Bearer {CART3R_API_KEY}"

        if auth != expected:
            response = JSONResponse(
                {"error": "Unauthorized"},
                status_code=401
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def validate_tiktok_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()

        allowed = (
            host == "tiktok.com"
            or host.endswith(".tiktok.com")
        )

        return (
            parsed.scheme == "https"
            and allowed
            and parsed.username is None
            and parsed.password is None
        )
    except Exception:
        return False


def run_command(args: list[str], timeout: int = 60):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-2000:] or "External command failed."
        )

    return result.stdout


def get_video_duration(video_path: Path) -> float:
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ])

    duration = float(output.strip())

    if duration <= 0:
        raise RuntimeError("Could not determine video duration.")

    if duration > MAX_VIDEO_SECONDS:
        raise RuntimeError(
            f"Video exceeds the {MAX_VIDEO_SECONDS}-second limit."
        )

    return duration


def extract_frames(
    video_path: Path,
    output_dir: Path,
    duration: float
) -> list[Path]:

    frames = []

    count = min(MAX_FRAMES, max(1, int(duration) + 1))

    for index in range(count):
        timestamp = (
            duration / 2
            if count == 1
            else duration * index / count
        )

        output = output_dir / f"frame_{index:03d}.jpg"

        run_command([
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", "scale=768:-1",
            "-q:v", "5",
            "-y",
            str(output)
        ])

        if output.exists() and output.stat().st_size > 0:
            frames.append(output)

    if not frames:
        raise RuntimeError("FFmpeg could not extract video frames.")

    return frames


def extract_audio(video_path: Path, output_path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "64k",
                "-y",
                str(output_path)
            ],
            capture_output=True,
            timeout=90,
            check=False
        )

        return (
            output_path.exists()
            and output_path.stat().st_size > 1000
        )
    except Exception:
        return False


# --------------------------------------------------
# MCP tool: TikTok metadata
# --------------------------------------------------

@mcp.tool()
async def resolve_tiktok(url: str) -> dict:
    """
    Retrieve public TikTok oEmbed metadata.

    This does not download the video.
    """

    if not validate_tiktok_url(url):
        return {"error": "Please provide a valid HTTPS TikTok URL."}

    endpoint = "https://www.tiktok.com/oembed"

    try:
        async with httpx.AsyncClient(timeout=20) as http:
            response = await http.get(
                endpoint,
                params={"url": url},
                follow_redirects=False
            )

        if response.status_code != 200:
            return {
                "error": "TikTok oEmbed did not return metadata.",
                "status_code": response.status_code,
                "note": "The video may be unavailable or unsupported."
            }

        data = response.json()

        return {
            "success": True,
            "title": data.get("title"),
            "author_name": data.get("author_name"),
            "author_url": data.get("author_url"),
            "thumbnail_url": data.get("thumbnail_url"),
            "embed_url": data.get("html"),
            "video_downloaded": False,
            "next_step": (
                "Upload an authorized video file using the "
                "upload instructions tool to analyze its contents."
            )
        }

    except Exception as exc:
        return {"error": f"Metadata lookup failed: {str(exc)}"}


# --------------------------------------------------
# MCP tool: upload instructions
# --------------------------------------------------

@mcp.tool()
async def get_upload_instructions() -> dict:
    """Show how to upload a video to this server."""

    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL is not configured."}

    return {
        "upload_url": f"{PUBLIC_BASE_URL}/upload",
        "method": "POST",
        "authorization": "Bearer YOUR_CART3R_API_KEY",
        "multipart_field": "file",
        "example": (
            f'curl -X POST "{PUBLIC_BASE_URL}/upload" '
            '-H "Authorization: Bearer YOUR_CART3R_API_KEY" '
            '-F "file=@video.mp4"'
        ),
        "max_upload_mb": MAX_UPLOAD_MB,
        "note": (
            "Replace YOUR_CART3R_API_KEY with your secret. "
            "Do not share that secret publicly."
        )
    }


# --------------------------------------------------
# MCP tool: analyze uploaded video
# --------------------------------------------------

@mcp.tool()
async def analyze_uploaded_video(upload_id: str) -> dict:
    """
    Analyze a video previously uploaded through /upload.
    """

    try:
        upload_uuid = uuid.UUID(upload_id)
    except ValueError:
        return {"error": "Invalid upload ID."}

    video_path = DATA_DIR / f"{upload_uuid}.video"

    if not video_path.is_file():
        return {
            "error": "Upload not found. Upload the video again."
        }

    try:
        with tempfile.TemporaryDirectory(
            dir=str(DATA_DIR)
        ) as temporary:

            work_dir = Path(temporary)
            duration = await asyncio.to_thread(
                get_video_duration, video_path
            )

            frames = await asyncio.to_thread(
                extract_frames,
                video_path,
                work_dir,
                duration
            )

            content = [
                {
                    "type": "input_text",
                    "text": (
                        "Analyze this video carefully. Describe what "
                        "happens, the visible people or objects without "
                        "identifying real people, actions, setting, "
                        "on-screen text, editing, and notable details. "
                        "Clearly distinguish observations from guesses. "
                        f"The video is {duration:.1f} seconds long."
                    )
                }
            ]

            for frame in frames:
                encoded = base64.b64encode(
                    frame.read_bytes()
                ).decode("ascii")

                content.append({
                    "type": "input_image",
                    "image_url": (
                        f"data:image/jpeg;base64,{encoded}"
                    )
                })

            vision_response = await client.responses.create(
                model=VISION_MODEL,
                input=[{
                    "role": "user",
                    "content": content
                }]
            )

            transcript = None
            audio_path = work_dir / "audio.mp3"

            has_audio = await asyncio.to_thread(
                extract_audio, video_path, audio_path
            )

            if has_audio:
                try:
                    with audio_path.open("rb") as audio_file:
                        transcription = (
                            await client.audio.transcriptions.create(
                                model=TRANSCRIPTION_MODEL,
                                file=audio_file
                            )
                        )

                    transcript = transcription.text

                except Exception as exc:
                    transcript = (
                        f"Transcription unavailable: {str(exc)}"
                    )

            return {
                "success": True,
                "duration_seconds": round(duration, 2),
                "frames_analyzed": len(frames),
                "visual_analysis": vision_response.output_text,
                "audio_transcript": transcript
            }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)
        }


# --------------------------------------------------
# HTTP routes
# --------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request):
    return JSONResponse({
        "status": "ok",
        "service": "CART3R TikTok Vision MCP"
    })


@mcp.custom_route("/upload", methods=["POST"])
async def upload_video(request: Request):
    try:
        form = await request.form()
        uploaded = form.get("file")

        if uploaded is None or not hasattr(uploaded, "read"):
            return JSONResponse(
                {"error": "Missing multipart file field named 'file'."},
                status_code=400
            )

        filename = Path(uploaded.filename or "video.mp4")
        extension = filename.suffix.lower()

        allowed_extensions = {
            ".mp4", ".mov", ".m4v", ".webm", ".mkv"
        }

        if extension not in allowed_extensions:
            return JSONResponse(
                {"error": "Unsupported video file extension."},
                status_code=400
            )

        upload_id = str(uuid.uuid4())
        destination = DATA_DIR / f"{upload_id}.video"

        total = 0
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024

        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await uploaded.read(1024 * 1024)

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > max_bytes:
                        output.close()
                        destination.unlink(missing_ok=True)

                        return JSONResponse(
                            {"error": "Upload exceeds size limit."},
                            status_code=413
                        )

                    output.write(chunk)

        finally:
            await uploaded.close()

        if total == 0:
            destination.unlink(missing_ok=True)
            return JSONResponse(
                {"error": "Uploaded file is empty."},
                status_code=400
            )

        return JSONResponse({
            "success": True,
            "upload_id": upload_id,
            "size_bytes": total,
            "next_step": (
                "Call the analyze_uploaded_video MCP tool "
                "with this upload_id."
            )
        })

    except Exception as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=500
        )


# --------------------------------------------------
# ASGI application
# --------------------------------------------------

mcp_app = mcp.streamable_http_app()

# The middleware protects /mcp and /upload.
# /health is intentionally public.
app = BearerAuthMiddleware(mcp_app)
