import warnings

# Suppress torchcodec warning from pyannote.audio - we use FFmpeg for audio decoding
# and pass WAV files directly to PyAnnote, so torchcodec is not needed
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")

from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger

from app.settings import settings


def auth_middleware(app):
    @app.before_request
    def authenticate():
        if not settings.ENABLE_AUTH:
            return None

        excluded_paths = ["/health", f"{settings.API_PREFIX}/health"]
        if request.path in excluded_paths:
            return None

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return (
                jsonify({"success": False, "message": "Missing Authorization header"}),
                401,
            )

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = auth_header

        if token != settings.APP_TOKEN:
            return jsonify({"success": False, "message": "Invalid token"}), 401

        return None


def create_app():
    app = Flask(__name__)

    CORS(app)

    if settings.ENABLE_AUTH:
        if not settings.APP_TOKEN:
            raise RuntimeError(
                "APP_TOKEN must be set when authentication is enabled. "
                "Set APP_TOKEN environment variable or disable auth with ENABLE_AUTH=false."
            )
        logger.info("Enabling authentication")
        auth_middleware(app)

    @app.route(f"{settings.API_PREFIX}", methods=["GET"])
    def api_info():
        from app import __version__

        return jsonify(
            {
                "success": True,
                "message": "FFmpeg Microservice",
                "version": __version__,
                "endpoints": {
                    "health": f"{settings.API_PREFIX}/health",
                    "extract": f"{settings.API_PREFIX}/extract",
                    "transcribe": f"{settings.API_PREFIX}/transcribe",
                    "transcribe_segments": f"{settings.API_PREFIX}/transcribe/segments",
                    "diarize": f"{settings.API_PREFIX}/diarize",
                },
            }
        )

    from app.api.endpoints import api_blueprint

    app.register_blueprint(api_blueprint, url_prefix=settings.API_PREFIX)

    return app
