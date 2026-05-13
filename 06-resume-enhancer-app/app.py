"""Entry point for Hugging Face Spaces deployment."""
from ui.gradio_app import build_app
import os

app = build_app()

app.queue(default_concurrency_limit=2).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    show_error=True,
)
