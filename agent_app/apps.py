import os
from django.apps import AppConfig


class AgentAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agent_app"

    def ready(self):
        media_dirs = [
            "media",
            "media/generated_images",
            "media/generated_videos",
            "media/blender_renders",
        ]
        for dir_path in media_dirs:
            os.makedirs(dir_path, exist_ok=True)
