import os
import urllib.request
import urllib.parse
import json
import uuid
from ..core import BaseTool


class ImageGenTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="image_gen",
            description="Generate images from text descriptions. Usage: image_gen: a beautiful sunset",
        )
        self.save_dir = "media/generated_images"
        self.api_mode = os.environ.get("IMAGE_GEN_MODE", "pollinations")
        self.base_url = os.environ.get("SITE_URL", "http://localhost:8000")

    def execute(self, input_data: str = "") -> str:
        prompt = self._clean(input_data)
        if not prompt:
            return json.dumps({"error": "Provide a description", "type": "text"})

        os.makedirs(self.save_dir, exist_ok=True)

        if self.api_mode == "pollinations":
            return self._generate_pollinations(prompt)
        else:
            return json.dumps({
                "type": "text",
                "content": f"Image generation not configured. Set IMAGE_GEN_MODE=pollinations in .env"
            })

    def _clean(self, query: str) -> str:
        if query.lower().startswith("image_gen:"):
            query = query[len("image_gen:"):]
        return query.strip()

    def _generate_pollinations(self, prompt: str) -> str:
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true"

            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(self.save_dir, filename)

            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                image_data = resp.read()

            with open(filepath, "wb") as f:
                f.write(image_data)

            relative_path = f"/media/generated_images/{filename}"
            full_url = f"{self.base_url}{relative_path}" if self.base_url else relative_path

            return json.dumps({
                "type": "image",
                "prompt": prompt,
                "local_path": relative_path,
                "url": full_url,
                "content": f"Image generated: {prompt}\nLocal: {relative_path}"
            })

        except Exception as e:
            return json.dumps({"error": f"Image generation error: {e}", "type": "text"})


class ImageEditTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="image_edit",
            description="Edit existing images. Usage: image_edit: describe the changes you want",
        )

    def execute(self, input_data: str = "") -> str:
        prompt = self._clean(input_data)
        if not prompt:
            return json.dumps({"type": "text", "content": "Describe the edits you want"})

        return json.dumps({
            "type": "text",
            "content": f"Image edit request received.\nEdit: {prompt}\nNote: Image editing requires uploading an image."
        })

    def _clean(self, query: str) -> str:
        if query.lower().startswith("image_edit:"):
            query = query[len("image_edit:"):]
        return query.strip()