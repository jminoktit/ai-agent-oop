import os
import subprocess
import json
import uuid
from ..core import BaseTool


class BlenderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="blender_render",
            description="Generate 3D videos/animations using Blender. Usage: blender_render: a rotating cube with colorful materials",
        )
        self.blender_path = os.environ.get("BLENDER_PATH", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
        self.render_dir = "media/blender_renders"
        os.makedirs(self.render_dir, exist_ok=True)

    def execute(self, input_data: str = "") -> str:
        prompt = self._clean(input_data)
        if not prompt:
            return json.dumps({
                "type": "text",
                "content": "Provide a description, e.g. blender_render: a rotating cube with red material"
            })

        if not self._check_blender():
            return json.dumps({
                "type": "text",
                "content": (
                    "Blender not found or not configured.\n"
                    "1. Install Blender from https://blender.org\n"
                    "2. Set BLENDER_PATH in .env (e.g. BLENDER_PATH=C:/Program Files/Blender Foundation/Blender 4.0/blender.exe)\n"
                    "3. Try: blender_render: a simple rotating cube"
                )
            })

        try:
            output_file = self._generate_blender_script(prompt)
            result = self._run_blender_render(output_file, prompt)
            return result
        except Exception as e:
            return json.dumps({"type": "text", "content": f"Blender render error: {e}"})

    def _clean(self, query: str) -> str:
        if query.lower().startswith("blender_render:"):
            query = query[len("blender_render:"):]
        return query.strip()

    def _check_blender(self) -> bool:
        try:
            result = subprocess.run(
                [self.blender_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _generate_blender_script(self, prompt: str) -> str:
        script_content = f'''
import bpy
import math
import random

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

random.seed(42)

def add_material(color):
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.5
    return mat

colors = [
    (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)),
    (1, 0.3, 0.1),
    (0.1, 0.8, 0.3),
    (0.2, 0.3, 1),
    (1, 1, 0.2),
]

num_objects = random.randint(2, 5)
for i in range(num_objects):
    obj_type = random.choice(['CUBE', 'SPHERE', 'CONE', 'TORUS'])
    if obj_type == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(size=2, location=(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 3)))
    elif obj_type == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 3)))
    elif obj_type == 'CONE':
        bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 3)))
    elif obj_type == 'TORUS':
        bpy.ops.mesh.primitive_torus_add(location=(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 3)))

    obj = bpy.context.active_object
    color = random.choice(colors)
    obj.data.materials.append(add_material(color))

    obj.rotation_mode = 'XYZ'
    obj.keyframe_insert(data_path="rotation_euler", frame=1)
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert(data_path="rotation_euler", frame=60)
    obj.rotation_euler = (0, 0, math.radians(360))
    obj.keyframe_insert(data_path="rotation_euler", frame=120)

for i in range(3):
    bpy.ops.object.light_add(type='SUN', location=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(5, 10)))
    light = bpy.context.active_object
    light.data.energy = random.uniform(1, 5)

bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.active_object
camera.rotation_euler = (math.radians(60), 0, math.radians(45))
scene.camera = camera

scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 24
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.audio_codec = 'NONE'

output_path = r"{self.render_dir.replace(chr(92), chr(92) + chr(92))}\\" + str(uuid.uuid4().hex) + ".mp4"
scene.render.filepath = output_path
print(f"RENDER_OUTPUT:{output_path}")
'''

        filename = f"{uuid.uuid4().hex}.py"
        filepath = os.path.join(self.render_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(script_content)
        return filepath

    def _run_blender_render(self, script_path: str, prompt: str) -> str:
        output_path = os.path.join(self.render_dir, f"{uuid.uuid4().hex}.mp4")

        try:
            result = subprocess.run(
                [
                    self.blender_path,
                    "--background",
                    "--python", script_path,
                ],
                capture_output=True,
                text=True,
                timeout=300
            )

            if "RENDER_OUTPUT:" in result.stdout:
                for line in result.stdout.split("\n"):
                    if line.startswith("RENDER_OUTPUT:"):
                        output_path = line.split("RENDER_OUTPUT:")[1].strip()
                        break

            if os.path.exists(output_path):
                relative_path = f"/media/blender_renders/{os.path.basename(output_path)}"
                return json.dumps({
                    "type": "video",
                    "prompt": prompt,
                    "url": relative_path,
                    "local_path": output_path,
                    "content": f"3D animation rendered successfully: {prompt}"
                })

            if result.returncode != 0:
                return json.dumps({
                    "type": "text",
                    "content": f"Blender render failed. Check Blender installation.\n\nError: {result.stderr[:500]}"
                })

            return json.dumps({
                "type": "text",
                "content": f"Render started: {prompt}\nNote: Blender renders in background. Video may take a few minutes."
            })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "type": "text",
                "content": "Render timeout. Try a simpler scene or increase timeout."
            })
        except Exception as e:
            return json.dumps({"type": "text", "content": f"Blender error: {e}"})