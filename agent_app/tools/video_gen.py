import os
import subprocess
import json
import uuid
import math
import random
import shutil
import time
from pathlib import Path
from ..core import BaseTool


class VideoGenTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="video_gen",
            description="Generate videos locally using Blender 3D animation. Usage: video_gen: a rotating cube",
        )
        self.save_dir = "media/generated_videos"
        self.blender_path = os.environ.get("BLENDER_PATH", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
        os.makedirs(self.save_dir, exist_ok=True)
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        for path in ["ffmpeg", "ffmpeg.exe", "C:/ffmpeg/bin/ffmpeg.exe"]:
            try:
                result = subprocess.run([path, "-version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
        return None

    def execute(self, input_data: str = "") -> str:
        prompt = self._clean(input_data)
        if not prompt:
            return json.dumps({
                "type": "text",
                "content": "Provide a description, e.g. video_gen: a rotating cube with red material"
            })

        if not self._check_blender():
            return json.dumps({
                "type": "text",
                "content": (
                    "Blender not found or not configured.\n"
                    "1. Install Blender from https://blender.org\n"
                    "2. Set BLENDER_PATH in .env\n"
                    "3. Or use: video_gen: simple rotating cube"
                )
            })

        try:
            script_path = self._generate_blender_script(prompt)
            result = self._run_blender_render(script_path, prompt)
            return result
        except Exception as e:
            return json.dumps({"type": "text", "content": f"Video generation error: {e}"})

    def _clean(self, query: str) -> str:
        if query.lower().startswith("video_gen:"):
            query = query[len("video_gen:"):]
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
        session_id = uuid.uuid4().hex
        frames_dir = os.path.join(self.save_dir, session_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        frames_path = os.path.join(frames_dir, "frame_").replace("\\", "\\\\")
        
        script_content = f'''
import bpy
import math
import random

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)

for mat in bpy.data.materials:
    bpy.data.materials.remove(mat)

random.seed(42)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 30
scene.render.engine = 'CYCLES'
scene.cycles.samples = 16
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 24
scene.render.image_settings.file_format = 'PNG'

prompt_lower = """{prompt}""".lower()

colors = [
    (0.8, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.3, 0.8),
    (0.9, 0.7, 0.1),
    (0.6, 0.2, 0.8),
    (0.1, 0.9, 0.9),
]

def add_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    return mat

if any(w in prompt_lower for w in ["sphere", "ball", "globe", "planet", "earth"]):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.materials.append(add_material("Mat", random.choice(colors)))
    obj.rotation_mode = 'XYZ'
    for f in range(0, 31, 5):
        obj.rotation_euler = (f * 0.05, f * 0.03, f * 0.02)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif any(w in prompt_lower for w in ["cube", "box", "square", "dice"]):
    bpy.ops.mesh.primitive_cube_add(size=2.5, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.materials.append(add_material("Mat", random.choice(colors)))
    obj.rotation_mode = 'XYZ'
    for f in range(0, 31, 5):
        obj.rotation_euler = (f * 0.08, f * 0.05, 0)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif any(w in prompt_lower for w in ["torus", "ring", "donut"]):
    bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.5, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.materials.append(add_material("Mat", random.choice(colors)))
    obj.rotation_mode = 'XYZ'
    for f in range(0, 31, 5):
        obj.rotation_euler = (f * 0.03, f * 0.06, 0)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif any(w in prompt_lower for w in ["cone", "pyramid"]):
    bpy.ops.mesh.primitive_cone_add(radius1=1.5, depth=3, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.materials.append(add_material("Mat", random.choice(colors)))
    obj.rotation_mode = 'XYZ'
    for f in range(0, 31, 5):
        obj.rotation_euler = (0, 0, f * 0.1)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)

else:
    for i in range(2):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        z = random.uniform(0, 2)
        obj_type = random.choice(['CUBE', 'SPHERE', 'TORUS'])
        
        if obj_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(size=1.5, location=(x, y, z))
        elif obj_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x, y, z))
        else:
            bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.3, location=(x, y, z))
        
        obj = bpy.context.active_object
        mat_name = "Mat" + str(i)
        obj.data.materials.append(add_material(mat_name, random.choice(colors)))
        obj.rotation_mode = 'XYZ'
        
        for f in range(0, 31, 10):
            obj.rotation_euler = (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))
            obj.keyframe_insert(data_path="rotation_euler", frame=f)

bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
bpy.ops.object.camera_add(location=(6, -6, 4))
camera = bpy.context.active_object
camera.rotation_euler = (math.radians(60), 0, math.radians(45))
scene.camera = camera

scene.render.filepath = r"{frames_path}"
bpy.ops.render.render(animation=True)
print("FRAMES_DIR:{frames_dir}")
'''
        filename = f"{session_id}.py"
        filepath = os.path.join(frames_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(script_content)
        return filepath, session_id, frames_dir

    def _run_blender_render(self, script_path: str, prompt: str) -> str:
        script_path, session_id, frames_dir = script_path
        
        try:
            result = subprocess.run(
                [self.blender_path, "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=300
            )

            png_files = sorted(Path(frames_dir).glob("frame_*.png"))
            
            if not png_files:
                return json.dumps({
                    "type": "text",
                    "content": f"Blender render failed.\n\nError: {result.stderr[:500]}"
                })

            video_path = None
            
            if self.ffmpeg_path:
                video_file = f"{session_id}.mp4"
                video_path = os.path.join(self.save_dir, video_file)
                frames_pattern = os.path.join(frames_dir, "frame_%04d.png")
                
                ffmpeg_cmd = [
                    self.ffmpeg_path,
                    "-framerate", "24",
                    "-i", frames_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    video_path
                ]
                
                ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
                
                if ffmpeg_result.returncode == 0 and os.path.exists(video_path):
                    relative_path = f"/media/generated_videos/{video_file}"
                    return json.dumps({
                        "type": "video",
                        "prompt": prompt,
                        "url": relative_path,
                        "local_path": video_path,
                        "content": f"Video generated locally with Blender: {prompt}\nSaved to: {relative_path}"
                    })

            first_frame = png_files[0]
            frames_html = "".join([f'<img src="/media/generated_videos/{session_id}/{f.name}" style="width:100%;display:none;" />' for f in png_files])
            
            return json.dumps({
                "type": "video",
                "prompt": prompt,
                "url": f"/media/generated_videos/{session_id}/{first_frame.name}",
                "local_path": str(first_frame),
                "frames": [f.name for f in png_files],
                "content": f"3D Animation generated: {prompt}\nFrames: {len(png_files)} - Use frontend to view animation"
            })

        except subprocess.TimeoutExpired:
            return json.dumps({"type": "text", "content": "Render timeout. Try a simpler scene."})
        except Exception as e:
            return json.dumps({"type": "text", "content": f"Video generation error: {e}"})


class VideoEditTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="video_edit",
            description="Edit existing videos. Usage: video_edit: describe the changes you want",
        )

    def execute(self, input_data: str = "") -> str:
        prompt = self._clean(input_data)
        if not prompt:
            return json.dumps({
                "type": "text",
                "content": "Describe the edits you want, e.g. video_edit: add more rotation"
            })

        return json.dumps({
            "type": "text",
            "content": f"Video edit request received.\nEdit: {prompt}\nNote: Video editing requires Blender to be installed."
        })

    def _clean(self, query: str) -> str:
        if query.lower().startswith("video_edit:"):
            query = query[len("video_edit:"):]
        return query.strip()