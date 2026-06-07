
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

prompt_lower = """a rotating cube""".lower()

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

scene.render.filepath = r"media/generated_videos\\a96bbb412c4b4d89bd43e1907842f32a\\frame_"
bpy.ops.render.render(animation=True)
print("FRAMES_DIR:media/generated_videos\a96bbb412c4b4d89bd43e1907842f32a")
