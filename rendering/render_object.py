# import importlib
import os
import sys

import bmesh
import bpy
import numpy as np
import time
import png
import yaml
from mathutils import Vector, Matrix

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

import utilities
#importlib.reload(utils)
from utilities import import_object, get_object_by_prefix, get_object_by_name, setup_environment, remove_doubles, fix_non_manifold, make_normals_outwards, scale_object, set_pose, fibonacci_sphere_sampling, uniform_random_quaternions

m_path = sys.argv[6]
# m_path = "../data/tmp/0.dat"
with open(m_path) as fp:
    p = yaml.safe_load(fp)
os.remove(m_path)

# scales = np.linspace(0.4, 1.0, p["scales"]) # TUB dataset
distances = np.linspace(p["camera_dist_limits"][0], p["camera_dist_limits"][1], p["camera_distances"])  # SIE dataset
passes = [p["line_image"], p["depth_image"], p["color_image"]]

# General settings
bpy.context.scene.cycles.device = p["device"]
# bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "OPENCL"
# bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True

# Load object and get camera/light
m_path = p["model"]
assert (import_object(m_path))
# r_path = p["model"] + "_%i_render"%p["count"]
# os.makedirs(r_path, exist_ok=True)

obj = get_object_by_prefix("Static_")[0]
camera = get_object_by_name("Static_Camera")[0]
light = get_object_by_name("Static_Sun")[0]

# Repair models
if p["object_repair"]:
    setup_environment(obj)
    remove_doubles(obj)
    fix_non_manifold(obj)
    make_normals_outwards()
    bpy.ops.object.mode_set(mode="OBJECT")

# Find split components
if p["object_split_components"]:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    bpy.ops.mesh.separate(type='LOOSE')
    objects = get_object_by_prefix("Static_")
    # print(objects, file=sys.stderr)
else:
    objects = [obj]


for cmp, obj in enumerate(objects):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True

    # Model adjustments
    if p["object_scale_factor"] == -1:
        factor = scale_object(obj)
    else:
        factor = scale_object(obj, p["object_scale_factor"])  # SIE dataset
    set_pose(obj)
    # set_material(obj)
    stat_path = p["folder"].replace("/objects", "/meta")
    fac_path = os.path.join(stat_path, "render_meta_%i.yml" % p["count"])
    p["b_scale_factor_cmp%i" % cmp] = str(factor)

    obj.location = Vector((0.0, 0.0, 10.0))  # Move objects away
    obj.hide_render = True  # and hide them

component_poses = []
total_poses = 0

for cmp, obj in enumerate(objects):
    # Find render poses of objects
    poses = []

    if p["static_poses"]:
        # def stop_playback(scene):
        #     if scene.frame_current == scene.frame_end:
        #         bpy.ops.screen.animation_cancel(restore_frame=False)
        #
        # def stop_playback_restore(scene):
        #     if scene.frame_current == scene.frame_end + 1:
        #         bpy.ops.screen.animation_cancel(restore_frame=True)
        #
        # bpy.app.handlers.frame_change_pre.append(stop_playback)

        bpy.ops.mesh.primitive_plane_add(radius=10, view_align=False, enter_editmode=False, location=(0, 0, 0))
        plane = get_object_by_name("Plane")[0]
        plane.select = True
        bpy.ops.rigidbody.object_add()
        plane.rigid_body.type = 'PASSIVE'
        plane.rigid_body.collision_shape = 'BOX'
        plane.rigid_body.use_margin = True
        plane.rigid_body.collision_margin = 0.5

        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.shade_flat()
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.collision_shape = p["static_collision_type"]
        obj.rigid_body.friction = 0.7
        obj.rigid_body.linear_damping = 0.5
        obj.rigid_body.angular_damping = 0.5
        obj.rotation_mode = "AXIS_ANGLE"

        locations = []
        rotations = []
        sam = p["static_pose_trials"]
        rot_samples = fibonacci_sphere_sampling(samples=sam, randomize=False, radius=1.0)
        for trial in range(sam):
            obj.location = Vector((0.0, 0.0, np.max(obj.dimensions) + 0.2))

            # angle = np.random.rand() * np.pi
            # aa = [angle]
            # aa.extend(rot_samples[trial])
            # obj.rotation_axis_angle = Vector(aa)
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = uniform_random_quaternions()
            obj.rotation_mode = "AXIS_ANGLE"

            for i in range(100):
                bpy.context.scene.frame_set(i)
            bpy.ops.object.visual_transform_apply()
            bpy.context.scene.frame_set(1)
            bpy.context.scene.update()

            locations.append(obj.location.copy())
            rotations.append((Matrix.Rotation(obj.rotation_axis_angle[0], 4, obj.rotation_axis_angle[1:]),
                              Vector(obj.rotation_axis_angle)))

        cmp_z = Vector([0.0, 0.0, 1.0])
        accept = []
        reject = []
        accept_count = []
        for a in range(sam):
            if a not in reject:
                accept.append(a)
            else:

                continue
            cnt = 0
            for b in range(a + 1, sam):
                if b in reject:
                    continue
                z1 = rotations[a][0].inverted() * cmp_z
                z2 = rotations[b][0].inverted() * cmp_z
                # print(z1, z2)
                if z1.dot(z2) > 0.7:
                    reject.append(b)
                    cnt += 1

            accept_count.append(cnt)

        a_list, a_cnt = [list(x)[::-1] for x in zip(*sorted(zip(accept, accept_count), key=lambda pair: pair[1]))]
        save_poses = []
        for i, a in enumerate(a_list):
            if a_cnt[i] < (p["static_pose_trials"] * p["static_pose_thresh"]):
                continue
            poses.append({"location": Vector([0.0, 0.0, locations[a][2]]), "rotation": rotations[a][1]})
            save_poses.append(
                {"location": [0.0, 0.0, locations[a][2]], "rotation": [rotations[a][1][i] for i in range(4)]})
        # print(name, a_cnt, len(poses), file=sys.stderr)
        p["b_object_poses_cmp%i"%cmp] = save_poses
        p["b_object_pose_counts_cmp%i"%cmp] = a_cnt
        # print(len(accept), len(reject))
        # time.sleep(2.0)
        positive_z = True
        sampling_factor = 2.0
        camera.constraints["Track To"].target = bpy.data.objects[obj.name]
        if p["static_max_poses"] != -1:
            if p["static_max_poses"] < len(poses):
                poses = poses[:p["static_max_poses"]]

        plane.select = True
        obj.select = False
        bpy.ops.object.delete()


    else:
        poses.append({"location": Vector([0.0, 0.0, 0.0]), "rotation": Vector([0.0, 1.0, 0.0, 0.0])})
        positive_z = False
        sampling_factor = 1.0

    component_poses.append(poses)
    total_poses += len(poses)


def adjust_pass(i, mode_map):
    if mode_map[i] == "depth":
        #            camera.data.angle = 1.0821#1.01229 Kinect
        obj.cycles_visibility.camera = True
        bpy.data.objects["Static_Floorplane"].hide_render = not p["render_depth_ground"]
        bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = True
        bpy.data.scenes["Scene"].render.use_freestyle = False
        bpy.data.scenes["Scene"].cycles.use_square_samples = False
        bpy.data.scenes["Scene"].render.image_settings.color_mode = "RGBA"
        bpy.data.scenes["Scene"].render.image_settings.color_depth = "16"

    if mode_map[i] == "color":
        #            camera.data.angle = 1.0821
        obj.cycles_visibility.camera = True
        bpy.data.objects["Static_Floorplane"].hide_render = True
        bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = False
        bpy.data.scenes["Scene"].render.use_freestyle = False
        bpy.data.scenes["Scene"].cycles.use_square_samples = False
        bpy.data.scenes["Scene"].render.image_settings.color_mode = p["render_color_mode"]
        bpy.data.scenes["Scene"].render.image_settings.color_depth = p["render_color_depth"]

    if mode_map[i] == "line":
        #            camera.data.angle = 1.0821
        obj.cycles_visibility.camera = False
        bpy.data.objects["Static_Floorplane"].hide_render = True
        bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = False
        bpy.data.scenes["Scene"].render.use_freestyle = True
        bpy.data.scenes["Scene"].cycles.use_square_samples = False
        bpy.data.scenes["Scene"].render.image_settings.color_mode = "RGBA"
        bpy.data.scenes["Scene"].render.image_settings.color_depth = "8"


# Adjust render settings
bpy.data.scenes["Scene"].render.resolution_x = p["camera_render_size"][0]
bpy.data.scenes["Scene"].render.resolution_y = p["camera_render_size"][1]

# Create camera sampling
if p["camera_sampling_type"] == "FIB_SPHERE":
    camera_sampling = fibonacci_sphere_sampling(samples=int(p["camera_samples"] * sampling_factor), randomize=False,
                                                radius=1.0, positive_z=positive_z)
if p["camera_sampling_type"] == "FIXED":
    camera_sampling = p["camera_sampling"]

if p["camera_sampling_type"] == "ARC":
    angles = np.linspace(np.deg2rad(p["camera_arc_angle_limits"][0]), np.deg2rad(p["camera_arc_angle_limits"][1]), p["camera_samples"], endpoint=False)
    zs = np.sin(angles)
    ys = np.cos(angles)
    camera_sampling = []
    for i_s in range(p["camera_samples"]):
        camera_sampling.append([0.0, ys[i_s], zs[i_s]])
# print(camera_sampling, file=sys.stderr)

# Create object sampling
object_sampling = [Matrix.Rotation(0.0, 4, Vector((0, 0, 1)))]
if p["object_sampling_type"] == "ROT_Z":
    delta_angle = 2 * np.pi / p["object_samples"]
    rot = Matrix.Rotation(delta_angle, 4, Vector((0, 0, 1)))
    for s in range(p["object_samples"] - 1):
        object_sampling.append(rot)

mode_map = ["line", "depth", "color"]
p["total_renderings"] = total_poses * sum(passes) * len(object_sampling) * len(distances) * len(camera_sampling)
with open(fac_path, "w") as f:
    yaml.dump(p, f, allow_unicode=True)

for cmp, obj in enumerate(objects):
    obj.hide_render = False
    camera.constraints["Track To"].target = bpy.data.objects[obj.name]
    poses = component_poses[cmp]
    # Render models with different poses, modes, scales and camera positions
    for h, pose in enumerate(poses):  # Iterate over static object poses
        obj.location = pose["location"]
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = pose["rotation"]
        obj.rotation_mode = "XYZ"

        for i, pa in enumerate(passes):  # Iterate over rendering types
            if not pa:  # Switch pass if it is not enabled
                continue
            adjust_pass(i, mode_map)

            for r, rotation in enumerate(object_sampling):  # Iterate over object rotations
                mat = obj.rotation_euler.to_matrix()
                mat.resize_4x4()
                obj.rotation_euler = (rotation * mat).to_euler()

                for j, scale in enumerate(distances):  # Iterate over camera distances

                    for k, position in enumerate(camera_sampling):  # Iterate over camera positions
                        camera.location = scale * Vector(position)
                        image_path = p["folder"].replace("/objects", "/images")
                        if not os.path.isdir("%s/%02d" % (image_path, p["count"])):
                            os.mkdir("%s/%02d" % (image_path, p["count"]))
                        path = "%s/%02d/%02d_cmp%02d_pose%02d_rot%04d_dist%02d_cam%03d_%s.png"
                        path = path % (image_path, p["count"], p["count"], cmp, h, r, j, k, mode_map[i])
                        #path = path.replace("results", "renderings")
                        #                        exit()
                        if not os.path.isfile(path):
                            bpy.data.scenes['Scene'].render.filepath = path
                            if p["debug"]:
                                print("Rendering file %s"%path, file=sys.stderr)
                            if mode_map[i] == "depth":
                                bpy.ops.render.render()
                                a = np.array(bpy.data.images['Viewer Node'].pixels[:])
                                a = a[::4]
                                a *= 1000  # Adjust for blender scale
                                a = np.clip(a, p["render_depth_range"][0],
                                            p["render_depth_range"][1])  # Clip depth values
                                a = a.astype(np.uint16)
                                #                        a[a < 400] = 0
                                a = a.reshape(p["camera_render_size"])
                                a = np.flipud(a)
                                # print(np.min(a), np.max(a))
                                with open(path, "wb") as f:
                                    writer = png.Writer(width=a.shape[1], height=a.shape[0],
                                                        bitdepth=16, greyscale=True)
                                    al = a.tolist()
                                    writer.write(f, al)
                            else:
                                # pass
                                bpy.ops.render.render(write_still=True)
                        # bpy.ops.mesh.primitive_uv_sphere_add(size=0.05, location=camera.location)

                        # print progress to stderr:
                        msg = 'Object: {}\tRot: {}/{}\tCampos: {}/{}'.format(
                            p["count"], r + 1, len(object_sampling), k + 1, len(camera_sampling))
                        print(50 * ' ', end='\r', flush=True, file=sys.stderr)
                        print(msg, end='\r', flush=True, file=sys.stderr)

    # obj.location = Vector((0.0, 0.0, 10.0))  # Move objects away
    obj.hide_render = True

    # if p["point_cloud"]:

    # obj.select = True
    # bpy.context.scene.objects.active = obj
    # scn = bpy.data.scenes["Scene"]
    # scn.objects.unlink(obj)
    # bpy.data.objects.remove(obj)
