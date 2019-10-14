import os
import random
import numpy as np
import bpy
from mathutils import Vector


def import_object(path):
    if not os.path.isfile(path):  # Skip if file is not existant
        return False
    if path.endswith('off'):
        bpy.ops.import_mesh.off(filepath=path, axis_up="-Z")
    if path.endswith('stl'):
        bpy.ops.import_mesh.stl(filepath=path, axis_up="-Z")
    elif path.endswith('obj'):
        bpy.ops.import_scene.obj(filepath=path, axis_up="-Z")
    return True


def get_object_by_name(name):
    objects = []
    for obj in bpy.data.objects:
        if name in obj.name:
            objects.append(obj)
    return objects


def get_object_by_prefix(prefix, inverse=True):
    objects = []
    for obj in bpy.data.objects:
        if inverse:
            if not obj.name.startswith(prefix):
                objects.append(obj)
        else:
            if obj.name.startswith(prefix):
                objects.append(obj)
    return objects


def make_normals_outwards():
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)


def setup_environment(o):
    bpy.ops.object.select_all(action='DESELECT')
    o.select = True
    bpy.context.scene.objects.active = o
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.reveal()


def scale_object(o, factor=None):
    if factor:
        md = factor
    else:
        md = 1.0 / np.max(o.dimensions)
    for p in o.data.polygons:
        p.use_smooth = True
    o.scale *= md
    o.select = True
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print("Object rescaled with factor %.2f" % md)
    return md


def set_pose(o):
    # bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    o.location = Vector([0, 0, 0])
    o.rotation_euler = Vector([0, 0, 0])


#    print("Object pose set")
#        local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
#        global_bbox_center = obj.matrix_world * local_bbox_center
#        obj.location -= global_bbox_center

def uniform_random_quaternions(N=1):
    # Generates a uniform random quaternion
    # James J. Kuffner 2004
    # A random array 3xN
    s = np.random.rand(3, N)
    sigma1 = np.sqrt(1.0 - s[0])
    sigma2 = np.sqrt(s[0])
    theta1 = 2 * np.pi * s[1]
    theta2 = 2 * np.pi * s[2]
    w = np.cos(theta2) * sigma2
    x = np.sin(theta1) * sigma1
    y = np.cos(theta1) * sigma1
    z = np.sin(theta2) * sigma2
    return np.array([w, x, y, z])


def fibonacci_sphere_sampling(samples=1, randomize=True, radius=1.0, positive_z=False):
    # Returns [x,y,z] tuples of a fibonacci sphere sampling
    if positive_z:
        samples *= 2
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        if positive_z:
            s = np.arcsin(z / radius) * 180.0 / np.pi
            if z > 0.0 and s > 30:
                points.append([radius * x, radius * y, radius * z])
        else:
            points.append([radius * x, radius * y, radius * z])

    return points


def remove_doubles(o):
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_limited()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    found = set([])  # set of found sorted vertices pairs

    for face in o.data.polygons:
        facevertsorted = sorted(face.vertices[:])  # sort vertices of the face to compare later
        if str(facevertsorted) not in found:  # if sorted vertices are not in the set
            found.add(str(facevertsorted))  # add them in the set
            o.data.polygons[face.index].select = False  # deselect faces i want to keep

    bpy.ops.object.mode_set(mode='EDIT', toggle=False)  # set to Edit Mode AGAIN
    bpy.ops.mesh.delete(type='FACE')  # delete double faces
    bpy.ops.mesh.select_mode(type="VERT")


def fix_non_manifold(o):
    mesh = o.data
    non_manifold_vertices = get_non_manifold_vertices(mesh)
    current_iteration = 0

    while len(non_manifold_vertices) > 0:

        if current_iteration > 100:
            raise RuntimeError("Exceeded maximum iterations, terminated early")

        fill_non_manifold()
        make_normals_outwards()
        # delete_newly_generated_non_manifold_vertices()

        new_non_manifold_vertices = get_non_manifold_vertices(mesh)
        if new_non_manifold_vertices == non_manifold_vertices:
            return
            # raise RuntimeError("Not possible to repair, non-ending loop occurred")
        else:
            non_manifold_vertices = new_non_manifold_vertices
            current_iteration += 1


def select_non_manifold_vertices():
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold()


def selected_vertices_to_coords(mesh):
    # Have to toggle mode for select vertices to work
    bpy.ops.object.mode_set(mode="OBJECT")
    selected_vertices = {(v.co[0], v.co[1], v.co[2]) for v in mesh.vertices if v.select}
    bpy.ops.object.mode_set(mode="EDIT")

    return selected_vertices


def get_non_manifold_vertices(mesh):
    select_non_manifold_vertices()
    print("Non-manifold remaining:", mesh.total_vert_sel)
    return selected_vertices_to_coords(mesh)


def fill_non_manifold():
    bpy.ops.mesh.fill_holes(sides=10)

    # fill selected edge faces, which could be additional holes
    # select_non_manifold_vertices()
    # bpy.ops.mesh.fill()


def delete_newly_generated_non_manifold_vertices():
    select_non_manifold_vertices()
    bpy.ops.mesh.delete(type="VERT")


def sample_surface(mesh, count):
    # len(mesh.faces) float array of the areas of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index
