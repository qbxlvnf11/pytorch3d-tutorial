import argparse

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--obj_path', help='input mesh data (obj) path',
		default='./processed/zilla_night_black_heat/textured_meshes/optimized_poisson_texture_mapped_mesh.obj') # Dataset: BigBIRD
	
	args = parser.parse_args()

	return args

args = parse_args()

# Load mesh data with object format
verts, faces, aux = load_obj(args.obj_path)
# v
print('verts:', verts, len(verts)) # ..., 14614
# f
print('faces:', faces, len(faces.verts_idx), len(faces.normals_idx), len(faces.textures_idx)) # ..., 14610, 14610, 14610
# vn
print('verts normals:', aux.normals, len(aux.normals)) # ..., 7307
# vt
print('verts texture:', aux.verts_uvs, len(aux.verts_uvs)) # ..., 43830

# Pytorch3D Meshes Class
"""
Args:
    verts:
        Can be either

        - List where each element is a tensor of shape (num_verts, 3)
          containing the (x, y, z) coordinates of each vertex.
        - Padded float tensor with shape (num_meshes, max_num_verts, 3).
          Meshes should be padded with fill value of 0 so they all have
          the same number of vertices.
    faces:
        Can be either

        - List where each element is a tensor of shape (num_faces, 3)
          containing the indices of the 3 vertices in the corresponding
          mesh in verts which form the triangular face.
        - Padded long tensor of shape (num_meshes, max_num_faces, 3).
          Meshes should be padded with fill value of -1 so they have
          the same number of faces.
    textures: Optional instance of the Textures class with mesh
        texture properties.
    verts_normals:
        Optional. Can be either

        - List where each element is a tensor of shape (num_verts, 3)
          containing the normals of each vertex.
        - Padded float tensor with shape (num_meshes, max_num_verts, 3).
          They should be padded with fill value of 0 so they all have
          the same number of vertices.
        Note that modifying the mesh later, e.g. with offset_verts_,
        can cause these normals to be forgotten and normals to be recalculated
        based on the new vertex positions.

Refer to comments above for descriptions of List and Padded representations.
"""

'''
    _INTERNAL_TENSORS = [
        "_verts_packed",
        "_verts_packed_to_mesh_idx",
        "_mesh_to_verts_packed_first_idx",
        "_verts_padded",
        "_num_verts_per_mesh",
        "_faces_packed",
        "_faces_packed_to_mesh_idx",
        "_mesh_to_faces_packed_first_idx",
        "_faces_padded",
        "_faces_areas_packed",
        "_verts_normals_packed",
        "_faces_normals_packed",
        "_num_faces_per_mesh",
        "_edges_packed",
        "_edges_packed_to_mesh_idx",
        "_mesh_to_edges_packed_first_idx",
        "_faces_packed_to_edges_packed",
        "_num_edges_per_mesh",
        "_verts_padded_to_packed_idx",
        "_laplacian_packed",
        "valid",
        "equisized",
    ]
'''

# Build Meshes class
test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# Batch size
## self._N = 0  # batch size (number of meshes)
print(f"len(test_mesh): {len(test_mesh)}") # 1

# Number of verts per mesh
"""
Return a 1D tensor x with length equal to the number of meshes giving
the number of vertices in each mesh.

Returns:
    1D tensor of sizes.
"""
print(f"num_verts_per_mesh: {test_mesh.num_verts_per_mesh()}") # 14614

# Get vertes
"""
Get the packed representation of the vertices.

Returns:
    tensor of vertices of shape (sum(V_n), 3).
"""
print(f"test_mesh.verts_packed(): {test_mesh.verts_packed()}")

# Number of edgss per mesh
"""
Return a 1D tensor x with length equal to the number of meshes giving
the number of edges in each mesh.

Returns:
    1D tensor of sizes.
"""
print(f"num_edges_per_mesh: {test_mesh.num_edges_per_mesh()}") # 21915

# Get edges
"""
Get the packed representation of the edges.

Returns:
    tensor of edges of shape (sum(E_n), 2).
"""
print(f"test_mesh.edges_packed(): {test_mesh.edges_packed()}")

# Number of faces per mesh
"""
Return a 1D tensor x with length equal to the number of meshes giving
the number of edges in each mesh.

Returns:
    1D tensor of sizes.
"""
print(f"num_faces_per_mesh: {test_mesh.num_faces_per_mesh()}") # 14610

# Get faces
"""
Get the packed representation of the faces.
Faces are given by the indices of the three vertices in verts_packed.

Returns:
    tensor of faces of shape (sum(F_n), 3).
"""
print(f"test_mesh.faces_packed(): {test_mesh.faces_packed()}")

# Get verts normals
"""
Get the packed representation of the vertex normals.

Returns:
    tensor of normals of shape (sum(V_n), 3).
"""
print(f"test_mesh.verts_normals_packed(): {test_mesh.verts_normals_packed()}, {len(test_mesh.verts_normals_packed())}") # ..., 14614

# Get faces normals
"""
Get the packed representation of the face normals.

Returns:
    tensor of normals of shape (sum(F_n), 3).
"""
print(f"test_mesh.faces_normals_packed(): {test_mesh.faces_normals_packed()}, {len(test_mesh.faces_normals_packed())}") # ..., 14610


