from pytorch3d.io import load_obj
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--obj_path', help='input mesh data (obj) path',
		default='./processed/zilla_night_black_heat/textured_meshes/optimized_poisson_texture_mapped_mesh.obj') # Dataset: BigBIRD
	
	args = parser.parse_args()

	return args

args = parse_args()

# Load mesh data with object format
verts, faces, aux = load_obj(args.obj_path)
print('verts:', verts) # v
print('faces:', faces) # f
print('verts normals:', aux.normals) # vn
print('verts texture:', aux.verts_uvs) # vt
