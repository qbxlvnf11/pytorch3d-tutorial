from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', help='input mesh data (obj) path',
		default='./processed/zilla_night_black_heat/textured_meshes/optimized_poisson_texture_mapped_mesh.obj') # Dataset: BigBIRD

	return args

args = parse_args()

# Load mesh data with object format
verts, faces, properties = load_obj(args.path)
print('verts:', verts) # v
print('faces:', faces) # f
print('normals:', properties.normals) # vn
print('texture:', properties.verts_uvs) # vt