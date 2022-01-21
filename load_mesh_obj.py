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
    
    """
    Load a mesh from a .obj file and optionally textures from a .mtl file.
    Currently this handles verts, faces, vertex texture uv coordinates, normals,
    texture images and material reflectivity values.
    Note .obj files are 1-indexed. The tensors returned from this function
    are 0-indexed. OBJ spec reference: http://www.martinreddy.net/gfx/3d/OBJ.spec
    Example .obj file format:
    ::
        # this is a comment
        v 1.000000 -1.000000 -1.000000
        v 1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 -1.000000
        v 1.000000 1.000000 -1.000000
        vt 0.748573 0.750412
        vt 0.749279 0.501284
        vt 0.999110 0.501077
        vt 0.999455 0.750380
        vn 0.000000 0.000000 -1.000000
        vn -1.000000 -0.000000 -0.000000
        vn -0.000000 -0.000000 1.000000
        f 5/2/1 1/2/1 4/3/1
        f 5/1/1 4/3/1 2/4/1
    The first character of the line denotes the type of input:
    ::
        - v is a vertex
        - vt is the texture coordinate of one vertex
        - vn is the normal of one vertex
        - f is a face
    Faces are interpreted as follows:
    ::
        5/2/1 describes the first vertex of the first triangle
        - 5: index of vertex [1.000000 1.000000 -1.000000]
        - 2: index of texture coordinate [0.749279 0.501284]
        - 1: index of normal [0.000000 0.000000 -1.000000]
    If there are faces with more than 3 vertices
    they are subdivided into triangles. Polygonal faces are assumed to have
    vertices ordered counter-clockwise so the (right-handed) normal points
    out of the screen e.g. a proper rectangular face would be specified like this:
    ::
        0_________1
        |         |
        |         |
        3 ________2
    The face would be split into two triangles: (0, 2, 1) and (0, 3, 2),
    both of which are also oriented counter-clockwise and have normals
    pointing out of the screen.
    Args:
        f: A file-like object (with methods read, readline, tell, and seek),
           a pathlib path or a string containing a file name.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas: Bool, If True a per face texture map is created and
            a tensor `texture_atlas` is also returned in `aux`.
        texture_atlas_size: Int specifying the resolution of the texture map per face
            when `create_texture_atlas=True`. A (texture_size, texture_size, 3)
            map is created per face.
        texture_wrap: string, one of ["repeat", "clamp"]. This applies when computing
            the texture atlas.
            If `texture_mode="repeat"`, for uv values outside the range [0, 1] the integer part
            is ignored and a repeating pattern is formed.
            If `texture_mode="clamp"` the values are clamped to the range [0, 1].
            If None, then there is no transformation of the texture values.
        device: Device (as str or torch.device) on which to return the new tensors.
        path_manager: optionally a PathManager object to interpret paths.
    Returns:
        6-element tuple containing
        - **verts**: FloatTensor of shape (V, 3).
        - **faces**: NamedTuple with fields:
            - verts_idx: LongTensor of vertex indices, shape (F, 3).
            - normals_idx: (optional) LongTensor of normal indices, shape (F, 3).
            - textures_idx: (optional) LongTensor of texture indices, shape (F, 3).
              This can be used to index into verts_uvs.
            - materials_idx: (optional) List of indices indicating which
              material the texture is derived from for each face.
              If there is no material for a face, the index is -1.
              This can be used to retrieve the corresponding values
              in material_colors/texture_images after they have been
              converted to tensors or Materials/Textures data
              structures - see textures.py and materials.py for
              more info.
        - **aux**: NamedTuple with fields:
            - normals: FloatTensor of shape (N, 3)
            - verts_uvs: FloatTensor of shape (T, 2), giving the uv coordinate per
              vertex. If a vertex is shared between two faces, it can have
              a different uv value for each instance. Therefore it is
              possible that the number of verts_uvs is greater than
              num verts i.e. T > V.
              vertex.
            - material_colors: if `load_textures=True` and the material has associated
              properties this will be a dict of material names and properties of the form:
              .. code-block:: python
                  {
                      material_name_1:  {
                          "ambient_color": tensor of shape (1, 3),
                          "diffuse_color": tensor of shape (1, 3),
                          "specular_color": tensor of shape (1, 3),
                          "shininess": tensor of shape (1)
                      },
                      material_name_2: {},
                      ...
                  }
              If a material does not have any properties it will have an
              empty dict. If `load_textures=False`, `material_colors` will None.
            - texture_images: if `load_textures=True` and the material has a texture map,
              this will be a dict of the form:
              .. code-block:: python
                  {
                      material_name_1: (H, W, 3) image,
                      ...
                  }
              If `load_textures=False`, `texture_images` will None.
            - texture_atlas: if `load_textures=True` and `create_texture_atlas=True`,
              this will be a FloatTensor of the form: (F, texture_size, textures_size, 3)
              If the material does not have a texture map, then all faces
              will have a uniform white texture.  Otherwise `texture_atlas` will be
              None.
    """
    
verts, faces, aux = load_obj(args.obj_path)
print('verts:', verts) # v
print('faces:', faces) # f
print('verts normals:', aux.normals) # vn
print('verts texture:', aux.verts_uvs) # vt
