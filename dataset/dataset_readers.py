import os
import sys
from PIL import Image
from typing import NamedTuple
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import imageio.v2 as imageio
import glob
# import pyexr
import re

from utils.sh import SH2RGB
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics import getWorld2View2, focal2fov, fov2focal, BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    image_id: int = None
    normal: Image.Image = None
    alpha: Image.Image = None
    depth: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        image, is_hdr = load_img(image_path)
        mask_path = os.path.join(images_folder, "../masks", image_name+'.png')
        # print(mask_path)
        if os.path.exists(mask_path):
            img_mask = (imageio.imread(mask_path, pilmode='L') > 0.1).astype(np.float32)
            # if pmask is available, mask the image for PSNR
            image *= img_mask[..., np.newaxis]
        else:
            img_mask = None
            
        image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
        alpha = Image.fromarray(np.array(np.tile(img_mask[..., np.newaxis],(1, 1, 3))*255.0, dtype=np.byte), "RGB") if img_mask is not None else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, image_id=extr.id, alpha=alpha)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, log=None, debug=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if debug:
        train_cam_infos = train_cam_infos[:5]
        test_cam_infos = test_cam_infos[:5]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        log.info("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, background, extension=".png", debug=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            if debug and idx > 5: break
            image_path = os.path.join(path, frame["file_path"] + extension)
            normal_path = os.path.join(path, frame["file_path"] + "_normal" + extension)
            alpha_path = os.path.join(path, frame["file_path"] + "_alpha" + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(image_path).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(background)

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if os.path.exists(normal_path):
                normal = Image.open(normal_path).convert('RGB')
            else:
                normal = None
            
            if os.path.exists(alpha_path):
                alpha = Image.open(alpha_path).convert('RGB')
            elif normal is not None:
                normal_data = np.array(normal) * 2. / 255.0 - 1.
                alpha = (np.linalg.norm(normal_data, ord=2, axis=2, keepdims=True) > 0.5)
                alpha = np.tile(alpha, [1, 1, 3])
                alpha = Image.fromarray(np.array(alpha*255.0, dtype=np.byte), "RGB")
            else:
                alpha = None
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, normal=normal, alpha=alpha,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, background, eval, extension=".png", log=None, debug=False):
    log.info("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", background, extension, debug=debug)
    log.info("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", background, extension, debug=debug)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        log.info(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_img(path):
    if not "." in os.path.basename(path):
        files = glob.glob(path + '.*')
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        path = files[0]
    if path.endswith(".exr"):
        assert False
        if pyexr is not None:
            exr_file = pyexr.open(path)
            # print(exr_file.channels)
            all_data = exr_file.get()
            img = all_data[..., 0:3]
            if "A" in exr_file.channels:
                mask = np.clip(all_data[..., 3:4], 0, 1)
                img = img * mask
        else:
            img = imageio.imread(path)
            import pdb;
            pdb.set_trace()
        img = np.nan_to_num(img)
        hdr = True
    else:  # LDR image
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb_np(img[..., 0:3])
        hdr = False
    return img, hdr


def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)

    return np.ascontiguousarray(data)


def load_depth(tiff_path):
    return imageio.imread(tiff_path)


def load_mask(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask


def loadCamsFromScene(path, valid_list, background, debug):
    with open(f'{path}/sfm_scene.json') as f:
        sfm_scene = json.load(f)

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    bbox_transform = bbox_transform.copy()
    bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    bbox_inv = np.linalg.inv(bbox_transform)

    # meta info
    image_list = sfm_scene['image_path']['file_paths']

    # camera parameters
    train_cam_infos = []
    test_cam_infos = []
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        if debug and i >= 5: break
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1

            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            c2w = np.linalg.inv(extrinsic)
            c2w[:3, 3] = (c2w[:4, 3] @ bbox_inv.T)[:3]
            extrinsic = np.linalg.inv(c2w)

            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]

            focal_length_x = camera_info['camera']['intrinsic']['focal'][0]
            focal_length_y = camera_info['camera']['intrinsic']['focal'][1]
            ppx = camera_info['camera']['intrinsic']['ppt'][0]
            ppy = camera_info['camera']['intrinsic']['ppt'][1]

            image_path = os.path.join(path, image_list[index])
            image_name = Path(image_path).stem

            image, is_hdr = load_img(image_path)

            depth_path = os.path.join(path + "/depths/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".tiff"))

            if os.path.exists(depth_path):
                depth = load_depth(depth_path)
                depth *= bbox_inv[0, 0]
            else:
                print("No depth map for test view.")
                depth = None

            normal_path = os.path.join(path + "/normals/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".pfm"))
            if os.path.exists(normal_path):
                normal = load_pfm(normal_path)
            else:
                print("No normal map for test view.")
                normal = None

            mask_path = os.path.join(path + "/pmasks/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".png"))
            if os.path.exists(mask_path):
                img_mask = (imageio.imread(mask_path, pilmode='L') > 0.1).astype(np.float32)
                # if pmask is available, mask the image for PSNR
                image *= img_mask[..., np.newaxis]
            else:
                img_mask = np.ones_like(image[:, :, 0])

            fovx = focal2fov(focal_length_x, image.shape[1])
            fovy = focal2fov(focal_length_y, image.shape[0])
            if int(index) in valid_list:
                image *= img_mask[..., np.newaxis]
                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
                alpha = Image.fromarray(np.array(np.tile(img_mask[..., np.newaxis],(1, 1, 3))*255.0, dtype=np.byte), "RGB")
                if normal is not None:
                    normal = Image.fromarray(np.array((normal+1)/2*255.0, dtype=np.byte), "RGB")
                test_cam_infos.append(CameraInfo(
                    uid=index, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                    image_path=image_path, image_name=image_name,
                    alpha=alpha, normal=normal, depth=depth,
                    width=image.size[0], height=image.size[1]))
            else:
                image *= img_mask[..., np.newaxis]
                depth *= img_mask
                normal *= img_mask[..., np.newaxis]
                image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")
                alpha = Image.fromarray(np.array(np.tile(img_mask[..., np.newaxis],(1, 1, 3))*255.0, dtype=np.byte), "RGB")
                if normal is not None:
                    normal = Image.fromarray(np.array((normal+1)/2*255.0, dtype=np.byte), "RGB")
                train_cam_infos.append(CameraInfo(
                    uid=index, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                    image_path=image_path, image_name=image_name,
                    alpha=alpha, normal=normal, depth=depth,
                    width=image.size[0], height=image.size[1]))
        

    return train_cam_infos, test_cam_infos, bbox_transform


def readNeILFInfo(path, background, eval, log=None, debug=False):
    validation_indexes = []
    if eval:
        if "dtu" in path.lower():
            validation_indexes = [6, 13, 30, 35] # same as neuTex
        else:
            raise NotImplementedError

    train_cam_infos, test_cam_infos, bbx_trans = loadCamsFromScene(
        f'{path}/inputs', validation_indexes, background, debug)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = f'{path}/inputs/model/sparse_bbx_scale.ply'
    if not os.path.exists(ply_path):
        org_ply_path = f'{path}/inputs/model/sparse.ply'

        # scale sparse.ply
        pcd = fetchPly(org_ply_path)
        inv_scale_mat = np.linalg.inv(bbx_trans)  # [4, 4]
        points = pcd.points
        xyz = (np.concatenate([points, np.ones_like(points[:, :1])], axis=-1) @ inv_scale_mat.T)[:, :3]
        normals = pcd.normals
        colors = pcd.colors

        storePly(ply_path, xyz, colors * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "NeILF": readNeILFInfo,
}