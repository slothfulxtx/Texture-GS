import os
import random
import json
from .dataset_readers import sceneLoadTypeCallbacks, SceneInfo
from utils.cameras import cameraList_from_camInfos, camera_to_JSON


class Scene:

    scene_info: SceneInfo

    def __init__(self, cfg, log, work_dir, debug=False):

        self.cfg = cfg
        self.log = log

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(cfg.data_root_dir, "sparse")):
            log.info("Found colmap folder, assuming Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](cfg.data_root_dir, cfg.image_path, cfg.eval, log=log, debug=debug)
        elif os.path.exists(os.path.join(cfg.data_root_dir, "transforms_train.json")):
            log.info("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](cfg.data_root_dir, cfg.background, cfg.eval, log=log, debug=debug)
        elif os.path.exists(os.path.join(cfg.data_root_dir, "inputs/sfm_scene.json")):
            print("Found sfm_scene.json file, assuming NeILF data set!")
            scene_info = sceneLoadTypeCallbacks["NeILF"](cfg.data_root_dir, cfg.background, cfg.eval, log=log, debug=debug)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info = scene_info

        if not debug and cfg.save_init_pcd:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(work_dir, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
        
        if not debug and cfg.save_cameras:
            def save_cameras(cameras, filename):
                json_cams = []
                camlist = cameras
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(work_dir, filename), 'w') as file:
                    json.dump(json_cams, file)
            
            all_cameras = []
            if scene_info.test_cameras:
                save_cameras(scene_info.test_cameras, "test_cameras.json")
                all_cameras.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                save_cameras(scene_info.train_cameras, "train_cameras.json")
                all_cameras.extend(scene_info.train_cameras)
            save_cameras(all_cameras, "cameras.json")

        if cfg.shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in cfg.resolution_scales:
            log.info("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, cfg)
            log.info("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, cfg)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]