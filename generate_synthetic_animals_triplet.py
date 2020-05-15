#!/usr/bin/env python
# coding: utf-8

import os, time, glob, itertools, random, imageio, re
import numpy as np
from tqdm import tqdm, trange
from unrealcv.util import read_png, read_npy

import unrealdb as udb
import unrealdb.asset
import unrealdb.asset.animal

from unrealdb import d3
from PIL import Image
import pdb
import argparse
import copy


def make_filename(num_img, mesh, anim, time, dist, az, el):
    def get_mesh_name(mesh_path):
        re_mesh = re.compile("SkeletalMesh'.*/(.*)\.(.*)'")
        match = re_mesh.match(mesh_path)
        return match.group(1)

    def get_anim_name(anim_path):
        re_anim = re.compile("AnimSequence'.*/(.*)\.(.*)'")
        match = re_anim.match(anim_path)
        return match.group(1)

    mesh_name = get_mesh_name(mesh)
    anim_name = get_anim_name(anim)
    template = '{num_img}_{mesh_name}_{anim_name}_{time:.2f}_{dist:.2f}_{az:.2f}_{el:.2f}.png'
    filename = template.format(**locals())
    return filename


def glob_images(image_folder):
    filenames = glob.glob(os.path.join(image_folder, '*.jpg'))
    filenames += glob.glob(os.path.join(image_folder, '*.png'))
    filenames += glob.glob(os.path.join(image_folder, '*.jpeg'))
    return filenames


def load_render_params(global_animal):
    if global_animal == 'tiger':
        opt = dict(
            mesh=[udb.asset.MESH_TIGER],
            anim=udb.asset.tiger_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[150, 200, 250],  # , 300, 350],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30]  # , 150, 160, 170, 180]
        )

    elif global_animal == 'horse':
        opt = dict(
            mesh=[udb.asset.MESH_HORSE],
            anim=udb.asset.horse_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[250, 300, 350, 400],  # , 450],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30]  # , 150, 160, 170, 180]
        )

    elif global_animal == 'domestic_sheep':
        # extra treatment with shifting camera
        opt = dict(
            mesh=[udb.asset.MESH_DOMESTIC_SHEEP],
            anim=udb.asset.domestic_sheep_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[150, 200, 250],  # , 300, 350],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30]  # , 150],#, 160, 170, 180]
        )

    elif global_animal == 'hellenic_hound':
        # extra treatment with shifting camera
        opt = dict(
            mesh=[udb.asset.MESH_HELLENIC_HOUND],
            anim=udb.asset.hellenic_hound_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[75, 100, 125],  # 200, 250],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30, 150, 160, 170, 180]
        )

    #    elif global_animal=='elephant':
    #        opt = dict(
    #            mesh = [udb.asset.MESH_ELEPHANT],
    #            anim = udb.asset.elephant_animations,
    #            ratio = np.arange(0.1, 0.9, 0.05),
    #            # ratio = [0],
    #            dist = [350, 400, 450, 500, 550],
    #            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
    #                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
    #            el = [0, 10, 20, 30, 150, 160, 170, 180]
    #        )
    #
    #    elif global_animal=='bat':
    #        opt = dict(
    #            mesh = [udb.asset.MESH_BAT],
    #            anim = udb.asset.bat_animations,
    #            ratio = np.arange(0.1, 0.9, 0.05),
    #            # ratio = [0],
    #            dist = [75, 100, 125, 150, 200],
    #            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
    #                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
    #            el = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    #        )
    elif global_animal == 'cat':
        # extra treatment with shifting camera
        opt = dict(
            mesh=[udb.asset.MESH_CAT],
            anim=udb.asset.cat_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[60, 90, 125],  # 200, 250],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30, 150, 160, 170, 180]
        )
    elif global_animal == 'scotland_cattle':
        # extra treatment with shifting camera
        opt = dict(
            mesh=[udb.asset.MESH_SCOTTLAND_CATTLE],
            anim=udb.asset.scotland_cattle_animations,
            ratio=np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist=[200, 250, 300],
            az=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el=[0, 10, 20, 30, 150, 160, 170, 180]
        )
    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
                                      opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params


def set_camera_params(cam_loc, cam_rot):
    udb.client.request('vset /camera/1/location {} {} {}'.format(*cam_loc))
    udb.client.request('vset /camera/1/rotation {} {} {}'.format(*cam_rot))


def shift_camera_animal(animal):
    cam_loc, cam_rot = get_camera_params()

    if animal == "hellenic_hound":
        cam_rot = [float(item) for item in cam_rot.split(' ')]
        cam_loc = [float(item) for item in cam_loc.split(' ')]
        # camera must be little bit lower
        cam_loc[2] -= 40
    elif animal == "cat":
        cam_rot = [float(item) for item in cam_rot.split(' ')]
        cam_loc = [float(item) for item in cam_loc.split(' ')]
        # camera must be little bit lower
        cam_loc[2] -= 50
    elif animal == "scotland_cattle":
        cam_rot = [float(item) for item in cam_rot.split(' ')]
        cam_loc = [float(item) for item in cam_loc.split(' ')]
        # camera must be little bit lower
        cam_loc[2] += 20
    elif animal == "domestic_sheep":
        cam_rot = [float(item) for item in cam_rot.split(' ')]
        cam_loc = [float(item) for item in cam_loc.split(' ')]
        # camera must be little bit lower
        cam_loc[2] -= 10

    set_camera_params(cam_loc, cam_rot)


def parse_kpts(filename, offset):
    res = udb.client.request('vget /animal/tiger/vertex obj'.format(**locals()))
    data = res.strip().split('\n')
    # kpts_3d_array = np.zeros((3299, 3))
    kpts_3d_array = np.zeros((5000, 3))
    for i, line in enumerate(data):
        _, x, y, z = line.split(' ')
        kpts_3d_array[i] = float(x), float(y), float(z) + offset - 80
    return kpts_3d_array


def get_camera_params():
    cam_loc = udb.client.request('vget /camera/1/location'.format(**locals()))
    cam_rot = udb.client.request('vget /camera/1/rotation'.format(**locals()))
    return cam_loc, cam_rot


def transform_kpts(cam_loc, cam_rot, kpts_3d, depth_img):
    # x, y, z = # Get camera location in world coordinate
    # pitch, yaw, roll, # camera rotation
    # width, height =  # image width 
    # f  = width / 2 # FOV = 90
    width = 640
    height = 480
    x, y, z = cam_loc
    pitch, yaw, roll = cam_rot
    cam_pose = d3.CameraPose(x, y, z, pitch, yaw, roll, width, height, width / 2)

    # points_2d = cam_pose.project_to_2d(points_3d)  # list of 2d point, x, y
    points_3d_cam = cam_pose.project_to_cam_space(kpts_3d)
    # x, y, z # z means distance to image plane.
    depth = depth_img  # Get depth image from the simulator. w x h x 1 float array.
    epsilon = 15
    kpts_2d = points_3d_cam[:, :2]
    kpts_z = points_3d_cam[:, 2]

    vis = np.zeros((kpts_3d.shape[0], 1))
    for i, (x, y, z) in enumerate(points_3d_cam):
        x = int(x)
        y = int(y)
        if y < 0 or y >= 480 or x < 0 or x >= 640:
            vis[i] = 0
        else:
            real_z = depth[y][x]
            if abs(real_z - z) < epsilon:
                # print(abs(real_z - z))
                vis[i] = 1
            else:
                # print(abs(real_z - z))
                vis[i] = 0

                # points_3d = # read 3D keypoint from AnimalParsing
    kpts = np.hstack((kpts_2d, vis))
    return kpts, kpts_z


def retrieve(animal, num_images, use_random_texture):
    udb.connect('localhost', 9900)

    # reset the program
    map_name = 'AnimalDataCapture'
    udb.client.request('vset /action/game/level {map_name}'.format(**locals()))
    udb.client.request('vset /camera/0/location 500 0 300')
    udb.client.request('vset /camera/0/rotation -20 180 0')

    random_texture_path = "val2017"
    # this path needs to be on the server!!
    val2017_dir = "/export/home/ffeldman/git/Learning-from-Synthetic-Animals/data_generation/" + random_texture_path  # os.path.abspath(random_texture_path)
    beautiful_textures = "/export/home/ffeldman/git/Learning-from-Synthetic-Animals/data_generation/texture_images/"
    bg_path_list = glob_images(val2017_dir)
    texture_path_list = glob_images(val2017_dir)
    beautiful_textures_path_list = glob_images(beautiful_textures)

    output_path = f"synthetic_animals_triplet/{animal}/"
    global_animal = animal

    render_params = load_render_params(global_animal)
    random.shuffle(render_params)
    obj_id = 'tiger'
    animal = udb.CvAnimal(obj_id)
    animal.spawn()

    # acquire offset
    obj_loc = udb.client.request('vget /object/tiger/location')
    obj_loc = [float(v) for v in obj_loc.split(' ')]
    offset = obj_loc[2]

    r, g, b = 155, 168, 157
    animal.set_mask_color(r, g, b)
    if global_animal == 'tiger':
        animal.set_mesh(udb.asset.MESH_TIGER)
    elif global_animal == 'horse':
        animal.set_mesh(udb.asset.MESH_HORSE)
    elif global_animal == 'domestic_sheep':
        animal.set_mesh(udb.asset.MESH_DOMESTIC_SHEEP)
    elif global_animal == 'hellenic_hound':  # Dog
        animal.set_mesh(udb.asset.MESH_HELLENIC_HOUND)
    elif global_animal == 'elephant':
        animal.set_mesh(udb.asset.MESH_ELEPHANT)
    # from here todo!
    elif global_animal == 'cat':
        animal.set_mesh(udb.asset.MESH_CAT)
    # elif global_animal=='zebra':
    #    animal.set_mesh(udb.asset.MESH_CAT)
    # elif global_animal=='celtic_wolfhound': # Dog
    #    animal.set_mesh(udb.asset.MESH_CAT)
    # elif global_animal=='pug': # mops -> dog
    #    animal.set_mesh(udb.asset.MESH_CAT)
    # elif global_animal=='cane_corso': # a dog
    #    animal.set_mesh(udb.asset.MESH_CAT)
    elif global_animal == 'scotland_cattle':  # a scottish cow
        animal.set_mesh(udb.asset.MESH_SCOTTLAND_CATTLE)
    # elif global_animal=='longhorn_cattle': # a cow
    #    animal.set_mesh(udb.asset.MESH_CAT)
    # elif global_animal=='longhorn_cattle_v2': # a cow
    #    animal.set_mesh(udb.asset.MESH_CAT)

    env = udb.CvEnv()

    output_dir = output_path
    if not os.path.isdir(output_dir): os.makedirs(output_dir)

    # masked_frames = []
    # whitened_frames = []
    # frame_names = []
    # extracted_kpts = []

    p0a0_frame_names = []
    p0a1_frame_names = []
    p1a1_frame_names = []
    p0a0_extracted_kpts = []
    p0a1_extracted_kpts = []
    p1a1_extracted_kpts = []
    p0a0_list_whitened = []
    p0a1_list_whitened = []
    p1a1_list_whitened = []
    p0a0_list_masked = []
    p0a1_list_masked = []
    p1a1_list_masked = []

    img_idx = 0
    sky_texture = "/export/home/ffeldman/Masterarbeit/data/white.jpg"  # random.choice(bg_path_list)
    floor_texture = "/export/home/ffeldman/Masterarbeit/data/white.jpg"  # random.choice(bg_path_list)
    # random.choice(texture_path_list)
    # process_params = random.choices(render_params, k=num_images)
    random.shuffle(render_params)
    for i, param in enumerate(tqdm(render_params)):
        random_animal_texture = random.randint(0, len(beautiful_textures_path_list) - 1)
        animal_texture = beautiful_textures_path_list[random_animal_texture]
        mesh, anim, ratio, dist, az, el = param
        filename = make_filename(img_idx, mesh, anim, ratio, dist, az, el)

        p0a0, p0a1, p1a1 = False, False, False
        p0a0_tried = False
        goto_p1a1 = False

        def check_triplet():
            return p0a0 and p0a1 and p1a1

        # Update the scene
        env.set_random_light()
        break_while = False
        print("Here before while.")
        while not check_triplet():
            print("Image idx:", img_idx)
            for triplet in ["p0a0", "p0a1", "p1a1"]:
                print(triplet, p0a0, p0a1, p1a1)
                if triplet == "p0a0":
                    if p0a0_tried and p0a0:
                        goto_p1a1 = True
                        continue
                    p0a0_tried = True
                elif triplet == "p0a1":
                    if (p0a0_tried and not p0a0):
                        # p0a0 was false so p0a1 will be false as well
                        # we set all of them true to break the while loop
                        p0a0, p0a1, p1a1 = True, True, True
                        break_while = True
                        print("Breaking the loop.")
                        break
                    if goto_p1a1:
                        continue
                    # update the appearance but leave the pose as is
                    random_texture = random_animal_texture
                    while random_animal_texture == random_texture:
                        random_texture = random.randint(0, len(beautiful_textures_path_list) - 1)
                        animal_texture = beautiful_textures_path_list[random_texture]
                    animal.set_texture(animal_texture)
                elif triplet == "p1a1":
                    if break_while:
                        break
                    # update the pose but leave the appearance as is
                    # print("Setting new pose.")
                    param = random.choice(render_params)
                    mesh, anim, ratio, dist, az, el = param

                env.set_floor(floor_texture)
                env.set_sky(sky_texture)

                animal.set_animation(anim, ratio)

                # Capture data
                animal.set_tracking_camera(dist, az, el)
                shift_camera_animal(global_animal)

                img = animal.get_img()
                seg = animal.get_seg()
                depth = animal.get_depth()
                mask = udb.get_mask(seg, [r, g, b])

                # get kpts
                ## get cam_loc and cam_rot
                cam_loc, cam_rot = get_camera_params()
                cam_loc = [float(item) for item in cam_loc.split(' ')]
                cam_rot = [float(item) for item in cam_rot.split(' ')]

                ## transform keypoints
                kp_3d_array = parse_kpts(filename, offset)
                kpts, kpts_z = transform_kpts(cam_loc, cam_rot, kp_3d_array, depth)

                ## transform images and kpts
                img = Image.fromarray(img[:, :, :3])
                seg_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                seg_mask[mask == False] = 0  # tiger/horse
                seg_mask[mask == True] = 255  # tiger/horse

                # # save imgs
                if global_animal == 'tiger':
                    kp_18_id = [2679, 2753, 2032, 1451, 1287, 3085, 1632, 229, 1441, 1280, 2201, 1662, 266, 158,
                                270,
                                152,
                                219, 129]
                elif global_animal == 'horse':
                    kp_18_id = [1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564, 993, 726, 1585, 1556, 427,
                                1548,
                                967, 877]
                elif global_animal == 'domestic_sheep':
                    kp_18_id = [2046, 1944, 1267, 1875, 1900, 1868, 1894, 687, 173, 1829, 1422, 821, 624, 580, 622,
                                575,
                                1370, 716]
                elif global_animal == 'hellenic_hound':
                    kp_18_id = [2028, 2580, 912, 878, 977, 1541, 1734, 480, 799, 1575, 1446, 602, 755, 673, 780,
                                1580,
                                466,
                                631]
                elif global_animal == 'elephant':
                    kp_18_id = [1980, 2051, 1734, 2122, 2155, 2070, 2166, 681, 923, 1442, 1041, 1528, 78, 599, 25,
                                595,
                                171,
                                570]
                else:
                    print("WARNING THIS ANIMAL HAS NO CORRECT KEYPOINTS YET - DO NOT USE!!")
                    kp_18_id = [2028, 2580, 912, 878, 977, 1541, 1734, 480, 799, 1575, 1446, 602, 755, 673, 780,
                                1580,
                                466,
                                631]
                # print(triplet, sum(kpts[kp_18_id, 2]))
                if sum(kpts[kp_18_id, 2]) >= 4:

                    arr = kpts[kp_18_id]
                    # set non visible points to zero
                    arr[arr[:, 2] == 0] = [0, 0, 0]
                    arr = arr[:, :2]
                    # create output folder for images e.g. synthetic_animals/{animal}/{video}
                    sequence_output_dir = output_dir
                    sequence_dir_filename = os.path.join(sequence_output_dir,
                                                         filename.replace(".png", f"_{triplet}.png"))
                    filename_mask = filename.replace(".png", f"_mask_{triplet}.png")
                    filename_mask_whitened = filename.replace(".png", f"_mask_white_{triplet}.png")
                    sequence_dir_filename_mask = os.path.join(sequence_output_dir, filename_mask)
                    sequence_dir_filename_mask_whitened = os.path.join(sequence_output_dir, filename_mask_whitened)
                    if not os.path.isdir(sequence_output_dir): os.makedirs(sequence_output_dir)
                    whitened_img = np.array(copy.deepcopy(img))
                    whitened_img[~mask] = 255
                    imageio.imwrite(sequence_dir_filename_mask, seg_mask)
                    imageio.imwrite(sequence_dir_filename_mask_whitened, whitened_img)
                    imageio.imwrite(sequence_dir_filename, img)

                    if triplet == "p0a0":
                        p0a0 = True
                        p0a0_list_whitened.append(os.path.join(sequence_output_dir, filename_mask_whitened))
                        p0a0_list_masked.append(os.path.join(sequence_output_dir, filename_mask))
                        p0a0_frame_names.append(sequence_dir_filename)
                        p0a0_extracted_kpts.append(arr)
                    if triplet == "p0a1":
                        p0a1 = True
                        p0a1_list_whitened.append(os.path.join(sequence_output_dir, filename_mask_whitened))
                        p0a1_list_masked.append(os.path.join(sequence_output_dir, filename_mask))
                        p0a1_frame_names.append(sequence_dir_filename)
                        p0a1_extracted_kpts.append(arr)
                    if triplet == "p1a1":
                        p1a1 = True
                        p1a1_list_whitened.append(os.path.join(sequence_output_dir, filename_mask_whitened))
                        p1a1_list_masked.append(os.path.join(sequence_output_dir, filename_mask))
                        p1a1_frame_names.append(sequence_dir_filename)
                        p1a1_extracted_kpts.append(arr)
                        img_idx += 1
                    if img_idx == num_images:
                        # assert len(p0a0_list_whitened) == len(p0a0_list_masked) == len(p0a0_frame_names) == len(
                        #    p0a0_extracted_kpts)
                        return p0a0_list_whitened, p0a0_list_masked, p0a0_frame_names, \
                               np.array(p0a0_extracted_kpts), p0a1_list_whitened, p0a1_list_masked, \
                               p0a1_frame_names, np.array(p0a1_extracted_kpts), p1a1_list_whitened, p1a1_list_masked, \
                               p1a1_frame_names, np.array(p1a1_extracted_kpts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic Animal Dataset Generation')
    """
    We want to create a triplet of animals containing the following:
    A_x = Appearance
    P_x = Pose
    
    Image_1: P_0A_0
    Image_2: P_1A_1
    Image_3: P_0A_1
    """

    # horse, tiger, , elephant, cat, scotland_cattle

    parser.add_argument('--animal', default='horse', type=str,
                        help='horse | tiger | domestic_sheep | hellenic_hound | cat | elephant')
    parser.add_argument('--num-imgs', default=100, type=int,
                        help='Number of images in the sequence (default: 100, to gen GT)')
    parser.add_argument('--random-texture-path', default='./data_generation/val2017', type=str,
                        help='coco val 2017')
    parser.add_argument('--use-random-texture', action='store_true', default=True,
                        help='whether use random texture for the animal or not')
    args = parser.parse_args()

    mydict = dict()

    p0a0_list_whitened, p0a0_list_masked, p0a0_frame_names, p0a0_extracted_kpts, p0a1_list_whitened, \
    p0a1_list_masked, p0a1_frame_names, p0a1_extracted_kpts, p1a1_list_whitened, p1a1_list_masked, \
    p1a1_frame_names, p1a1_extracted_kpts = retrieve(animal=args.animal, num_images=args.num_imgs,
                                                     use_random_texture=args.use_random_texture)
    mydict["p0a0"] = (p0a0_list_whitened, p0a0_list_masked, p0a0_frame_names, p0a0_extracted_kpts)
    mydict["p0a1"] = (p0a1_list_whitened, p0a1_list_masked, p0a1_frame_names, p0a1_extracted_kpts)
    mydict["p1a1"] = (p1a1_list_whitened, p1a1_list_masked, p1a1_frame_names, p1a1_extracted_kpts)
    # Save as Metadatset
    META_DIR = f"synthetic_animals_triplet/{args.animal}s_meta/"
    META_LABEL_DIR = META_DIR + "labels/"
    if not os.path.isdir(META_DIR): os.makedirs(META_DIR)
    if not os.path.isdir(META_LABEL_DIR): os.makedirs(META_LABEL_DIR)

    from edflow.data.believers import meta_util
    import yaml

    dict_file = {
        'description': f"Synthetic animals triplet - {args.animal}",
        # 'loaders': {
        #    'frames': 'image'},
        'loader_kwargs': {
            # 'frames': {
            #    'support': "0->1"
            # },
        },
    }

    for key, triplet in mydict.items():
        meta_util.store_label_mmap(np.array(triplet[0]), META_LABEL_DIR, f"{key}_whitened_frames:image")
        meta_util.store_label_mmap(np.array(triplet[1]), META_LABEL_DIR, f"{key}_masked_frames:image")
        meta_util.store_label_mmap(np.array(triplet[2]), META_LABEL_DIR, f"{key}_frames:image")
        meta_util.store_label_mmap(np.array(triplet[3]), META_LABEL_DIR, f"{key}_kps")
        dict_file["loader_kwargs"][f"{key}_whitened_frames:image"] = {'support': "0->1"}
        dict_file["loader_kwargs"][f"{key}_masked_frames:image"] = {'support': "0->1"}
        dict_file["loader_kwargs"][f"{key}_frames:image"] = {'support': "0->1"}

    with open(f"{META_DIR}meta.yaml", 'w') as file:
        documents = yaml.dump(dict_file, file)

    # meta_util.store_label_mmap(np.array(p0a0_list_whitened), META_LABEL_DIR, "p0a0_list_whitened:image")
    # meta_util.store_label_mmap(np.array(masked_frames), META_LABEL_DIR, "masked_frames:image")
    # meta_util.store_label_mmap(np.array(whitened_frames), META_LABEL_DIR, "whitened_frames:image")
    # meta_util.store_label_mmap(np.array(fid), META_LABEL_DIR, "fid")
    # meta_util.store_label_mmap(np.array(extracted_kpts), META_LABEL_DIR, "kps")
