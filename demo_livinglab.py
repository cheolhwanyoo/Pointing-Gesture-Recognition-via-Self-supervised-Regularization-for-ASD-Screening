import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import pyk4a
import copy
import torch.nn as nn
import timm.models
import operator
import torchvision.transforms as transforms
import mxnet as mx

from argparse import ArgumentParser
from lib_openpose.network.rtpose_vgg import get_model
from lib_openpose.config import cfg
from lib_openpose.evaluate.coco_eval import get_outputs_openpose
from lib_openpose.utils.common import CocoPart, draw_humans
from lib_openpose.utils.paf_to_pose import paf_to_pose_cpp
from lib_retinaface.inference import get_outputs_retina
from lib_retinaface import cfg_mnet, cfg_re50
from lib_retinaface.inference import load_model
from pyk4a import Config, PyK4A
from pyk4a import PyK4APlayback
from tqdm import tqdm
from utility import *
from model.resnet import Resnet50_fc, Resnet50_Siam, Resnet50_Barlow
from BYOL.byol_pytorch import BYOL
from model.transformer import vit_Siam, vit_Barlow
from model.retinaface import RetinaFace
from lib_gradcam import GradCAM
from lib_gradcam.utils.image import show_cam_on_image
from torchvision import models
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from numpy import linalg as LA
from PIL import Image
from pathlib import Path
from model.model import build_net


eps = 1e-7
IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])
pointing_label = ['No pointing', 'Pointing']
font_color = [(0, 0, 255), (0, 255, 0)]
m = nn.Softmax(dim=1)
monitor_mm = (695.04, 390.96)
monitor_pixel = (3840, 2160)
mm_per_pixel = (monitor_mm[0] / monitor_pixel[0], monitor_mm[1] / monitor_pixel[1])

tf_Resize = transforms.Resize((224, 224))
tf_ToTensor = transforms.ToTensor()
tf_Normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    # Retina face parameters
    parser.add_argument('-m', '--trained_model', default='checkpoints/mobilenet0.25_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.4, type=float, help='visualization_threshold')  #0.5
    parser.add_argument('--scale_img', default=0.3, type=float, help='scale image for faster inference speed')

    # hand detection
    parser.add_argument('--det_config', help='Config file for detection',
                        default='lib_detecthand/cascade_rcnn_x101_64x4d_fpn_1class.py')
    parser.add_argument('--det_checkpoint',
                        default='checkpoint/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth')

    # hand pose estimation
    parser.add_argument('--pose_config',
                        default='configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/mobilenetv2_coco_wholebody_hand_256x256.py')
    parser.add_argument('--pose_checkpoint',
                        default='checkpoints/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909.pth')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt_thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    # grad cam
    parser.add_argument('--aug_smooth', action='store_true', default=False,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true', default=False,
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    ## visualize
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')

    parser.add_argument(
        '--saveVideo',
        type=bool,
        default=False)

    parser.add_argument(
        '--viz_skeleton',
        default=False,
        help='whether to visualize skeleton.')

    parser.add_argument(
        '--mosaic',
        default=False,
        help='whether to process mosaicing.')
    parser.add_argument(
        '--vis_grad_cam',
        default=False,
        help='whether to use gradCAM')

    parser.add_argument(
        '--vis_pointing_ray',
        default=False,
        help='whether to show pointing ray')

    parser.add_argument(
        '--demo_atten_point',
        default='space',
        type=str,
        choices=['space', 'monitor'])

    parser.add_argument(
        '--input',
        type=str,
        default='mkv',
        help='live or saved video',
        choices=['mkv', 'kinect', 'webcam', 'mkv_frames'])

    # User parameters (thresholds)
    parser.add_argument('--aggre_results', type=str, default='hand_only', choices=['and_op', 'average', 'hand_only'])
    parser.add_argument('--k_val', type=float, default=1.5)  #1.5
    parser.add_argument('--box_half_len', type=int, default=70)  #어른: 100 #애기: 70
    parser.add_argument('--positive_persist_thres', type=int, default=2)  # 5
    parser.add_argument('--dist_thres', type=int, default=2000)  #1.4m(콘텐츠 검사)
    parser.add_argument('--detect_hand', type=str, default='openpose', choices=['rcnn', 'openpose'])
    parser.add_argument('--select_person', type=str, default='min_bone', choices=['min_bone', 'nearest'])
    parser.add_argument('--showFps', type=bool, default=False)
    parser.add_argument('--viz_scale', type=int, default=1.0)

    parser.add_argument('--model_name_arm', type=str, default='0504_arm_resnet50_SGD_whole_testDB')
    parser.add_argument('--model_name_hand', type=str, default='0929_resnet50_ntu')
    parser.add_argument('--test_name', type=str, default=None)
    parser.add_argument('--SSL', type=str, default='None', choices=['None', 'SimSiam', 'BYOL', 'Barlow Twins', 'MoCov3'])  # 50
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'vit_B_32', 'vit_B_16', 'vit_S_16', 'vit_T_16', 'vit_hybrid_T_16'])  # 50

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    ## video start
    if args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        input_list = ['temp']
        labels = [0]

    elif args.input == 'kinect':
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True, )
        )
        k4a.start()

        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        k4a.whitebalance = 4510
        assert k4a.whitebalance == 4510

        # webcamera configuration
        camera_param = {
            'fx': 601.7,
            'fy': 601.6,
            'cx': 639.6,
            'cy': 366.2
        }
        fx = camera_param['fx']
        fy = camera_param['fy']
        cx = camera_param['cx']
        cy = camera_param['cy']

        BOX_HALF_LEN = args.box_half_len
        k_val = args.k_val

        input_list = ['temp']
        labels = [0]

        if args.demo_atten_point == 'monitor':
            img_monitor = np.zeros((monitor_pixel[1], monitor_pixel[0], 3))

            for i in range(1, 5, 2):
                for j in range(1, 5, 2):
                    y_pt = int(monitor_pixel[1] / 4.0 * i)
                    x_pt = int(monitor_pixel[0] / 4.0 * j)
                    img_monitor = cv2.circle(img_monitor, (x_pt, y_pt), 8, (255, 255, 255), -1)
                    img_monitor = cv2.circle(img_monitor, (x_pt, y_pt), 130, (255, 255, 255), 2)

    elif args.input == 'mkv' or args.input == 'mkv_frames':
        base_dir_pos = '/home/ych/data/living_lab_db/contents/pointing_positive_final/'
        folder_list_pos = os.listdir(base_dir_pos)
        input_list_pos = []
        labels_pos = []
        labels = []

        for i in range(len(folder_list_pos)):
            file_dir = base_dir_pos + folder_list_pos[i] + '/08/rec/'

            input_list_pos += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                               f.split(".")[-1] == "mkv"]
        labels_pos = [0] * len(input_list_pos)

        base_dir_neg = '/home/ych/data/living_lab_db/contents/pointing_negative_final/'
        folder_list_neg = os.listdir(base_dir_neg)
        input_list_neg = []
        labels_neg = []

        for i in range(len(folder_list_neg)):
            file_dir = base_dir_neg + folder_list_neg[i] + '/08/rec/'
            input_list_neg += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                               f.split(".")[-1] == "mkv"]
        labels_neg = [1] * len(input_list_neg)

        input_list = input_list_pos + input_list_neg
        labels = labels_pos + labels_neg

        # camera configuration
        camera_param = {
            'fx': 964.9,
            'fy': 963.6,
            'cx': 1024.4,
            'cy': 779.7
        }

        fx = camera_param['fx']
        fy = camera_param['fy']
        cx = camera_param['cx']
        cy = camera_param['cy']

        BOX_HALF_LEN = args.box_half_len
        k_val = args.k_val

    # Load OpenPose
    model_oepnpose = get_model('vgg19')
    model_oepnpose.load_state_dict(torch.load('checkpoints/pose_model.pth'))
    model_oepnpose.cuda()
    model_oepnpose.float()
    model_oepnpose.eval()
    print('loading openpose done...')

    ## Load pointing classifier model ##
    ## model(armNet) ##
    model_PointDetNet_arm = models.resnet50(pretrained=True)
    model_PointDetNet_arm.fc = torch.nn.Linear(2048, 2)

    ## load checkpoint
    checkpoint = torch.load('checkpoints/logs/' + args.model_name_arm + '/model_28.checkpoint')
    model_PointDetNet_arm.load_state_dict(checkpoint)
    model_PointDetNet_arm.cuda()
    model_PointDetNet_arm.eval()
    print('loading PointDetNet(arm) done...')

    ## model(handNet) ##
    model_PointDetNet_hand = build_net(args)

    if args.SSL == 'None':
        #checkpoint = torch.load('checkpoints/logs/' + args.model_name_hand + '/model_22.checkpoint', map_location='cuda:0')
        checkpoint = torch.load('checkpoints/logs/' + args.model_name_hand + '/model_best.checkpoint', map_location='cuda:0')
        model_PointDetNet_hand.load_state_dict(checkpoint)

    else:
        #state_dict = torch.load('checkpoints/logs/' + args.model_name_hand + '/model_50.checkpoint', map_location='cuda:0')
        state_dict = torch.load('checkpoints/logs/' + args.model_name_hand + '/model_best.checkpoint', map_location='cuda:0')
        ##
        model_dict = model_PointDetNet_hand.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)

        model_PointDetNet_hand.load_state_dict(model_dict)

    model_PointDetNet_hand.cuda()
    model_PointDetNet_hand.eval()
    print('loading PointDetNet(hand) done...')


    # model_PointDetNet_hand = vit_Siam(2)

    ########### ONNX  ###########
    #export_onnx_handnet(model_PointDetNet_hand)

    ## Load hand pose estimator
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    print('loading hand pose estimator(mobilenetv2) done...')

    ## gradCAM_
    #target_layers = [model_PointDetNet_arm.layer4[-1]]
    vis_grad_cam = args.vis_grad_cam

    ## Load RetinaFace
    cfg_retina = cfg_mnet
    # net and model
    model_retina = RetinaFace(cfg=cfg_retina, phase='test')
    model_retina = load_model(model_retina, args.trained_model, False)
    model_retina = model_retina.cuda()
    model_retina.eval()
    print('loading retina-face done...')

    running_corrects = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ndata = 0
    video_num_cnt = 0

    if args.test_name == None:
        args.test_name = args.model_name_hand
        txt_dir = args.model_name_hand + '/Results_test.txt'
    else:
        txt_dir = 'Results_parsing_DB_final_' + args.test_name + '.txt'

    video_dir = 'result_video/saved_video_final_' + args.test_name
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    video_cnt = 0

    for idx_list in tqdm(range(len(input_list))):

        text_color_arm = (0, 0, 255)
        text_color_finger = (0, 0, 255)
        text_color_pointing = (0, 0, 255)

        prob_hand_list_video = []
        pred_list_video = []
        prob_pointing = 0
        b_flag_pointing = False
        frame_count = 0
        pos_cnt = 0
        pred_buf = 0
        prob_hand = 0
        prob_arm = 0
        fr_idx = 0
        text_color = (0, 0, 255)

        if args.input == 'mkv':
            #playback = PyK4APlayback('/home/ych/data/linving_lab_db_aods_bedevel/AI-135-03/ados/D4_3-R_kinect7.mkv')
            playback = PyK4APlayback(input_list[idx_list])
            playback.open()

        if args.saveVideo == True:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_dir = video_dir + '/' + str(video_num_cnt) + '_result.mp4'
            out = cv2.VideoWriter(output_dir, fourcc, 15, (2048, 1536))

        if args.input == 'mkv_frames':

            folder_dir = input_list[idx_list].split(".mkv")[-2]

            color_dir = os.path.join(folder_dir, 'colors')
            depth_dir = os.path.join(folder_dir, 'depths')
            file_len = len(os.listdir(color_dir))

        while True:
            try:
                last_time = time.time()

                if args.input == 'webcam':
                    flag, img = cap.read()
                    if not flag:
                        break

                elif args.input == 'kinect':
                    capture = k4a.get_capture()
                    img = capture.color[:, :, :3]
                    depth = capture.transformed_depth

                elif args.input == 'mkv':
                    capture = playback.get_next_capture()

                    if capture.color is not None and capture.depth is not None:
                        #img = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                        img = mx.image.imdecode(capture.color, to_rgb=0).asnumpy()
                        depth = capture.transformed_depth
                        # colorized_depth = colorize(depth, (None, 5000))

                    else:
                        continue
                elif args.input == 'mkv_frames':

                   if fr_idx < file_len:

                        color_img = color_dir + '/' + str(fr_idx) + '.png'
                        depth_img = depth_dir + '/' + str(fr_idx) + '.png'

                        img = cv2.imread(color_img)
                        depth = Image.open(depth_img)
                        depth = np.array(depth)

                        fr_idx +=1
                   else:
                       break

                img_copy = copy.deepcopy(img)
                # img_copy2 = copy.deepcopy(img)

                center_pt3_2d = (0, 0)
                pt_rel_list = []

                with torch.no_grad():

                    if args.mosaic == True:
                        ## RetinaFace part
                        dets = get_outputs_retina(img, model_retina, cfg_mnet, args)

                        for b in dets:
                            if b[4] < args.vis_thres:
                                continue
                            text = "{:.4f}".format(b[4])
                            b = list(map(int, b))
                            b = [int(i / args.scale_img + 0.5) for i in b]  # re-scale image

                            face_img = img[max(b[1] - 50, 0):min(b[3] + 50, img.shape[0]),
                                       max(b[0] - 50, 0):min(b[2] + 50, img.shape[1]),
                                       :]  # 인식된 얼굴 이미지 crop (여분: 20 pixel)
                            h = face_img.shape[0]
                            w = face_img.shape[1]

                            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)  # 축소
                            face_img = cv2.resize(face_img, dsize=(w, h), interpolation=cv2.INTER_AREA)  # 확대

                            img[max(b[1] - 50, 0):min(b[3] + 50, img.shape[0]),
                            max(b[0] - 50, 0):min(b[2] + 50, img.shape[1]), :] = face_img  # 인식된 얼굴 영역 모자이크 처리

                    ## Openpose part
                    paf, heatmap, imscale = get_outputs_openpose(
                        img_copy, model_oepnpose, 'rtpose')

                    humans = paf_to_pose_cpp(heatmap, paf, cfg)

                    # generate detection result
                    image_h, image_w = img.shape[:2]
                    hand_bbox_results = []
                    arm_bbox_results = []
                    id_joint_list = []

                    dist_min = 1e20
                    baby_idx = None

                    # 어깨너비 min 값 택해서 애기 id take
                    if args.select_person == 'min_bone':
                        for i in range(len(humans)):
                            if 0 in humans[i].body_parts.keys() and 2 in humans[i].body_parts.keys() and 5 in humans[
                                i].body_parts.keys():

                                x_scale = int(humans[i].body_parts[2].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[2].y * image_h + 0.5)
                                pt_idx_0 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                x_scale = int(humans[i].body_parts[5].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[5].y * image_h + 0.5)
                                pt_idx_1 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                # 선생님은 주로 젤 멀리있으니까 제외시켜주자. 1.4m
                                if pt_idx_0[2] > args.dist_thres:
                                    continue

                                dist = cal_dist(pt_idx_0, pt_idx_1)

                                if dist <= dist_min:
                                    dist_min = dist
                                    baby_idx = i

                    elif args.select_person == 'nearest':
                        for i in range(len(humans)):
                            if 0 in humans[i].body_parts.keys():

                                x_scale = int(humans[i].body_parts[0].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[0].y * image_h + 0.5)
                                # pt_idx_0 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                dist = depth[y_scale, x_scale]

                                if dist == 0 or dist > args.dist_thres:
                                    continue

                                # # 선생님은 주로 젤 멀리있으니까 제외시켜주자. 1.4m
                                # if pt_idx_0[2] > args.dist_thres:
                                #     continue
                                # pt_idx_1 = (0, 0, 0)
                                #
                                # dist = cal_dist(pt_idx_0, pt_idx_1)

                                if dist <= dist_min:
                                    dist_min = dist
                                    baby_idx = i

                    img = np.ascontiguousarray(img, dtype=np.uint8)

                    for idx, human in enumerate(humans):
                        # draw point

                        if idx == baby_idx:  # 애기 id만 care

                            for i in range(CocoPart.Background.value):

                                if i not in humans[baby_idx].body_parts.keys():
                                    continue

                                # 왼쪽 오른쪽 손목 index
                                if i == 4 or i == 7:

                                    pre_idx = i - 1  # 팔꿈치 index

                                    if pre_idx in human.body_parts.keys():

                                        ## 눈 좌표
                                        if 14 in human.body_parts.keys() or 15 in human.body_parts.keys():

                                            if 14 in human.body_parts.keys() and 15 in human.body_parts.keys():

                                                ref_center_pt_2d = (int((human.body_parts[14].x + human.body_parts[
                                                    15].x) / 2.0 * image_w + 0.5),
                                                                    int((human.body_parts[14].y + human.body_parts[
                                                                        15].y) / 2.0 * image_h + 0.5))
                                            elif 14 in human.body_parts.keys():
                                                ref_center_pt_2d = (int(human.body_parts[14].x * image_w + 0.5),
                                                                    int(human.body_parts[14].y * image_h + 0.5))

                                            elif 15 in human.body_parts.keys():
                                                ref_center_pt_2d = (int(human.body_parts[15].x * image_w + 0.5),
                                                                    int(human.body_parts[15].y * image_h + 0.5))

                                            ref_center_pt_3d = transform_2d_to_3d(depth, ref_center_pt_2d[0],
                                                                                  ref_center_pt_2d[1], camera_param)

                                        ## 손목 좌표
                                        # if i == 4:
                                        #     ref_center_pt_2d = (int(human.body_parts[4].x * image_w + 0.5), int(human.body_parts[4].y * image_h + 0.5))
                                        #     ref_center_pt_3d = transform_2d_to_3d(depth, ref_center_pt_2d[0], ref_center_pt_2d[1], camera_param)
                                        #
                                        # elif i == 7:
                                        #     ref_center_pt_2d = (int(human.body_parts[7].x * image_w + 0.5), int(human.body_parts[7].y * image_h + 0.5))
                                        #     ref_center_pt_3d = transform_2d_to_3d(depth, ref_center_pt_2d[0], ref_center_pt_2d[1], camera_param)

                                        body_part_pre = human.body_parts[i - 1]
                                        body_part = human.body_parts[i]

                                        # 3d bbox detection
                                        pre_x_scale = int(body_part_pre.x * image_w + 0.5)
                                        pre_y_scale = int(body_part_pre.y * image_h + 0.5)
                                        x_scale = int(body_part.x * image_w + 0.5)
                                        y_scale = int(body_part.y * image_h + 0.5)

                                        center_pt1 = transform_2d_to_3d(depth, pre_x_scale, pre_y_scale, camera_param)
                                        center_pt2 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                        grad = (center_pt2[0] - center_pt1[0], center_pt2[1] - center_pt1[1],
                                                center_pt2[2] - center_pt1[2])

                                        # co-linearity
                                        center_pt3 = (center_pt1[0] + k_val * grad[0], center_pt1[1] + k_val * grad[1],
                                                      center_pt1[2] + k_val * grad[2])
                                        center_pt4 = (
                                        center_pt1[0] + k_val * 3 * grad[0], center_pt1[1] + k_val * 3 * grad[1],
                                        center_pt1[2] + k_val * 3 * grad[2])

                                        if center_pt1[2] <= 0 or center_pt2[2] <= 0 or center_pt3[2] <= 0:
                                            continue

                                        # center_pt2_2d = (int(fx * center_pt2[0] / center_pt2[2] + cx),
                                        #                  int(fy * center_pt2[1] / center_pt2[2] + cy))
                                        # center_pt3_2d = (int(fx * center_pt3[0] / center_pt3[2] + cx),
                                        #                  int(fy * center_pt3[1] / center_pt3[2] + cy))
                                        # center_pt4_2d = (int(fx * center_pt4[0] / center_pt4[2] + cx),
                                        #                  int(fy * center_pt4[1] / center_pt4[2] + cy))

                                        Xmin = center_pt3[0] - BOX_HALF_LEN
                                        Xmax = center_pt3[0] + BOX_HALF_LEN
                                        Ymin = center_pt3[1] - BOX_HALF_LEN
                                        Ymax = center_pt3[1] + BOX_HALF_LEN
                                        Zmin = center_pt3[2] - BOX_HALF_LEN
                                        Zmax = center_pt3[2] + BOX_HALF_LEN

                                        # imgpts = []  # 8-point cube
                                        # imgpts.append((fx * Xmin / Zmin + cx, fy * Ymin / Zmin + cy))
                                        # imgpts.append((fx * Xmax / Zmin + cx, fy * Ymin / Zmin + cy))
                                        # imgpts.append((fx * Xmax / Zmin + cx, fy * Ymax / Zmin + cy))
                                        # imgpts.append((fx * Xmin / Zmin + cx, fy * Ymax / Zmin + cy))
                                        # imgpts.append((fx * Xmin / Zmax + cx, fy * Ymin / Zmax + cy))
                                        # imgpts.append((fx * Xmax / Zmax + cx, fy * Ymin / Zmax + cy))
                                        # imgpts.append((fx * Xmax / Zmax + cx, fy * Ymax / Zmax + cy))
                                        # imgpts.append((fx * Xmin / Zmax + cx, fy * Ymax / Zmax + cy))
                                        # imgpts = np.int32(imgpts)

                                        # img = img.astype(np.uint8)

                                        # draw line
                                        # cv2.line(img, center_pt2_2d, center_pt4_2d, (255, 0, 0))

                                        # draw 3d bbox
                                        # img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 0, 255), 1)
                                        # # draw pillars in blue color
                                        # for j, k in zip(range(4), range(4, 8)):
                                        #     img = cv2.line(img, tuple(imgpts[j]), tuple(imgpts[k]), (0, 0, 255), 1)
                                        # # draw top layer in red color
                                        # img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 1)

                                        # 여기 z 값 고찰 필요...
                                        xmin = fx * Xmin / center_pt3[2] + cx
                                        xmax = fx * Xmax / center_pt3[2] + cx
                                        ymin = fy * Ymin / center_pt3[2] + cy
                                        ymax = fy * Ymax / center_pt3[2] + cy

                                        xmin = np.clip(xmin, 0, image_w)

                                        box_arr_hand = np.array([xmin, ymin, xmax, ymax, 0.99])
                                        hand_bbox_results.append({'bbox': box_arr_hand})
                                        id_joint_list.append({'id': i, 'ref_center_pt_2d': ref_center_pt_2d,
                                                              'ref_center_pt_3d': ref_center_pt_3d})

                                        ## arm position 추가
                                        x_list = []
                                        y_list = []

                                        if i == 4:
                                            p_list = [2, 3]
                                        elif i == 7:
                                            p_list = [5, 6]

                                        for p in p_list:
                                            if p not in humans[baby_idx].body_parts.keys():
                                                continue
                                            body_part = human.body_parts[p]
                                            x_scale = int(body_part.x * image_w + 0.5)
                                            y_scale = int(body_part.y * image_h + 0.5)
                                            x_list.append(x_scale)
                                            y_list.append(y_scale)

                                        xmin = min(max(int(min(x_list)) - 20, 0), xmin)
                                        xmax = max(min(int(max(x_list)) + 20, image_w - 1), xmax)
                                        ymin = min(max(int(min(y_list)) - 20, 0), ymin)
                                        ymax = max(min(int(max(y_list)) + 20, image_h - 1), ymax)
                                        ########################

                                        box_arr_arm = np.array([xmin, ymin, xmax, ymax, 0.99])

                                        arm_bbox_results.append({'bbox': box_arr_arm})

                    if args.viz_skeleton == True:
                        img = draw_humans(img, humans, baby_idx)

                    preds_list = []
                    prob_hand_list = []

                    for i in range(len(hand_bbox_results)): # 손 갯수
                        ## Hand
                        x_min = max(int(hand_bbox_results[i]['bbox'][0]), 0)
                        y_min = max(int(hand_bbox_results[i]['bbox'][1]), 0)
                        x_max = min(int(hand_bbox_results[i]['bbox'][2]), image_w)
                        y_max = min(int(hand_bbox_results[i]['bbox'][3]), image_h)

                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, 4)

                        hand_roi = img_copy[y_min:y_max, x_min:x_max, :]
                        if hand_roi.size == 0:
                            continue

                        #hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                        hand_roi = hand_roi[:, :, ::-1]
                        hand_roi = Image.fromarray(hand_roi)
                        hand_roi = tf_Resize(hand_roi)
                        hand_roi = tf_ToTensor(hand_roi)
                        hand_roi = tf_Normalize(hand_roi)
                        hand_roi = torch.unsqueeze(hand_roi, 0)

                        ## PointDetNet model inference
                        if args.SSL == 'None':
                            outputs_hand = model_PointDetNet_hand(hand_roi.cuda())
                        else:
                            outputs_hand, _ = model_PointDetNet_hand(hand_roi.cuda(), hand_roi.cuda())

                        ## Arm
                        # x_min = max(int(arm_bbox_results[i]['bbox'][0]), 0)1
                        # y_min = max(int(arm_bbox_results[i]['bbox'][1]), 0)
                        # x_max = min(int(arm_bbox_results[i]['bbox'][2]), image_w)
                        # y_max = min(int(arm_bbox_results[i]['bbox'][3]), image_h)
                        #
                        # arm_roi = img_copy[y_min:y_max, x_min:x_max, :]
                        #
                        # arm_roi = cv2.cvtColor(arm_roi, cv2.COLOR_BGR2RGB)
                        # arm_roi = Image.fromarray(arm_roi)
                        # arm_roi = tf_Resize(arm_roi)
                        # arm_roi = tf_ToTensor(arm_roi)
                        # arm_roi = tf_Normalize(arm_roi)
                        # arm_roi = torch.unsqueeze(arm_roi, 0)

                        ## PointDetNet model inference
                        #outputs_arm = model_PointDetNet_arm(arm_roi.cuda())

                        if args.aggre_results == 'average':
                            ## Aggregate predictions (late fusion - average)
                            # _, predictions = torch.max(m(outputs_hand) + m(outputs_arm), 1)
                            # _, predictions = torch.max(outputs_hand + outputs_arm, 1)
                            # pred = predictions.cpu().detach().numpy()
                            pass
                        elif args.aggre_results == 'and_op':
                            # And logic
                            # _, predictions_hand = torch.max(outputs_hand, 1)
                            # pred_hand = predictions_hand.cpu().detach().numpy()
                            # _, predictions_arm = torch.max(outputs_arm, 1)
                            # pred_arm = predictions_arm.cpu().detach().numpy()
                            #
                            # #if pred_hand == 0:
                            # if pred_hand == 0 and pred_arm == 0: ######################## And operation
                            #     pred = 0
                            # else:
                            #     pred = 1
                            pass

                        elif args.aggre_results == 'hand_only':
                            # And logic
                            _, predictions_hand = torch.max(outputs_hand, 1)
                            pred = predictions_hand.cpu().detach().numpy()

                        prob_hand = softmax(outputs_hand.cpu().detach().numpy())
                        prob_hand = prob_hand[0][0] * 100
                        # prob_arm = softmax(outputs_arm.cpu().detach().numpy())
                        # prob_arm = prob_arm[0][0] * 100

                        preds_list.append(pred)
                        prob_hand_list.append(prob_hand)

                        if pred == 1:  # pointing gesture no
                            text_color = (0, 0, 255)

                        else:  # pointing gesture yes
                            text_color = (0, 255, 0)

                            if args.vis_pointing_ray == True:
                                # pointing gesture 수행 중이라고 판단될때만 pointing direction estimation 수
                                # test a single image, with a list of bboxes.
                                pose_results, returned_outputs = inference_top_down_pose_model(
                                    pose_model,
                                    img_copy,
                                    hand_bbox_results[i:i + 1],
                                    bbox_thr=args.bbox_thr,
                                    format='xyxy',
                                    dataset=dataset,
                                    dataset_info=dataset_info,
                                    return_heatmap=False,
                                    outputs=False)

                                # show the results of hand pose estimation
                                img = vis_pose_result(
                                    pose_model,
                                    img,
                                    pose_results,
                                    dataset=dataset,
                                    dataset_info=dataset_info,
                                    kpt_score_thr=args.kpt_thr,
                                    radius=args.radius,
                                    thickness=args.thickness,
                                    show=False)

                                # ray casting
                                #####
                                # 검지 손가락 tip 좌표
                                # tip이랑 2번째 관절 평
                                finger_pt_2d = (
                                int((pose_results[0]['keypoints'][8][0] + pose_results[0]['keypoints'][6][0]) / 2.0),
                                int((pose_results[0]['keypoints'][8][1] + pose_results[0]['keypoints'][6][1]) / 2.0))
                                finger_pt_3d = transform_2d_to_3d(depth, int(finger_pt_2d[0]), int(finger_pt_2d[1]),
                                                                  camera_param)

                                # 검지 손가락 mcp 좌표
                                ref_pt_2d = (
                                int(pose_results[0]['keypoints'][5][0]), int(pose_results[0]['keypoints'][5][1]))
                                ref_pt_3d = transform_2d_to_3d(depth, int(ref_pt_2d[0]), int(ref_pt_2d[1]), camera_param)

                                # 피벗 좌표
                                # ref_pt_2d = id_joint_list[i]['ref_center_pt_2d']
                                # ref_pt_3d = id_joint_list[i]['ref_center_pt_3d']

                                grad = (finger_pt_3d[0] - ref_pt_3d[0], finger_pt_3d[1] - ref_pt_3d[1],
                                        finger_pt_3d[2] - ref_pt_3d[2])

                                extended_pt_3d = (ref_pt_3d[0] + 10.0 * grad[0], ref_pt_3d[1] + 10.0 * grad[1],
                                                  ref_pt_3d[2] + 10.0 * grad[2])

                                if extended_pt_3d[2] > 0:
                                    extended_pt_2d = (int(fx * extended_pt_3d[0] / extended_pt_3d[2] + cx),
                                                      int(fy * extended_pt_3d[1] / extended_pt_3d[2] + cy))

                                    cv2.line(img, ref_pt_2d, extended_pt_2d, (255, 0, 0), thickness=3)

                            if args.input == 'kinect':

                                ## finding attention point on monitor
                                if args.demo_atten_point == 'monitor':
                                    x_attention = 0
                                    y_attention = 0

                                    # pointint ray
                                    if grad[2] != 0:
                                        t = -ref_pt_3d[2] / grad[2]
                                        x_3d = (ref_pt_3d[0] + t * grad[0], ref_pt_3d[1] + t * grad[1], 0)
                                        x_2d = (-x_3d[0] / mm_per_pixel[0] + monitor_mm[0] / 2.0,
                                                x_3d[1] / mm_per_pixel[1] + monitor_mm[1])

                                        x_attention = int(min(max(0, x_2d[0]), monitor_pixel[0] - 1))
                                        y_attention = int(min(max(0, x_2d[1]), monitor_pixel[1] - 1))

                                        img_monitor = cv2.circle(img_monitor, (x_attention, y_attention), 20,
                                                                 (255, 0, 0), -1)

                                else:
                                    pass
                                    ## finding attention point on 3d space (3d point hit search algorithm)
                                    # h = depth.shape[0]
                                    # w = depth.shape[1]
                                    # similarity_max = 0
                                    # x_attention = 0
                                    # y_attention = 0
                                    #
                                    # c, r = np.meshgrid(np.arange(w), np.arange(h), sparse=True)
                                    # valid = abs(depth - finger_pt_3d[2]) >= 100 # 최소 거리
                                    # z = np.where(valid, depth, 0.0)
                                    # x = np.where(valid, z * (c - cx) / fx, 0)
                                    # y = np.where(valid, z * (r - cy) / fy, 0)
                                    # point_cloud = np.dstack((x, y, z))
                                    # point_cloud_reshape = np.reshape(point_cloud, (h*w, 3))
                                    # point_cloud_reshape[:, 0] = point_cloud_reshape[:, 0] - ref_pt_3d[0]
                                    # point_cloud_reshape[:, 1] = point_cloud_reshape[:, 1] - ref_pt_3d[1]
                                    # point_cloud_reshape[:, 2] = point_cloud_reshape[:, 2] - ref_pt_3d[2]
                                    #
                                    # norm_XYZ = point_cloud_reshape.copy()
                                    # norm1 = LA.norm(point_cloud_reshape, axis=1)
                                    # norm1 = np.expand_dims(norm1, axis=1)
                                    # point_cloud_reshape = np.divide(point_cloud_reshape, norm1)
                                    #
                                    # # pointint ray
                                    # lineEq = np.cross(finger_pt_3d, ref_pt_3d)
                                    #
                                    # norm_XYZ[:, 0] = norm_XYZ[:, 0] / norm_XYZ[:, 2]
                                    # norm_XYZ[:, 1] = norm_XYZ[:, 1] / norm_XYZ[:, 2]
                                    # norm_XYZ[:, 2] = norm_XYZ[:, 2] / norm_XYZ[:, 2]
                                    # coef = np.dot(norm_XYZ, lineEq)
                                    # dist = (coef * coef) / (lineEq[0]*lineEq[0] + lineEq[1]*lineEq[1]) #거리 제곱
                                    #
                                    # ray_pointing = tuple(map(operator.sub, finger_pt_3d, ref_pt_3d))
                                    # ray_pointing = np.asarray(ray_pointing)
                                    # norm2 = math.sqrt(ray_pointing[0] * ray_pointing[0] + ray_pointing[1] * ray_pointing[1] + ray_pointing[2] * ray_pointing[2])
                                    # ray_pointing = ray_pointing / norm2
                                    #
                                    # ## unit vector간의 cosine 유사도
                                    # sim_matrix = np.dot(point_cloud_reshape, ray_pointing)
                                    #
                                    # sim_matrix = np.where(dist<0.0005, sim_matrix, 0)
                                    #
                                    # max_idx = np.argmax(sim_matrix)
                                    #
                                    # ## euclidean 거리(잘 안됨...왜일)
                                    # # dist_matrix = np.cross(point_cloud_reshape, ray_pointing)
                                    # # dist = LA.norm(point_cloud_reshape, axis=1)
                                    # # min_idx = np.argmin(dist)
                                    #
                                    # y_attention = int(max_idx / w)
                                    # x_attention = int(max_idx - y_attention * w)
                                    #
                                    # img = cv2.circle(img, (x_attention, y_attention), 10, (0, 0, 255), -1)

                        # visualize pointing or not in each hand
                        text = pointing_label[1 - int(pred)]
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                        text_w, text_h = text_size
                        pos = (x_min, max(0,y_min - 60))
                        cv2.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h + 10),
                                      text_color, -1)
                        cv2.putText(img, text, (pos[0], pos[1] + text_h + 2 - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

                        ## alpha blending
                        img_cp = img.copy()
                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, -1)
                        alpha = 0.1
                        img = cv2.addWeighted(img, alpha, img_cp, 1 - alpha, gamma=0)
                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, 4)


                        if vis_grad_cam == True:
                            pass
                            ## Grad-CAM model inference
                            # target_category = None
                            #
                            # cam_algorithm = GradCAM
                            # with cam_algorithm(model=model_PointDetNet_arm,
                            #                    target_layers=target_layers,
                            #                    use_cuda=True) as cam:
                            #     # AblationCAM and ScoreCAM have batched implementations.
                            #     # You can override the internal batch size for faster computation.
                            #     cam.batch_size = 32
                            #     torch.set_grad_enabled(True)
                            #
                            #     grayscale_cam = cam(input_tensor=arm_roi,
                            #                         target_category=target_category,
                            #                         aug_smooth=args.aug_smooth,
                            #                         eigen_smooth=args.eigen_smooth)
                            #
                            #     torch.set_grad_enabled(False)
                            #
                            #     # Here grayscale_cam has only one image in the batch
                            #     grayscale_cam = grayscale_cam[0, :]
                            #
                            #     cam_image = show_cam_on_image(arm_roi_copy, grayscale_cam, use_rgb=True)
                            #
                            #     cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                            #     win_name = 'gradCAM_' + str(i)
                            #     cv2.imshow(win_name, cam_image)

                    prob_hand_list_video.append(prob_hand_list)
                    pred_list_video.append(preds_list)

                    if len(preds_list) == 0:
                        aggregated_pred = 0
                    else:
                        aggregated_pred = 1 - min(preds_list)  # 손이 2개 detect 됐는데 둘다 포인팅 아니면 둘다 1, min 값 취해도 1

                    positive_persist_thres = args.positive_persist_thres

                    if args.positive_persist_thres == 0:
                        pred_buf = 1
                        positive_persist_thres = 1

                    if (pred_buf == 1 and int(aggregated_pred) == 1):
                        pos_cnt += 1
                        text_color_arm = (0, 0, 255)
                    if (int(aggregated_pred) == 0):
                        pos_cnt = 0
                    if (pos_cnt >= positive_persist_thres and b_flag_pointing == False):
                        #pos_cnt = 0
                        b_flag_pointing = True
                        text_color_pointing = (0, 255, 0)

                        #### pointing이라고 판단됐을때 pointing 정반응 확률 계산.
                        for i in range(args.positive_persist_thres + 1): #코드수정:+1
                            index = pred_list_video[-1 - i].index(0) #0(poining) index finding
                            prob_pointing += prob_hand_list_video[-1 - i][index]

                        prob_pointing /= float(args.positive_persist_thres + 1)

                        ## pointing이라고 판단되면 while 문 out
                        break

                    pred_buf = int(aggregated_pred)

                if args.showFps:
                    fps = 1 / (time.time() - last_time)
                    img = cv2.putText(img, 'fps: ' + "%.2f" % (fps), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 0, 255), 2)
                    # img = cv2.putText(img, 'frames: ' + "%d" % (frame_count), (25, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    #                   1.2, (0, 0, 255), 2)


                # img = cv2.putText(img, 'pos_count: '  "%s" % (pos_cnt), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                #                   text_color_finger, 2)
                img = cv2.putText(img, 'b_pointing: '  "%s" % (b_flag_pointing), (25, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color_pointing, 2)
                img = cv2.putText(img, 'label: '  "%s" % (pointing_label[1 - labels[idx_list]]), (25, 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color_finger, 2)
                # img = cv2.putText(img, 'prob_hand: '  "%s" % (prob_hand), (25, 250),
                #                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color_finger, 2)
                # img = cv2.putText(img, 'prob_arm: '  "%s" % (prob_arm), (25, 300),
                #                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color_finger, 2)

                if args.viz_scale != 1:
                    img = cv2.resize(img, dsize=(0, 0), fx=args.viz_scale, fy=args.viz_scale,
                                     interpolation=cv2.INTER_LINEAR)
                if args.show:
                    # cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('Image', img)
                    # cv2.imshow("Transformed Depth", colorized_depth)
                    k = cv2.waitKey(1)

                    if k == 27:  # esc key
                        break
                    if k == 113:  # q key
                        vis_grad_cam = True
                    if k == 119:  # w key
                        vis_grad_cam = False
                        cv2.destroyWindow('gradCAM_0')
                        cv2.destroyWindow('gradCAM_1')

                if args.saveVideo == True:
                    out.write(img)


                frame_count += 1

            except EOFError:
                break

        ### write final pointing probability
        pointing_prob = None

        #print('b_flag_pointing: {} '.format(b_flag_pointing))

        if b_flag_pointing == True:
            pointing_prob = prob_pointing
        elif b_flag_pointing == False:

            new_list = []
            prob_hand_list_video = np.array(prob_hand_list_video)
            for i in range(len(prob_hand_list_video)):
                new_list.append(np.mean(prob_hand_list_video[i]))

            new_list = [x for x in new_list if math.isnan(x) == False]

            pointing_prob = np.mean(new_list)

        #print('pointing_prob: {} '.format(pointing_prob))

        fp = open(os.path.join('checkpoints/logs', txt_dir), 'a')

        if labels[idx_list] == 0:
            gt = True
        else:
            gt = False

        if gt == b_flag_pointing:
            correct = 1
        else:
            correct = 0

        fp.write(
            'path: {}, gt_pointing.: {}, b_flag_pointing: {}, correct: {}, pointing_prob: {:.1f} \n'.
                format(input_list[idx_list], gt, b_flag_pointing, correct, pointing_prob))
        fp.close()


        if args.saveVideo == True:
            out.release()
            video_num_cnt += 1

        # Calculate accuracy
        if b_flag_pointing == True:
            pred = 0
        else:
            pred = 1

        if pred == labels[idx_list]:

            running_corrects += 1

            if pred == 0:
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if pred == 0:
                FP = FP + 1
            else:
                FN = FN + 1

            print(input_list[idx_list])

        ndata += 1

        if args.input == 'mkv':
            playback.close()

        if args.show:
            cv2.destroyAllWindows()

    # Final accuracy
    acc = running_corrects / ndata

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    # acc2 = (TP + TN) / (TP + FN + FP + TN)
    f_score = 2 * (precision * recall) / (precision + recall + eps)

    print('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f}'.format(acc, recall, precision,
                                                                                             f_score))
    print('TP: {}, FP: {}, TN: {}, FN: {} '.format(TP, FP, TN, FN))

    fp = open(os.path.join('checkpoints/logs', txt_dir), 'a')
    fp.write('ensemble_th: {} \n'.format(args.positive_persist_thres))
    fp.write(
        'model_arm: {} model_hand: {} test_Accuracy: {:.3f}, recall:{:.3f}, precision:{:.3f}, f_socre:{:.3f}'
        ', TP: {}, FP: {}, TN: {}, FN: {}  \n'.
            format(args.model_name_arm, args.model_name_hand, acc, recall, precision, f_score, TP, FP, TN, FN))
    fp.close()