from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os

from .configs.loader import load_yaml
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode
from tqdm import tqdm

"""
Load the configures and args
"""
args = load_yaml('utils/FaceDetector/configs/args_face_detector.yaml')
cfg_mnet = load_yaml('utils/FaceDetector/configs/cfg_mnet.yaml')
cfg_re50 = load_yaml('utils/FaceDetector/configs/cfg_re50.yaml')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    # unused_pretrained_keys = ckpt_keys - model_keys
    # missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_path = os.path.join(os.getcwd(), "utils/FaceDetector/weights", pretrained_path)

    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def get_detector():
    torch.set_grad_enabled(False)

    cfg = None
    if args['network'] == "mobile0.25":
        cfg = cfg_mnet
    elif args['network'] == "resnet50":
        cfg = cfg_re50

    # Initialize the net (detector)
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args['trained_model'], args['cpu'])
    net.eval()
    # print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args['cpu'] else "cuda")
    net = net.to(device)
    return net


def get_slices(frame_num, batch_size):
    """
    Return the a list of slices, which are used for separating the image sequence (video) into batch.
    :param frame_num: Total frame num of the input imgs (video) [Type: Int]
    :param batch_size: [Type: Int]
    :return: slices: [Type: list]
    """
    num_batch = frame_num // batch_size
    batch_size_last = frame_num % batch_size
    slices = []
    for i in range(num_batch):
        slices.append(slice(i*batch_size, (i+1)*batch_size))
    if batch_size_last != 0:
        slices.append(slice(num_batch* batch_size, num_batch*batch_size+batch_size_last))
    return slices


def predict_face_box_batch(net, imgs):
    """
    - Predict the face bounding-box from the input image sequence (video)
    - Modified from the original `predict_face_box` to support batch-prediction, improving the efficiency.
    :param net: RetinaFace
    :param imgs: Image sequence [Type: list][Shape: (N, (H, W, C))]
    :return: face_boxes [Type: list]
             confidence_scores [Type: list]
    """

    cfg = None
    if args['network'] == "mobile0.25":
        cfg = cfg_mnet
    elif args['network'] == "resnet50":
        cfg = cfg_re50
    device = torch.device("cpu" if args['cpu'] else "cuda")

    """
    Adjust the image size.
    Default setting is keep the original size. 
    If the input video frame's size is too large, 
    you could enable the adjustment by modifying "./configs/args_face_detector.yaml"-> "origin_size: False".
    """
    im_shape = imgs[0].shape
    im_height, im_width, _ = imgs[0].shape
    if args['origin_size']:
        resize = 1
    else:
        target_size = 1600
        max_size = 2150
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

    """
    Empirically set the batch_size.
    You could change to larger batch_size if supported.
    """
    im_size_max = np.max(im_shape[0:2])
    if im_size_max < 1000:
        batch_size = 16
    elif im_size_max < 1500:
        batch_size = 8
    else:
        batch_size = 4
    # print("Image max size: {}, Batch size: {}".format(im_size_max, batch_size))
    scale = torch.Tensor([im_shape[1], im_shape[0], im_shape[1], im_shape[0]])
    scale = scale.to(device)


    frame_num = len(imgs)
    slices = get_slices(frame_num, batch_size)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    face_boxes = []
    confidence_scores = []

    locs = []
    confs = []

    for batch_slice in tqdm(slices):
        samples = []
        img_batch = imgs[batch_slice]
        for img in img_batch:
            sample = np.float32(img)
            sample -= (104, 117, 123)
            sample = sample.transpose(2, 0, 1)
            samples.append(np.expand_dims(sample, axis=0))
        img_batch = torch.from_numpy(np.concatenate(samples))
        img_batch = img_batch.to(device)
        loc, conf, _ = net(img_batch)  # forward pass
        locs.append(loc)
        confs.append(conf)

    # Concat the loc and conf
    locs = torch.cat(locs, dim=0)
    confs = torch.cat(confs, dim=0)

    # print("Misc-Stage")
    for loc, conf in zip(locs, confs):
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args['confidence_threshold'])[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]

        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args['nms_threshold'])
        # keep = nms(dets, args['nms_threshold'],force_cpu=args['cpu'])
        dets = dets[keep, :]

        """
        Handle the situation when the len of dets[] is zero
        """
        if len(dets) == 0:
            if len(face_boxes) == 0:
                face_boxes.append((0, 0, 1, 1))  # A meaning less box. Just to avoid the crash.
            else:
                face_boxes.append(face_boxes[-1])
            confidence_scores.append(0)
        else:
            # Only return the first face
            det = dets[0]
            box, score = (det[0], det[1], det[2], det[3]), det[4]

            face_boxes.append(box)
            confidence_scores.append(score)

    return face_boxes, confidence_scores


def predict_face_box(net, frame):
    """
    - Face prediction function in the original repo (slightly modified and just for reference)
    - Perform the prediction on a single frame.
    :param net:
    :param frame:
    :return:
    """
    cfg = None
    if args['network'] == "mobile0.25":
        cfg = cfg_mnet
    elif args['network'] == "resnet50":
        cfg = cfg_re50
    device = torch.device("cpu" if args['cpu'] else "cuda")

    # _t = {'forward_pass': Timer(), 'misc': Timer()}
    img = np.float32(frame)
    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    if args['origin_size']:
        resize = 1
    else:
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)


    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, _ = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args['confidence_threshold'])[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]

    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args['nms_threshold'])
    # keep = nms(dets, args['nms_threshold'],force_cpu=args['cpu'])
    dets = dets[keep, :]

    # Only return the first face
    det = dets[0]
    box, score = (det[0], det[1], det[2], det[3]), det[4]
    return box, score