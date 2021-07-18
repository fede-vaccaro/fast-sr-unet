import time
from threading import Thread
import data_loader as dl
import torch

torch.backends.cudnn.benchmark = True
import numpy as np
from models import *
import utils
from tqdm import tqdm
import cv2
from pytorch_unet import UNet, SRUnet, SimpleResNet
from queue import Queue


# from apex import amp

def save_with_cv(pic, imname):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(imname, npimg)


def write_to_video(pic, writer):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = npimg.astype('uint8')
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(npimg, '540p CRF 23 + bicubic', (50, 1030), font, 1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(npimg, 'SR-Unet (ours)', (1920 // 2 + 50, 1020), font, 1, (10, 10, 10), 2, cv2.LINE_AA)

    writer.write(npimg)


def get_padded_dim(H_x, W_x, border=0, mod=16):
    modH, modW = H_x % (mod + border), W_x % (mod + border)
    padW = ((mod + border) - modW) % (mod + border)
    padH = ((mod + border) - modH) % (mod + border)

    new_H = H_x + padH
    new_W = W_x + padW

    return new_H, new_W, padH, padW


def pad_input(x, padH, padW):
    x = F.pad(x, [0, padW, 0, padH])
    return x


def cv2toTorch(im):
    im = im / 255
    im = torch.Tensor(im).cuda()
    im = im.permute(2, 0, 1).unsqueeze(0)
    im = dl.normalize_img(im)
    return im


def torchToCv2(pic, rescale_factor=1.0):
    if rescale_factor != 1.0:
        pic = F.interpolate(pic, scale_factor=rescale_factor, align_corners=True, mode='bicubic')
    pic = dl.de_normalize(pic.squeeze(0))
    pic = pic.permute(1, 2, 0) * 255
    npimg = pic.byte().cpu().numpy()
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    return npimg


def blend_images(i1, i2):
    w = i1.shape[-1]
    w_4 = w // 4
    i1 = i1[:, :, :, w_4:w_4 * 3]
    i2 = i2[:, :, :, w_4:w_4 * 3]
    out = torch.cat([i1, i2], dim=3)
    return out


if __name__ == '__main__':
    args = utils.ARArgs()
    enable_write_to_video = False
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR

    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    model_path = args.MODEL_NAME
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model.reparametrize()

    path = args.CLIPNAME
    cap = cv2.VideoCapture(path)
    reader = torchvision.io.VideoReader(path, "video")

    if enable_write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter('rendered.mp4', fourcc, 30, (1920, 1080))

    metadata = reader.get_metadata()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)

    frame_queue = Queue(1)
    out_queue = Queue(1)

    reader.seek(0)


    def read_pic(cap, q):
        count = 0
        start = time.time()
        while True:
            cv2_im = next(cap)['data']  # .cuda().float()
            cv2_im = cv2_im.cuda().float()

            x = dl.normalize_img(cv2_im / 255.).unsqueeze(0)

            x_bicubic = torch.clip(F.interpolate(x, scale_factor=args.UPSCALE_FACTOR * args.DOWNSAMPLE, mode='bicubic'),
                                   min=-1, max=1)

            x = F.pad(x, [0, padW, 0, padH])
            count += 1
            q.put((x, x_bicubic))


    def show_pic(cap, q):
        while True:
            out = q.get()
            scale = 1
            cv2_out = torchToCv2(out, rescale_factor=scale)
            cv2.imshow('rendering', cv2_out)
            cv2.waitKey(1)


    t1 = Thread(target=read_pic, args=(reader, frame_queue)).start()
    t2 = Thread(target=show_pic, args=(cap, out_queue)).start()
    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frametime = 1000 / target_fps

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:
            t0 = time.time()

            x, x_bicubic = frame_queue.get()
            out = model(x)[:, :, :int(height) * 2, :int(width) * 2]

            out_true = i // (target_fps * 3) % 2 == 0

            if not args.SHOW_ONLY_HQ:
                out = blend_images(x_bicubic, out)
            out_queue.put(out)
            frametime = time.time() - t0
            if frametime < target_frametime * 1e-3:
                time.sleep(target_frametime * 1e-3 - frametime)

            if enable_write_to_video:
                write_to_video(out, hr_video_writer)
                if i == 30 * 10:
                    hr_video_writer.release()
                    print("Releasing video")

            tqdm_.set_description("frame time: {}; fps: {}; {}".format(frametime * 1e3, 1000 / frametime, out_true))
