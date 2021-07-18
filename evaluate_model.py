# from apex import amp
import pandas as pd
import utils, os

args = utils.ARArgs()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

from pathlib import Path

import tqdm
import data_loader as dl
import pytorch_ssim as torch_ssim
import lpips
import numpy as np

from models import *
from pytorch_unet import *
from render import cv2toTorch, torchToCv2
import cv2
from queue import Queue
from threading import Thread
import shutil


def cat_dim(t1, t2):
    return torch.cat([t1, t2], dim=1)


def save_with_cv(pic, imname):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(imname, npimg)


def evaluate_model(test_dir_prefix, output_generated, video_prefix, filename, from_second=0, to_second=None,
                   test_lq=True,
                   skip_model_testing=False, crf=None):
    device = 'cuda'

    test_tof = False and not skip_model_testing
    test_tlp = False and not skip_model_testing

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

    print("Loading model: ", filename)
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model = model.cuda()

    # model = amp.initialize(model, opt_level='O2')

    lpips_metric = lpips.LPIPS(net='alex')
    lpips_metric = lpips_metric.to(device)
    ssim = torch_ssim.SSIM(window_size=11)
    ssim = ssim.to(device)

    resolution_lq = args.TEST_INPUT_RES
    resolution_hq = args.TEST_OUTPUT_RES
    if crf is not None:
        crf_ = crf
    else:
        crf_ = 23

    lq_file_path = str(test_dir_prefix) + f"/encoded{resolution_lq}CRF{crf_}/" + video_prefix + ".mp4"

    cap_lq = cv2.VideoCapture(lq_file_path)
    video_size = cap_lq.get(cv2.CAP_PROP_BITRATE)  # os.path.getsize(lq_file_path) / 1e6
    time_length = cap_lq.get(cv2.CAP_PROP_FRAME_COUNT) / cap_lq.get(cv2.CAP_PROP_FPS)
    cap_hq = cv2.VideoCapture(str(test_dir_prefix) + f"/{video_prefix}" + ".y4m")

    gaussian_filter = utils.get_gaussian_kernel(sigma=0.5, kernel_size=5)
    gaussian_filter.to(device)

    lq_queue = Queue(1)
    hq_queue = Queue(1)
    out_queue = Queue(1)

    total_frames = int(cap_hq.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap_hq.get(cv2.CAP_PROP_FPS))

    from_frame = fps * from_second
    to_frame = fps * to_second

    if to_frame is None:
        to_frame = total_frames

    to_frame = min(to_frame, total_frames)

    def read_pic(cap, q, from_frame_, to_frame_):
        count = 0
        cap.set(cv2.CAP_PROP_POS_MSEC, from_second)
        while cap.isOpened():
            success, cv2_im = cap.read()
            if success:
                cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                x = cv2toTorch(cv2_im)
                x_bicubic = x  # torch.clip(F.interpolate(x, scale_factor=2, mode='bicubic'), min=-1, max=1)
                q.put((x, x_bicubic))
                count += 1
                if count == (to_frame_ - from_frame_):
                    print("Releasing cap")
                    cap.release()
            else:
                cap.release()

    finish = False

    def save_pic(q):
        count = 0
        while True:
            imname = video_prefix + f"_{count}.png"

            frame_name_pattern = imname.split(".")[0].split("_")[:-1]
            frame_name_pattern = "_".join(frame_name_pattern) + "_frame"
            frame_name_pattern = dest / frame_name_pattern

            imname = str(dest / imname)
            y_fake = q.get()
            if y_fake is not None:
                save_with_cv(y_fake, imname)
                count += 1
            else:
                break

    ssim_ = []
    lpips_ = []
    tLP = []
    tOF = []

    ssim_x = []
    lpips_x = []
    tLP_x = []
    tOF_x = []
    print("Evaluation")
    tqdm_ = tqdm.tqdm(range(to_frame - from_frame))

    dest = test_dir_prefix.split("/")
    dest_dir = Path("/".join(dest))
    dest = dest_dir / "out"
    dest.mkdir(exist_ok=True, parents=True)
    border = 0

    H_x, W_x = cap_lq.get(cv2.CAP_PROP_FRAME_HEIGHT), cap_lq.get(cv2.CAP_PROP_FRAME_WIDTH)
    H_y, W_y = cap_hq.get(cv2.CAP_PROP_FRAME_HEIGHT), cap_hq.get(cv2.CAP_PROP_FRAME_WIDTH)

    framerate = int(cap_lq.get(cv2.CAP_PROP_FPS))

    H_x = int(H_x)
    W_x = int(W_x)

    H_y = int(H_y)
    W_y = int(W_y)

    print(f"Src resolution: {W_x}x{H_x}")
    print(f"Dest resolution: {W_y}x{H_y}")

    modH, modW = H_x % (16 + border), W_x % (16 + border)
    padW = ((16 + border) - modW) % (16 + border)
    padH = ((16 + border) - modH) % (16 + border)

    new_H = H_x + padH
    new_W = W_x + padW


    model.batch_size = 1
    model.width = new_W  # x.shape[-1] + (patch_size - modW) % patch_size
    model.height = new_H  # x.shape[-2] + (patch_size - modW) % patch_size

    print(f"Padded src resolution: {new_W}x{new_H}")

    prev_sr = None
    prev_gt = None
    prev_x = None

    thread1 = Thread(target=read_pic, args=(cap_lq, lq_queue, from_frame, to_frame))  # .start()
    thread2 = Thread(target=read_pic, args=(cap_hq, hq_queue, from_frame, to_frame))  # .start()
    thread3 = Thread(target=save_pic, args=(out_queue,))  # .start()

    thread1.start()
    thread2.start()
    thread3.start()

    model = model.eval()

    for i in tqdm_:
        with torch.no_grad():
            y_true, _ = hq_queue.get()
            x, x_bicubic = lq_queue.get()

            x = F.pad(x, [0, padW, 0, padH])

            if not skip_model_testing:
                y_fake = model(x)
                y_fake = y_fake[:, :, :H_y, :W_y]
                if output_generated:
                    out_queue.put(y_fake)

            if not skip_model_testing:
                y_true = y_true.to(device)
                ssim_loss = ssim(y_fake, y_true).mean()
                lpips_loss = lpips_metric(y_fake, y_true).mean()

                ssim_ += [float(ssim_loss)]
                lpips_ += [float(lpips_loss)]

            if prev_gt is not None and not skip_model_testing:
                # compute tLP
                if test_tlp:
                    lp_gt = lpips_metric(prev_gt, y_true)
                    lp_sr = lpips_metric(prev_sr, y_fake)
                    tlp_step = abs(float(lp_gt - lp_sr))
                    tLP += [tlp_step]

            if test_lq:
                x = x[:, :, :H_x, :W_x]
                x_rescaled = F.interpolate(x, scale_factor=args.UPSCALE_FACTOR, mode='bicubic')
                ssim_loss_x = ssim(x_rescaled, y_true).mean()
                lpips_loss_x = lpips_metric(x_rescaled, y_true).mean()

                if prev_gt is not None:
                    prev_x = prev_x[:, :, :H_y, :W_y]

                    if test_tlp:
                        lp_gt = lpips_metric(prev_gt, y_true)
                        lp_x = lpips_metric(prev_x, x_rescaled)
                        tlp_step = abs(float(lp_gt - lp_x))
                        tLP_x += [tlp_step]

                ssim_x += [float(ssim_loss_x)]
                lpips_x += [float(lpips_loss_x)]

            prev_gt = y_true.clone()
            prev_x = F.interpolate(x.clone(), scale_factor=args.UPSCALE_FACTOR, mode='bicubic')
            if not skip_model_testing:
                prev_sr = y_fake.clone()


    finish = True
    out_queue.put(None)

    out_dict = {'vid': vid, 'encode_res': resolution_lq, 'dest_res': resolution_hq}

    out_dict['ssim'] = np.mean(ssim_)
    out_dict['lpips'] = np.mean(lpips_)
    out_dict['size'] = video_size
    out_dict['time'] = time_length

    print("Mean ssim:", np.mean(ssim_))
    print("Mean lpips:", np.mean(lpips_))
    if test_tlp:
        print("Mean tLP:", np.mean(tLP))
        out_dict['tLP'] = np.mean(tLP)

    if test_tof:
        print("Mean tOF:", np.mean(tOF))
        out_dict['tOF'] = np.mean(tOF)

    if test_lq:
        print("Mean ssim_encoded:", np.mean(ssim_x))
        print("Mean lpips_encoded:", np.mean(lpips_x))

        out_dict['ssim_encoded'] = np.mean(ssim_x)
        out_dict['lpips_encoded'] = np.mean(lpips_x)

        if test_tlp:
            out_dict['tLP_encoded'] = np.mean(tLP_x)
            print("Mean tLP H264:", np.mean(tLP_x))
        if test_tof:
            out_dict['tOF_encoded'] = np.mean(tOF_x)
            print("Mean tOF H264:", np.mean(tOF_x))

    from_minute = from_second // 60
    to_minute = to_second // 60

    from_second_ = from_second % 60
    to_second_ = to_second % 60

    if output_generated and not skip_model_testing:
        ffmpeg_command = f"ffmpeg -nostats -loglevel 0 -framerate {fps} -start_number 0 -i\
         {test_dir_prefix}/out/{video_prefix}_%d.png -crf 5  -c:v libx264 -r {fps} -pix_fmt yuv420p {dest_dir / 'output_testing.mp4 -y'}"
        print("Putting output images together.\n", ffmpeg_command)
        os.system(ffmpeg_command)

        ## test vmaf
        vmaf_command = f"./ffmpeg -nostats -loglevel 0\
            -r {fps} -i {dest_dir / (video_prefix + '.mp4')} \
            -r {fps} -i {dest_dir / 'output_testing.mp4'} \
            -ss 00:{from_minute}:{from_second_} -to 00:{to_minute}:{to_second_} \
            -lavfi '[0:v]setpts=PTS-STARTPTS[reference]; \
                [1:v]scale=-1:{resolution_hq}:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
                [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout' \
            -f null - | grep -i 'aggregateVMAF'"
        print(vmaf_command)
        out = os.popen(vmaf_command).read()

        # parse output
        aggregate_vmaf = float(out.split(" ")[2][len('aggregateVMAF="'):-1])

        print("VMAF: ", aggregate_vmaf)
        out_dict['vmaf'] = aggregate_vmaf

        shutil.rmtree(test_dir_prefix + "/out")
    if test_lq:
        vmaf_command = f"./ffmpeg -nostats -loglevel 0\
                -r {fps} -i {dest_dir / (video_prefix + '.mp4')} \
                -r {fps} -i {dest_dir / f'encoded{resolution_lq}CRF{crf_}' / (video_prefix + '.mp4')} \
                -ss 00:{from_minute}:{from_second_} -to 00:{to_minute}:{to_second_} \
                -lavfi '[0:v]setpts=PTS-STARTPTS[reference]; \
                    [1:v]scale=-1:{resolution_hq}:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
                    [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout' \
                -f null - | grep -i 'aggregateVMAF'"
        print(vmaf_command)
        out = os.popen(vmaf_command).read()
        # parse output
        aggregate_vmaf_x = float(out.split(" ")[2][len('aggregateVMAF="'):-1])

        print("VMAF base: ", aggregate_vmaf_x)
        out_dict['vmaf_encoded'] = aggregate_vmaf_x

    print("Test completed")

    return out_dict


if __name__ == '__main__':
    test_dir = Path(args.TEST_DIR)
    videos = [vid.strip(".y4m") for vid in os.listdir(test_dir) if vid.endswith('.y4m') and '1080' in vid]

    second_start = 0
    second_finish = 120  # test no more than the 2nd minutes - none of the test videos last so much

    for crf, filename in [
        (args.CRF, args.MODEL_NAME),
    ]:
        print(f"Testing CRF {crf}")
        output = []

        for i, vid in enumerate(videos):
            print(f"Testing: {vid}; {i + 1}/{len(videos)}")
            dict = evaluate_model(str(test_dir), video_prefix=vid, output_generated=True, filename=filename,
                                  from_second=second_start, test_lq=True, skip_model_testing=False,
                                  to_second=second_finish, crf=crf)
            output += [dict]

        df = pd.DataFrame(output)
        print(df.mean(axis=0, skipna=True))
        name = filename.strip(".pkl") + f"_{output[0]['encode_res']}_{output[0]['dest_res']}_TEST_CRF{crf}.csv"
        df.to_csv(name)
