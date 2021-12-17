from cv2 import cv2
import cupy
import time
import model
import video_colorizer
import numpy as np

use_gpu = True

video = cv2.VideoCapture("../colorization/imgs/input.mp4")
writer = cv2.VideoWriter("output/colorise.avi", cv2.VideoWriter_fourcc(*"MJPG"), 24, (960, 540))
colorizer = model.load_model(pretrained=True).eval()
temporal = video_colorizer.load_model(pretrained=True).eval()


if (use_gpu):
    colorizer.cuda()
    temporal.cuda()

progress = 0

history = None
while True:
    read, frame = video.read()
    print(read)
    if read:
        if progress % 100 == 0:
            start = time.time()
        (tens_l_orig, tens_l_rs) = video_colorizer.preprocess_img(cupy.asarray(frame), HW=(256, 256))
        if progress % 100 == 0:
            half = time.time()
            print(half - start)
        if (use_gpu):
            tens_l_rs = tens_l_rs.cuda()
            tens_l_orig = tens_l_orig.cuda()
        if progress % 100 == 0:
            thrf = time.time()
            print(thrf - half)

        current = colorizer(tens_l_rs)
        if history is None:
            out_img = video_colorizer.postprocess_tens(tens_l_orig,current )
        else:
            new = temporal(tens_l_rs,history)
            # figure out if new needs to be accepted or not
            out_img = video_colorizer.postprocess_tens(tens_l_orig, new)
        history = current


        if progress % 100 == 0:
            end = time.time()
            print(end - thrf)
        out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img.get(), cv2.COLOR_RGB2BGR)
        print(progress)
        progress += 1
        writer.write(out_img)

    else:
        break

video.release()
