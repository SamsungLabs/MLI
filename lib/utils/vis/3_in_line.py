import argparse
import glob
import os
import pathlib

import PIL
import cv2
from PIL import Image, ImageFont
from tqdm import tqdm

from lib.utils.visualise import VideoWriter
import visual_utils

VIDEO_FORMATS = ['mp4', 'avi']
IMAGE_FORMATS = ['jpg', 'png', 'jpeg', 'bmp']


def image_or_video(path):
    ext = path.split('.')[-1].lower()

    if ext in VIDEO_FORMATS:
        return 'video'
    elif ext in IMAGE_FORMATS:
        return 'image'
    else:
        return None


def frame_3x_2small_1big(step_images, font, video_w, video_h, color=(0, 0, 0), font_color=(255, 255, 255)):
    tables = []
    # captions = ['result', 'source', 'reference']
    for i in range(3):
        tables.append(visual_utils.layout_2small_1big(step_images[i * 3: i * 3 + 3],
                                                      # captions=captions,
                                                      font=font,
                                                      pading=40,
                                                      result_size=(512, 768 + 80),
                                                      orientation='u',
                                                      color=color,
                                                      font_color=font_color))

    return visual_utils.black_with_windows(tables, result_size=(video_w, video_h), color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default=None,
                        help='Path to video data')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output video')
    parser.add_argument('--mode', type=str, default=None,
                        help='mode')
    parser.add_argument('--h', type=int, default=1080,
                        help='Result video height')
    parser.add_argument('--w', type=int, default=1920,
                        help='Result video width')
    parser.add_argument('--frame_size', type=int, default=400,
                        help='Frame size')
    parser.add_argument('--font_path', type=str, default=None,
                        help='Path to font')
    parser.add_argument('--font_size', type=int, default=15,
                        help='Frame size')
    parser.add_argument('--num_videos', type=int, default=1,
                        help='Num input videos')
    parser.add_argument('--captions',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=None,
                        help='Path to video data')

    opts = parser.parse_args()
    captions = opts.captions
    length = opts.num_videos
    mode = opts.mode
    # assert len(captions) == 3

    font = opts.font_path
    font_size = opts.font_size
    if font is not None:
        font = ImageFont.truetype(font, font_size)
    else:
        font = ImageFont.truetype(os.path.join(pathlib.Path(__file__).parent.absolute(), 'Roboto-Regular.ttf'),
                                  font_size)

    frame_size = opts.frame_size
    video_h = opts.h
    video_w = opts.w

    min_side = min(video_h, video_w)
    max_side = max(video_h, video_w)
    if frame_size > min_side:
        frame_size = min(min_side, max_side / length)

    output_path = opts.output_path
    if output_path is None:
        output_path = os.path.join(opts.path, 'result.mp4')

    video_paths = []
    video_sources = []
    num_frames = []
    for i in range(length):
        video_paths.append(glob.glob(os.path.join(opts.path, f'{i}.*'))[0])
        content_type = image_or_video(video_paths[-1])

        assert content_type is not None, f"{video_paths[-1]} is not video or image."

        if content_type == 'video':
            video_sources.append(cv2.VideoCapture(video_paths[-1]))
            num_frames.append(int(video_sources[-1].get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            video_sources.append(Image.open(video_paths[-1]))

    num_frames = min(num_frames)

    video_writer = VideoWriter(output_path)
    result_frames = []
    for i in tqdm(range(num_frames)):
        step_images = []

        print(f"Num video sources {len(video_sources)}")
        for video_source in video_sources:
            if isinstance(video_source, cv2.VideoCapture):
                _, image = video_source.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            else:
                image = video_source
            step_images.append(visual_utils.process_image(image, center_crop='max'))
            # step_images.append(visual_utils.process_image(image, center_crop=450))

        # # Comparison 3 images
        # # --------------------------------------
        # # captions = ['Ours', '3d Inpainting', 'SynSin']
        if mode == 'compare3':
            frame = visual_utils.black_with_windows(step_images,
                                                    # captions=captions,
                                                    font=font,
                                                    result_size=(video_w, video_h),
                                                    color=(255, 255, 255))
        # # --------------------------------------

        # # Teaser  slide (1-2)
        # # --------------------------------------
        if mode == 'teaser':
            frame = frame_3x_2small_1big(step_images,
                                         font,
                                         video_w,
                                         video_h,
                                         color=(255, 255, 255),
                                         font_color=(0, 0, 0))
        # --------------------------------------


        # # Grid
        # --------------------------------------
        if mode == 'grid':
            frame = visual_utils.images_grid(step_images,
                                             row_size=7,
                                             row_padding=10,
                                             column_padding=10,
                                             images_size=256,
                                             resize=False,
                                             color=(255, 255, 255))
            frame = visual_utils.black_with_windows([frame], result_size=(video_w, video_h), color=(255, 255, 255))
        # --------------------------------------


        # # Grid 4 * 3
        # --------------------------------------
        if mode == 'grid4x3':
            frame = visual_utils.images_grid(step_images,
                                             row_size=4,
                                             row_padding=40,
                                             column_padding=100,
                                             images_size=256,
                                             resize=False,
                                             color=(255, 255, 255))
            frame = visual_utils.black_with_windows([frame], result_size=(video_w, video_h), color=(255, 255, 255))

        # frame = visual_utils.images_grid(step_images,
        #                                  row_size=4,
        #                                  row_padding=10,
        #                                  column_padding=10,
        #                                  images_size=450,
        #                                  resize=False,
        #                                  color=(255, 255, 255))
        frame = visual_utils.black_with_windows([frame], result_size=(video_w, video_h), color=(255, 255, 255))
        # --------------------------------------


        result_frames.append(frame)

    video_writer.process_pil_list(result_frames)
    video_writer.finalize()


if __name__ == '__main__':
    main()
