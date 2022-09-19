from __future__ import print_function

import math

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from PIL import ImageDraw


def process_image(image, center_crop=None, scale_factor=None, resize=None):
    width, height = image.size
    if center_crop:
        if center_crop == 'max':
            center_crop = min(height, width)
        image = TF.center_crop(image, center_crop)

    if scale_factor is not None:
        width = int(image.size[0] * scale_factor)
        height = int(image.size[1] * scale_factor)
        dim = (height, width)
        image = TF.resize(image, dim)
    elif resize is not None and (width != resize[0]) and (height != resize[1]):
        image = TF.resize(image, size=[resize[1], resize[0]])

    return image


def frames_in_frame(main_frame, sub_frames, frames_pos, frames_sizes, frame_width=2):
    '''
    Receive main frame and several additional frames, draw additional frames above main frame,

    :param main_frame: main frame
    :param sub_frames: additional frames
    :param frames_pos: position of additional
    :param frames_sizes: additional frame positions
    :return:
    '''
    result_h, result_w = main_frame.size

    new_im = Image.new('RGB', main_frame.size)
    draw = ImageDraw.Draw(new_im)
    new_im.paste(main_frame, (0, 0))

    for window_frame, window_pos, window_size in zip(sub_frames, frames_pos, frames_sizes):
        window_img_resized = TF.resize(window_frame, window_size)

        window_pos = list(window_pos)
        if window_pos[0] < 0:
            window_pos[0] = result_h + window_pos[0] - window_img_resized.size[0]
        if window_pos[1] < 0:
            window_pos[1] = result_w + window_pos[1] - window_img_resized.size[1]

        new_im.paste(window_img_resized, window_pos)  # PIL CHANGES INPLACE window_pos, IT ADDS FINAL RECT POINT.

        if frame_width > 0:
            draw.rectangle(window_pos, width=frame_width)

    return new_im


def black_with_windows(frames, captions=None, font=None, result_size=(1280, 720),
                       color=(0, 0, 0), font_color=(255, 255, 255)):
    '''

    Draw several frames with captions on solid color background.

    :param frames: frames
    :param captions: frame captions
    :param font: captions font
    :param result_size: result image size
    :return:
    '''
    new_im = Image.new('RGB', result_size, tuple(color))
    draw = ImageDraw.Draw(new_im)

    captions_on = False
    if font and captions:
        if len(captions) == len(frames):
            captions_on = True

    overall_frames_size_x = 0
    overall_frames_size_y = 0

    for frame in frames:
        overall_frames_size_x += frame.size[0]
        overall_frames_size_y = max(overall_frames_size_y, frame.size[1])

    padding_x_size = int((result_size[0] - overall_frames_size_x) / (len(frames) + 1))
    padding_y_size = int((result_size[1] - overall_frames_size_y) / 2)

    frame_pos_x = padding_x_size
    frame_pos_y = padding_y_size

    for i, frame in enumerate(frames):
        new_im.paste(frame, (frame_pos_x, frame_pos_y))

        if captions_on:
            w, h = draw.textsize(captions[i], font=font)
            draw.text((frame_pos_x + frame.size[0] / 2 - w / 2, frame_pos_y + frame.size[1]),
                      captions[i], fill=font_color, align='center', font=font)

        frame_pos_x += padding_x_size + frame.size[0]

    return new_im


def layout_2small_1big_sq(frames,
                          captions=None,
                          font=None,
                          pading=30,
                          result_size=(778, 532),
                          orientation='l',
                          color=(0, 0, 0),
                          font_color=(255, 255, 255)):
    assert len(frames) == 3, f"{len(frames)}"

    new_im = Image.new('RGB', result_size, color=color)
    draw = ImageDraw.Draw(new_im)

    frames_data = [{}, {}, {}]

    if orientation in ['l', 'r']:
        small_size = int((result_size[1] - 2 * pading) / 2)
        big_size = min(result_size[0] - (pading + small_size), result_size[1] - pading)

        if orientation == 'l':
            frames_data[0]['pos'] = (0, 0)
            frames_data[1]['pos'] = (big_size + pading, 0)
            frames_data[2]['pos'] = (big_size + pading, small_size + pading)
        elif orientation == 'r':
            frames_data[0]['pos'] = (small_size + pading, 0)
            frames_data[1]['pos'] = (0, 0)
            frames_data[2]['pos'] = (0, small_size + pading)

    else:
        small_size = int((result_size[0] - pading) / 2)
        big_size = min(result_size[1] - (pading * 2 + small_size), result_size[0])

        if orientation == 'u':
            frames_data[0]['pos'] = (0, 0)
            frames_data[1]['pos'] = (0, big_size + pading)
            frames_data[2]['pos'] = (small_size + pading, big_size + pading)
        elif orientation == 'd':
            frames_data[0]['pos'] = (0, small_size + pading)
            frames_data[1]['pos'] = (0, 0)
            frames_data[2]['pos'] = (small_size + pading, 0)

    frames_data[0]['frame'] = process_image(frames[0], center_crop='max', resize=[big_size, big_size])
    frames_data[1]['frame'] = process_image(frames[1], center_crop='max', resize=[small_size, small_size])
    frames_data[2]['frame'] = process_image(frames[2], center_crop='max', resize=[small_size, small_size])

    if font and captions:
        if len(captions) > 2:
            for i, caption in enumerate(captions[:3]):
                frames_data[i]['caption'] = caption

    for i, frame_data in enumerate(frames_data):
        new_im.paste(frame_data['frame'], (frame_data['pos'][0], frame_data['pos'][1]))

        if 'caption' in frame_data:
            w, h = draw.textsize(frame_data['caption'], font=font)
            draw.text((frame_data['pos'][0] + frame_data['frame'].size[0] / 2 - w / 2,
                       frame_data['pos'][1] + frame_data['frame'].size[1]),
                      frame_data['caption'],
                      fill=font_color,
                      align='center',
                      font=font)

    return new_im


def paired_images_grid(images1,
                       images2,
                       row_size=5,
                       images_padding=0,
                       row_padding=5,
                       column_padding=0,
                       images_size=256,
                       resize=False,
                       color=(0, 0, 0)):
    '''
    Get two image sequence and draw paired grid.

    :param images1:
    :param images2:
    :param row_size:
    :param images_padding:
    :param row_padding:
    :param column_padding:
    :param images_size:
    :param resize:
    :return:
    '''
    grid_len = min(len(images1), len(images2))
    rows_num = int(math.ceil(grid_len / row_size))

    if resize:
        images1 = [TF.center_crop(TF.resize(img, images_size), images_size) for img in images1]
        images2 = [TF.center_crop(TF.resize(img, images_size), images_size) for img in images2]
    else:
        images1 = [TF.resize(img, images_size) for img in images1]
        images2 = [TF.resize(img, images_size) for img in images2]

    result_image_height = 2 * rows_num * images_size + (rows_num - 1) * row_padding + images_padding * rows_num
    result_image_width = images_size * row_size

    new_im = Image.new('RGB', (result_image_width, result_image_height), color=color)
    new_im.paste(Image.fromarray(
        np.ones([result_image_height, result_image_width, 3], dtype=np.uint8) * np.array(color, dtype=np.uint8)),
        (0, 0))

    image_pos_y = 0
    for i in range(rows_num):
        image_pos_x = 0
        for j in range(row_size):
            grid_pos = row_size * i + j
            if grid_pos < grid_len:
                new_im.paste(images1[grid_pos], (image_pos_x, image_pos_y))
                new_im.paste(images2[grid_pos], (image_pos_x, image_pos_y + images_size + images_padding))
                image_pos_x += images_size + column_padding

        image_pos_y += images_size * 2 + row_padding + images_padding

    return new_im


def images_grid(images_sequence,
                row_size=None,
                row_padding=5,
                column_padding=0,
                frame_size=256,
                central_crop=False,
                color=[0, 0, 0]
                ):
    grid_len = len(images_sequence)
    if row_size is None:
        row_size = grid_len
    rows_num = int(math.ceil(grid_len / row_size))
    if isinstance(frame_size, int):
        frame_size = [frame_size, frame_size]

    if central_crop:
        images_sequence = [process_image(img, center_crop='max') for img in images_sequence]
    images_sequence = [TF.resize(img, frame_size) for img in images_sequence]

    result_image_height = rows_num * frame_size[0] + (rows_num - 1) * row_padding
    result_image_width = frame_size[1] * row_size + (row_size - 1) * column_padding

    new_im = Image.new('RGB', (result_image_width, result_image_height), color=tuple(color))
    new_im.paste(Image.fromarray(
        np.ones([result_image_width, result_image_height, 3], dtype=np.uint8) * np.array(color, dtype=np.uint8)),
        (0, 0))

    image_pos_y = 0
    for i in range(rows_num):
        image_pos_x = 0
        for j in range(row_size):
            grid_pos = row_size * i + j
            if grid_pos < grid_len:
                new_im.paste(images_sequence[grid_pos], (image_pos_x, image_pos_y))
                image_pos_x += frame_size[1] + column_padding

        image_pos_y += frame_size[0] + row_padding

    return new_im


def layout_2small_1big(frames,
                       padding=30,
                       orientation='l',
                       color=[0, 0, 0],
                       ):
    assert len(frames) == 3, f"{len(frames)}"

    big_w, big_h = frames[0].size
    frames_data = [{}, {}, {}]

    if orientation in ['l', 'r']:
        small_h = (big_h - padding) // 2
        small_w = int(max(frames[1].size[0] * (small_h / frames[1].size[1]),
                          frames[2].size[0] * (small_h / frames[2].size[1])))
        new_im = Image.new('RGB', (big_w + small_w + padding, big_h), color=tuple(color))

        if orientation == 'l':
            frames_data[0]['pos'] = (0, 0)
            frames_data[1]['pos'] = (big_w + padding, 0)
            frames_data[2]['pos'] = (big_w + padding, small_h + padding)
        elif orientation == 'r':
            frames_data[0]['pos'] = (small_w + padding, 0)
            frames_data[1]['pos'] = (0, 0)
            frames_data[2]['pos'] = (0, small_h + padding)

    else:
        small_w = (big_w - padding) // 2
        small_h = int(max(frames[1].size[1] * (small_w / frames[1].size[0]),
                          frames[2].size[1] * (small_w / frames[1].size[0])))
        new_im = Image.new('RGB', (big_w, big_h + small_h + padding), color=tuple(color))

        if orientation == 'u':
            frames_data[0]['pos'] = (0, 0)
            frames_data[1]['pos'] = (0, big_h + padding)
            frames_data[2]['pos'] = (small_w + padding, big_h + padding)
        elif orientation == 'd':
            frames_data[0]['pos'] = (0, small_h + padding)
            frames_data[1]['pos'] = (0, 0)
            frames_data[2]['pos'] = (small_w + padding, 0)

    frames_data[0]['frame'] = frames[0]
    frames_data[1]['frame'] = process_image(frames[1], resize=[small_w, small_h])
    frames_data[2]['frame'] = process_image(frames[2], resize=[small_w, small_h])

    for i, frame_data in enumerate(frames_data):
        new_im.paste(frame_data['frame'], (frame_data['pos'][0], frame_data['pos'][1]))

    return new_im


def two_blocks(images_sequence,
               row1_size,
               row1_layout,
               row2_layout,
               padding=10,
               color=[0, 0, 0],
               orientation='l',
               ):
    row1_frame = globals()[row1_layout['mode']](images_sequence[:row1_size], **row1_layout['params'])
    row2_frame = globals()[row2_layout['mode']](images_sequence[row1_size:], **row2_layout['params'])

    row1_w, row1_h = row1_frame.size
    row2_w, row2_h = row2_frame.size
    block_data = [{}, {}]

    if orientation in ['l', 'r']:
        result_image_height = max(row1_h, row2_h)
        result_image_width = row1_w + row2_w + padding
        row1_pos_y = (result_image_height - row1_h) // 2
        row2_pos_y = (result_image_height - row2_h) // 2

        if orientation == 'l':
            block_data[0]['pos'] = (0, row1_pos_y)
            block_data[1]['pos'] = (row1_w + padding, row2_pos_y)

        elif orientation == 'r':
            block_data[0]['pos'] = (row2_w + padding, row1_pos_y)
            block_data[1]['pos'] = (0, row2_pos_y)

    else:
        result_image_height = row1_h + row2_h + padding
        result_image_width = max(row1_w, row2_w)
        row1_pos_x = (result_image_width - row1_w) // 2
        row2_pos_x = (result_image_width - row2_w) // 2

        if orientation == 'u':
            block_data[0]['pos'] = (row1_pos_x, 0)
            block_data[1]['pos'] = (row2_pos_x, row1_h + padding)

        elif orientation == 'd':
            block_data[0]['pos'] = (row1_pos_x, row2_h + padding)
            block_data[1]['pos'] = (row2_pos_x, 0)

    block_data[0]['frame'] = row1_frame
    block_data[1]['frame'] = row2_frame

    new_im = Image.new('RGB', (result_image_width, result_image_height), color=tuple(color))
    new_im.paste(Image.fromarray(
        np.ones([result_image_width, result_image_height, 3], dtype=np.uint8) * np.array(color, dtype=np.uint8)),
        (0, 0))

    for i, block in enumerate(block_data):
        new_im.paste(block['frame'], (block['pos'][0], block['pos'][1]))
    # new_im.paste(row1_frame, (row1_pos_x, row1_pos_y))
    # new_im.paste(row2_frame, (row2_pos_x, row2_pos_y))

    return new_im


def base_frame(frame,
               padding_up=0,
               padding_down=0,
               padding_left=0,
               padding_right=0,
               result_size=(1280, 720),
               color=(0, 0, 0)
               ):
    '''
    Draw several frames with captions on solid color background.

    :param frames: frames
    :param captions: frame captions
    :param font: captions font
    :param result_size: result image size
    :return:
    '''

    new_im = Image.new('RGB', result_size, tuple(color))
    frame_w, frame_h = frame.size
    scale = min((result_size[0] - padding_left - padding_right) / frame_w,
                (result_size[1] - padding_up - padding_down) / frame_h)
    frame = process_image(frame, scale_factor=scale)

    frame_w, frame_h = frame.size
    frame_pos_x = (result_size[0] - padding_left - padding_right - frame_w) // 2 + padding_left
    frame_pos_y = (result_size[1] - padding_up - padding_down - frame_h) // 2 + padding_up
    new_im.paste(frame, (frame_pos_x, frame_pos_y))

    return new_im


def draw_arrow(start, end, size, imageDraw, width=2):
    cos = 0.866
    sin = 0.250
    line_color = (0, 0, 0)
    end1 = [int(end[0] + (size[0] * cos + size[1] * -sin)),
            int(end[1] + (size[0] * sin + size[1] * cos))]
    end2 = [int(end[0] + (size[0] * cos + size[1] * sin)),
            int(end[1] + (size[0] * -sin + size[1] * cos))]

    imageDraw.line([tuple(start), tuple(end)], fill=line_color, width=width)
    imageDraw.polygon([tuple(end), tuple(end1), tuple(end2)], fill=line_color, outline=line_color)
    imageDraw.line([tuple(end), tuple(end1)], fill=line_color, width=width)
    imageDraw.line([tuple(end), tuple(end2)], fill=line_color, width=width)


def images_sequences_with_arrow(images_sequences, source_col, left_img, right_img,
                                caption_left='', caption_right='', font=None, row_size=5,
                                row_padding=5, column_padding=5,
                                images_size=256, resize=True):
    '''
    Get images sequence and draw grid.

    :param images_sequence:
    :param images2:
    :param row_size:
    :param images_padding:
    :param row_padding:
    :param column_padding:
    :param images_size:
    :param resize:
    :return:
    '''
    source_col_pad_scale = 5

    rows_num = len(images_sequences)

    result_image_height = (rows_num + 1) * images_size + (rows_num) * row_padding
    result_image_width = images_size * row_size + column_padding * (row_size - 1 + source_col_pad_scale)

    new_im = Image.new('RGB', (result_image_width, result_image_height))
    new_im.paste(Image.fromarray(np.ones([result_image_height, result_image_width, 3], dtype=np.uint8) * 255), (0, 0))

    if resize:
        left_img_resized, right_img_resized = [TF.center_crop(TF.resize(img, images_size), images_size) for img in
                                               [left_img, right_img]]
    else:
        left_img_resized, right_img_resized = [TF.resize(img, images_size) for img in [left_img, right_img]]

    new_im.paste(left_img_resized, (images_size + column_padding * source_col_pad_scale, 0))
    new_im.paste(right_img_resized, (result_image_width - images_size - column_padding, 0))

    imageDraw = ImageDraw.Draw(new_im)

    draw_arrow([result_image_width - int(2.5 * images_size / 2), int(1.5 * images_size / 2)],
               [int(4.5 * images_size / 2) + column_padding * source_col_pad_scale,
                int(1.5 * images_size / 2)],
               [50, 0], imageDraw, width=5)
    draw_arrow([int(4.5 * images_size / 2) + column_padding * source_col_pad_scale,
                int(1.5 * images_size / 2)],
               [result_image_width - int(2.5 * images_size / 2), int(1.5 * images_size / 2)],
               [-50, 0], imageDraw, width=5)

    if font:
        w, h = imageDraw.textsize(caption_left, font=font)
        imageDraw.text((int(4.5 * images_size / 2) + column_padding * source_col_pad_scale,
                        int(images_size / 2) - h),
                       caption_left, (0, 0, 0), font=font)

        w, h = imageDraw.textsize(caption_left, font=font)
        imageDraw.text((result_image_width - int(2.5 * images_size / 2) - w,
                        int(images_size / 2) - h),
                       caption_right, (0, 0, 0), font=font)

    image_pos_y = images_size + row_padding
    for image_sequence, source_frame in zip(images_sequences, source_col):
        skip_rate = int(math.ceil(len(image_sequence) / row_size))

        images_sequence_procesed = []
        if resize:
            image_sequence_procesed = [TF.center_crop(TF.resize(img, images_size), images_size) for img in
                                       [source_frame] + image_sequence[::skip_rate]]
        else:
            image_sequence_procesed = [TF.resize(img, images_size) for img in
                                       [source_frame] + image_sequence[::skip_rate]]

        image_pos_x = 0
        for i, frame in enumerate(image_sequence_procesed):
            new_im.paste(frame, (image_pos_x, image_pos_y))

            if i != 0:
                image_pos_x += images_size + column_padding
            else:
                image_pos_x += images_size + column_padding * source_col_pad_scale

        image_pos_y += images_size + row_padding

    return new_im


def image_grid_frame(images_sequence, frame_width=1280, frame_height=720, row_size=4,
                     row_padding=5, column_padding=0,
                     images_size=256, resize=False):
    '''
    Get images sequence and draw grid.

    :param images_sequence:
    :param images2:
    :param row_size:
    :param images_padding:
    :param row_padding:
    :param column_padding:
    :param images_size:
    :param resize:
    :return:
    '''

    grid_len = len(images_sequence)
    rows_num = int(math.ceil(grid_len / row_size))

    if resize:
        images_sequence = [TF.center_crop(TF.resize(img, images_size), images_size) for img in images_sequence]
    else:
        images_sequence = [TF.resize(img, images_size) for img in images_sequence]

    result_image_height = rows_num * images_size + (rows_num - 1) * row_padding
    result_image_width = images_size * row_size + (row_size - 1) * column_padding

    new_im = Image.new('RGB', (frame_width, frame_height))
    new_im.paste(Image.fromarray(np.ones([frame_height, frame_width, 3], dtype=np.uint8) * 0), (0, 0))

    global_pad_y = int((frame_height - result_image_height) / 2)
    global_pad_x = int((frame_width - result_image_width) / 2)

    image_pos_y = global_pad_y
    for i in range(rows_num):
        image_pos_x = global_pad_x
        for j in range(row_size):
            grid_pos = row_size * i + j
            if grid_pos < grid_len:
                new_im.paste(images_sequence[grid_pos], (image_pos_x, image_pos_y))
                image_pos_x += images_size + column_padding

        image_pos_y += images_size + row_padding

    return new_im


def image_frame(image, frame_width=1280, frame_height=720, images_size=256):
    '''
    Get images sequence and draw grid.

    :param images_sequence:
    :param images2:
    :param row_size:
    :param images_padding:
    :param row_padding:
    :param column_padding:
    :param images_size:
    :param resize:
    :return:
    '''

    result_image_width, result_image_height = image.size

    new_im = Image.new('RGB', (frame_width, frame_height))
    new_im.paste(Image.fromarray(np.ones([frame_height, frame_width, 3], dtype=np.uint8) * 0), (0, 0))

    global_pad_y = int((frame_height - result_image_height) / 2)
    global_pad_x = int((frame_width - result_image_width) / 2)

    image_pos_y = global_pad_y
    image_pos_x = global_pad_x

    new_im.paste(image, (image_pos_x, image_pos_y))

    return new_im
