# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by LAFITE and HyperCGAN: stores image captions as CLIP tokens.

"""Convert an image-caption dataset into a HyperCGAN dataset archive."""

import functools
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import clip
import numpy as np
import PIL.Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    return os.path.splitext(str(fname))[1].lower() in PIL.Image.EXTENSION # type: ignore

def parse_captions(text: str):
    return [line.strip() for line in text.splitlines() if line.strip() != '']

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    if max_images is not None:
        input_images = input_images[:max_images]

    def iterate_images():
        for fname in input_images:
            rel_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
            txt_fname = os.path.splitext(fname)[0] + '.txt'
            try:
                img = PIL.Image.open(fname)
                img.load()
                with open(txt_fname, 'r', encoding='utf-8') as f:
                    captions = parse_captions(f.read())
            except Exception as err: # pylint: disable=broad-except
                print(f'Skipping {rel_fname}: {err}')
                continue
            yield dict(img=img, captions=captions, source_fname=rel_fname)
    return len(input_images), iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        all_names = set(z.namelist())
        input_images = [str(f) for f in sorted(all_names) if is_image_ext(f)]
    if max_images is not None:
        input_images = input_images[:max_images]

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for fname in input_images:
                txt_fname = os.path.splitext(fname)[0] + '.txt'
                try:
                    with z.open(fname, 'r') as f:
                        img = PIL.Image.open(f)
                        img.load()
                    with z.open(txt_fname, 'r') as f:
                        captions = parse_captions(f.read().decode('utf-8'))
                except Exception as err: # pylint: disable=broad-except
                    print(f'Skipping {fname}: {err}')
                    continue
                yield dict(img=img, captions=captions, source_fname=fname)
    return len(input_images), iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source) and source.lower().endswith('.zip'):
        return open_image_zip(source, max_images=max_images)
    error(f'--source must be a directory or a .zip archive: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    if dest.lower().endswith('.zip'):
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or zip with images and same-named .txt caption files', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int]
):
    """Convert an image-caption dataset into a dataset archive usable with HyperCGAN.

    The source is a directory (or zip) where every image has a caption file with the
    same name and a .txt extension, containing one caption per line:

    \b
    source/
        some/sub/dir/image_0001.jpg
        some/sub/dir/image_0001.txt
        ...

    Sub-directories are allowed. CLIP R-precision (clip_rprecision.py) treats the
    parent directory of each source image as its class, e.g. keep the
    `001.Black_footed_Albatross/` folders of CUB to sample negatives from other classes.

    The output contains uncompressed PNGs and a `dataset.json` with:

    \b
    {
        "clip_token_features": [["00000/img00000000.png", [[49406, ..., 0], ...]], ...],  # CLIP tokens, 77 per caption
        "captions":            [["00000/img00000000.png", ["a small bird ...", ...]], ...],
        "source_fnames":       [["00000/img00000000.png", "some/sub/dir/image_0001.jpg"], ...],
        "labels": null
    }

    Captions longer than CLIP's context (77 tokens) are truncated. Grayscale images are
    converted to RGB. Images without captions are skipped. Example:

    \b
    python dataset_tool.py --source=~/data/cub/train --dest=~/datasets/cub_train.zip \\
        --transform=center-crop --width=256 --height=256
    """
    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None
    clip_token_features = []
    captions_list = []
    source_fnames = []

    for image in tqdm(input_iter, total=num_files):
        if len(image['captions']) == 0:
            print(f"Skipping {image['source_fname']}: no captions")
            continue

        idx = len(clip_token_features)
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(np.array(image['img'].convert('RGB')))

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across the whole dataset.
        cur_image_attrs = dict(width=img.shape[1], height=img.shape[0], channels=img.shape[2])
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            error(f"Image {image['source_fname']} attributes must be equal across all images of the dataset.  Got:\n" + '\n'.join(err))

        # Tokenize the captions.
        tokens = clip.tokenize(image['captions'], truncate=True).numpy().tolist()

        # Save the image as an uncompressed PNG.
        image_bits = io.BytesIO()
        PIL.Image.fromarray(img, 'RGB').save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())

        clip_token_features.append([archive_fname, tokens])
        captions_list.append([archive_fname, image['captions']])
        source_fnames.append([archive_fname, image['source_fname']])

    metadata = {
        'labels': None,
        'clip_token_features': clip_token_features,
        'captions': captions_list,
        'source_fnames': source_fnames,
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()
    print(f'Saved {len(clip_token_features)} of {num_files} images.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
