"""Generate images from text with a trained HyperCGAN, optionally beyond the training
field of view (extrapolation) or at a higher resolution (super-resolution)."""

import json
import os
import re
from typing import List, Union

import click
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from torch_utils import misc
from training.text_encoder import CLIPTextEncoder

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Parse a comma separated list of numbers or ranges, e.g. '1,2,5-10'."""
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def load_generator(network_pkl: str, device: torch.device, reload_code: bool=True):
    """Loads G_ema from a training snapshot. With `reload_code`, the network is re-instantiated
    from the code in this repository (and the weights copied), so that the latest generator code
    (e.g. extrapolation / super-resolution support) is used regardless of the code stored in the pickle."""
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    if reload_code:
        G_new = dnnlib.util.construct_class_by_name(class_name='training.networks.Generator', **G.init_kwargs).eval().requires_grad_(False)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new
    return G.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------

def decode_tokens(tokens) -> str:
    """CLIP token ids -> text (for datasets that only store tokens)."""
    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    special = {tokenizer.encoder['<|startoftext|>'], tokenizer.encoder['<|endoftext|>'], 0}
    return tokenizer.decode([int(t) for t in tokens if int(t) not in special]).strip()

def sample_dataset_captions(data: str, num: int, seed: int):
    """Returns `num` (caption, tokens) pairs: a random caption of a random image of the dataset."""
    dataset = dnnlib.util.construct_class_by_name(class_name='training.dataset.ImageFolderDataset', path=data, use_clip=True)
    rnd = np.random.RandomState(seed)
    has_captions = True
    out = []
    for idx in rnd.randint(len(dataset), size=num):
        all_tokens = dataset.get_all_tkn_features(idx)
        j = rnd.randint(len(all_tokens))
        tokens = np.array(all_tokens[j], dtype=np.int64)
        caption = None
        if has_captions:
            try:
                caption = dataset.get_captions(idx)[j]
            except IOError:
                has_captions = False
        if caption is None:
            caption = decode_tokens(tokens)
        out.append((caption, tokens))
    return out

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename or URL', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--prompt', 'prompts', help='Text prompt (can be given multiple times)', multiple=True)
@click.option('--prompts-file', help='Text file with one prompt per line', type=str, metavar='FILE')
@click.option('--data', help='Instead of prompts, use random captions of this dataset (zip)', type=str, metavar='PATH')
@click.option('--num', help='Number of captions to sample from --data', type=int, default=100, show_default=True)
@click.option('--seeds', help='Random seeds (e.g. 0,1,4-6); one image per prompt and seed', type=parse_range, default='0', show_default=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1.0, show_default=True)
@click.option('--extrapolate', help='Field-of-view multiplier, e.g. 1.5 renders a 1.5x larger area around the image (input.resolution * value must be an integer)', type=float, default=1.0, show_default=True)
@click.option('--superres', help='Super-resolution factor, e.g. 4 renders 1024x1024 images with a 256x256 model', type=float, default=1.0, show_default=True)
@click.option('--superres-block-res', help='Block at which the resolution is increased: 4 evaluates the whole INR on the denser grid [default: input resolution of the generator, i.e. 32]', type=int, metavar='INT')
@click.option('--batch', 'batch_size', help='Batch size', type=int, default=8, show_default=True)
@click.option('--reload-code', help='Re-instantiate the generator with the code of this repository', type=bool, default=True, show_default=True)
def generate_images(network_pkl, outdir, prompts, prompts_file, data, num, seeds, truncation_psi, extrapolate, superres, superres_block_res, batch_size, reload_code):
    """Generate images from text with a pretrained HyperCGAN.

    \b
    # Text-to-image.
    python generate.py --network=network-snapshot.pkl --outdir=out --seeds=0-3 \\
        --prompt="this bird has a red crown and a short black beak"

    \b
    # Extrapolation: render 1.5x the training field of view (384x384 for a 256x256 model).
    python generate.py --network=network-snapshot.pkl --outdir=out --extrapolate=1.5 \\
        --prompt="a living room with a couch and a television"

    \b
    # Super-resolution: render 1024x1024 images with a 256x256 model.
    python generate.py --network=network-snapshot.pkl --outdir=out --superres=4 \\
        --prompt="a painting of a calm lake at sunset"

    \b
    # Extrapolation with 2x field of view, rendered at 2x pixel density (1024x1024).
    python generate.py --network=network-snapshot.pkl --outdir=out --extrapolate=2 \\
        --superres=2 --superres-block-res=4 --prompt="a living room with a couch and a television"

    \b
    # One image per random test caption (e.g. to compute metrics with external tools).
    python generate.py --network=network-snapshot.pkl --outdir=out --data=datasets/cub_test.zip --num=30000
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_generator(network_pkl, device, reload_code=reload_code)
    text_encoder = CLIPTextEncoder(device=device)
    os.makedirs(outdir, exist_ok=True)

    # Collect (caption, tokens) pairs and seeds.
    prompts = list(prompts)
    if prompts_file is not None:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts += [line.strip() for line in f if line.strip() != '']
    if data is not None:
        if len(prompts) > 0:
            raise click.ClickException('Use either --prompt/--prompts-file or --data, not both.')
        jobs = [(caption, tokens, seed) for seed, (caption, tokens) in enumerate(sample_dataset_captions(data, num, seeds[0]), start=seeds[0])]
    else:
        if len(prompts) == 0:
            raise click.ClickException('Specify --prompt, --prompts-file or --data.')
        prompt_tokens = CLIPTextEncoder.tokenize(prompts).numpy()
        jobs = [(prompt, tokens, seed) for prompt, tokens in zip(prompts, prompt_tokens) for seed in seeds]

    # By default, super-resolution is applied at the last block that runs at the input resolution.
    synthesis_kwargs = dict(noise_mode='const', coord_scale=extrapolate)
    if superres != 1:
        if superres_block_res is None:
            superres_block_res = G.synthesis.cfg.input.resolution
        synthesis_kwargs.update(upsampling_factor=superres, upsampling_block_res=superres_block_res)

    # Generate.
    metadata = {}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        tkn_fts = torch.from_numpy(np.stack([tokens for _, tokens, _ in batch])).to(device)
        _txt_fts, wrd_fts = text_encoder.encode(tkn_fts)
        z = torch.from_numpy(np.concatenate([np.random.RandomState(seed).randn(1, G.z_dim) for _, _, seed in batch])).to(device)
        img = G(z=z, wrd_fts=wrd_fts, tkn_fts=tkn_fts, truncation_psi=truncation_psi, **synthesis_kwargs)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for i, (caption, _, seed) in enumerate(batch):
            fname = f'{start + i:06d}_seed{seed:04d}.png'
            PIL.Image.fromarray(img[i], 'RGB').save(os.path.join(outdir, fname))
            metadata[fname] = caption
        print(f'Generated {min(start + batch_size, len(jobs))}/{len(jobs)} images', end='\r')
    print()

    with open(os.path.join(outdir, 'captions.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved {len(jobs)} images of size {img.shape[1]}x{img.shape[2]} to {outdir}')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
