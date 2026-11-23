"""CLIP R-precision of a trained HyperCGAN.

For each sample, an image is generated from a ground-truth test caption and CLIP ViT-B/32 ranks
the ground-truth caption against 99 captions of images from other classes. R@1 is the fraction of
samples where the ground-truth caption is ranked first. We report the mean and std over 10 splits.
"""

import json
import os

import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

import dnnlib
from generate import load_generator
from training.text_encoder import CLIPTextEncoder

#----------------------------------------------------------------------------

def load_classes(dataset):
    """Class id of every image: the parent directory of the source image if the dataset was created
    with dataset_tool.py from a class-structured folder (e.g. CUB), otherwise every image is its own class."""
    try:
        with dataset._open_file(dataset.json_name) as f: # pylint: disable=protected-access
            source_fnames = json.load(f).get('source_fnames')
    except Exception: # pylint: disable=broad-except
        source_fnames = None
    if source_fnames is None:
        return np.arange(dataset._raw_shape[0]) # pylint: disable=protected-access
    source_fnames = dict(source_fnames)
    dirs = [os.path.dirname(source_fnames[fname]) for fname in dataset._image_fnames] # pylint: disable=protected-access
    if len(set(dirs)) == 1:
        return np.arange(len(dirs))
    names = {d: i for i, d in enumerate(sorted(set(dirs)))}
    return np.array([names[d] for d in dirs])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename or URL', required=True)
@click.option('--data', help='Test dataset (zip created with dataset_tool.py)', required=True, metavar='PATH')
@click.option('--num', 'num_samples', help='Number of generated images', type=int, default=30000, show_default=True)
@click.option('--num-negatives', help='Number of negative captions per sample', type=int, default=99, show_default=True)
@click.option('--splits', 'num_splits', help='Number of splits for mean/std', type=int, default=10, show_default=True)
@click.option('--batch', 'batch_size', help='Batch size', type=int, default=16, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=1234, show_default=True)
def clip_rprecision(network_pkl, data, num_samples, num_negatives, num_splits, batch_size, seed):
    """Compute CLIP R-precision (R@1).

    \b
    python clip_rprecision.py --network=network-snapshot.pkl --data=datasets/cub_test.zip
    """
    rnd = np.random.RandomState(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_generator(network_pkl, device)
    text_encoder = CLIPTextEncoder(device=device)

    dataset = dnnlib.util.construct_class_by_name(class_name='training.dataset.ImageFolderDataset', path=data, use_clip=True)
    num_images = dataset._raw_shape[0] # pylint: disable=protected-access
    classes = load_classes(dataset)
    print(f'{num_images} images, {len(np.unique(classes))} classes')

    def random_caption(idx):
        all_tokens = dataset.get_all_tkn_features(idx)
        return all_tokens[rnd.randint(len(all_tokens))]

    def sample_negatives(class_id):
        negatives = []
        while len(negatives) < num_negatives:
            idx = rnd.randint(num_images)
            if classes[idx] != class_id:
                negatives.append(random_caption(idx))
        return negatives

    scores = []
    sample_indices = rnd.choice(num_images, num_samples, replace=True)
    for start in tqdm(range(0, num_samples, batch_size), desc='CLIP R@1'):
        indices = sample_indices[start:start + batch_size]
        # Ground-truth caption first, then the negatives.
        tokens = [[random_caption(idx)] + sample_negatives(classes[idx]) for idx in indices]
        tokens = torch.tensor(np.array(tokens), dtype=torch.long, device=device) # [b, 1 + num_negatives, 77]
        b, n, _ = tokens.shape

        txt_fts, wrd_fts = text_encoder.encode(tokens.view(b * n, -1))
        txt_fts = txt_fts.view(b, n, -1)
        wrd_fts = wrd_fts.view(b, n, *wrd_fts.shape[1:])[:, 0]

        with torch.no_grad():
            z = torch.randn([b, G.z_dim], device=device)
            img = G(z=z, wrd_fts=wrd_fts, tkn_fts=tokens[:, 0], noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            img = torch.stack([text_encoder.preprocess(PIL.Image.fromarray(x)) for x in img]).to(device)
            img_fts = text_encoder.model.encode_image(img)
            img_fts = img_fts / img_fts.norm(dim=-1, keepdim=True)

        sim = torch.einsum('bd,bnd->bn', img_fts.float(), txt_fts.float()) # [b, 1 + num_negatives]
        scores.append((sim.argmax(dim=1) == 0).float().cpu().numpy())

    scores = np.concatenate(scores)
    rnd.shuffle(scores)
    split_means = [split.mean() for split in np.array_split(scores, num_splits)]
    print(f'CLIP R@1: {np.mean(split_means):.4f} +- {np.std(split_means):.4f} ({num_samples} samples, {num_splits} splits)')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    clip_rprecision() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
