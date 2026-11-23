# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by LAFITE and HyperCGAN.

"""Train HyperCGAN: text-to-continuous-image generation with hypernetwork-modulated INR-GANs."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from omegaconf import OmegaConf
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # Model config.
    cfg_file   = None, # Generator config (YAML): <path>, default = configs/hypercgan.yaml
    map_num    = None, # Number of mapping network layers: <int>, default = 8

    # Contrastive losses (LAFITE).
    f_dim      = None, # Dimensionality of the discriminator's projected image features: <int>, default = 512
    d_use_norm = None, # L2-normalize the discriminator's projected image features: <bool>, default = False
    d_use_fts  = None, # Use the discriminator's projected image features: <bool>, default = True
    lam        = None, # Weighting of the two softmax directions in the contrastive loss: <float>, default = 0
    temp       = None, # Temperature of the contrastive loss: <float>, default = 0.5
    gather     = None, # Gather negatives across GPUs: <bool>, default = False
    itd        = None, # Weight of the discriminator-feature/text contrastive loss: <float>, default = 5
    itc        = None, # Weight of the CLIP-image/text contrastive loss: <float>, default = 10

    # General options.
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    metric_only_test = None, # Compute real-image statistics on --test_data instead of --data: <bool>, default = False
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    data       = None, # Training dataset (required): <path>
    test_data  = None, # Dataset whose captions are used for metrics: <path>, default = same as --data
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    gamma      = None, # R1 gamma: <float>, default = 10
    kimg       = None, # Training duration: <int>, default = 25000
    batch      = None, # Total batch size: <int>, default = depends on resolution and GPU count

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'noaug' (default), 'ada', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = 0.6
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options.
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # Model config
    # ------------------------------------------
    if cfg_file is None:
        cfg_file = os.path.join(REPO_DIR, 'configs', 'hypercgan.yaml')
    if not os.path.isfile(cfg_file):
        raise UserError(f'--cfg_file: {cfg_file} does not exist')
    model_cfg = OmegaConf.load(cfg_file)
    cfg_name = os.path.splitext(os.path.basename(cfg_file))[0]

    if map_num is None:
        map_num = 8

    # ------------------------------------------
    # Contrastive losses
    # ------------------------------------------
    args.f_dim = 512 if f_dim is None else f_dim
    args.d_use_norm = False if d_use_norm is None else d_use_norm
    args.d_use_fts = True if d_use_fts is None else d_use_fts
    args.lam = 0. if lam is None else lam
    args.temp = 0.5 if temp is None else temp
    args.gather = False if gather is None else gather
    args.itd = 5. if itd is None else itd
    args.itc = 10. if itc is None else itc

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics
    args.metric_only_test = False if metric_only_test is None else metric_only_test

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, test_data, subset, mirror
    # -----------------------------------
    assert data is not None
    assert isinstance(data, str)
    if test_data is None:
        test_data = data
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=False, max_size=None, xflip=False, use_clip=True)
    args.testing_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=test_data, use_labels=False, max_size=None, xflip=False, use_clip=True)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=False, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        args.testing_set_kwargs.resolution = training_set.resolution
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True
        args.testing_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: gamma, kimg, batch
    # ------------------------------------
    desc += f'-{cfg_name}-gpus{gpus:d}'
    res = args.training_set_kwargs.resolution
    spec = dnnlib.EasyDict(
        kimg  = 25000,
        mb    = max(min(gpus * min(4096 // res, 32), 64), gpus), # keep gpu memory consumption at bay
        fmaps = 1 if res >= 512 else 0.5,
        lrate = 0.002 if res >= 1024 else 0.0025,
        gamma = 10.,
        ramp  = 0.05,
    )
    spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
    spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', use_norm=args.d_use_norm, use_fts=args.d_use_fts, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = map_num
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    # Interpolations such as ${coords} are relative to the generator section, so re-root it before resolving.
    args.G_kwargs.cfg = OmegaConf.to_container(OmegaConf.create(OmegaConf.to_container(model_cfg.generator)), resolve=True)
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.D_kwargs.epilogue_kwargs.f_dim = args.f_dim

    if model_cfg.generator.get('fp32', False):
        args.G_kwargs.synthesis_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = None

    if model_cfg.get('discriminator', {}).get('fp32', False):
        args.D_kwargs.num_fp16_res = 0
        args.D_kwargs.conv_clamp = None

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    # Note: path length regularization (pl_weight=2) and style mixing (prob=0.9) use the StyleGAN2Loss defaults.
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        args.loss_kwargs.r1_gamma = gamma
    desc += f'-gamma{args.loss_kwargs.r1_gamma:g}'

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        args.batch_size = batch
        args.batch_gpu = batch // gpus
    desc += f'-batch{args.batch_size}'

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------
    if aug is None:
        aug = 'noaug'
    assert isinstance(aug, str)
    desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6
    elif aug == 'noaug':
        pass
    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------
    assert resume is None or isinstance(resume, str)
    if resume is None or resume == 'noresume':
        resume = 'noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# Model config.
@click.option('--cfg_file', help='Generator config [default: configs/hypercgan.yaml]', metavar='YAML')
@click.option('--map_num', help='Number of mapping network layers [default: 8]', type=int, metavar='INT')

# Contrastive losses.
@click.option('--f_dim', help='Dimensionality of the discriminator image features [default: 512]', type=int, metavar='INT')
@click.option('--d_use_norm', help='L2-normalize the discriminator image features [default: false]', type=bool, metavar='BOOL')
@click.option('--d_use_fts', help='Use the discriminator image features for the contrastive loss [default: true]', type=bool, metavar='BOOL')
@click.option('--gather', help='Gather negative samples across GPUs [default: false]', type=bool, metavar='BOOL')
@click.option('--lam', help='Softmax-direction weighting of the contrastive loss [default: 0]', type=float)
@click.option('--temp', help='Temperature of the contrastive loss [default: 0.5]', type=float)
@click.option('--itd', help='Weight of the discriminator/text contrastive loss [default: 5]', type=float)
@click.option('--itc', help='Weight of the CLIP-image/text contrastive loss [default: 10]', type=float)

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--metric_only_test', help='Compute real-image metric statistics on --test_data [default: false]', type=bool, metavar='BOOL')
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--data', help='Training data (directory or zip created with dataset_tool.py)', metavar='PATH', required=True)
@click.option('--test_data', help='Data whose captions are used for metrics [default: same as --data]', metavar='PATH')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--gamma', help='R1 gamma [default: 10]', type=float)
@click.option('--kimg', help='Training duration in thousands of images [default: 25000]', type=int, metavar='INT')
@click.option('--batch', help='Total batch size [default: depends on resolution and #GPUs]', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: noaug]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# Transfer learning.
@click.option('--resume', help='Resume training from a network pickle [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train HyperCGAN.

    Examples (the settings used in the paper, see scripts/):

    \b
    # CUB, 256x256, 8 GPUs, with ADA.
    python train.py --outdir=training-runs --data=datasets/cub_train.zip \\
        --gpus=8 --batch=128 --mirror=1 --aug=ada --augpipe=bg

    \b
    # MS-COCO, 256x256, 8 GPUs.
    python train.py --outdir=training-runs --data=datasets/coco_train.zip \\
        --gpus=8 --batch=128 --mirror=1
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Metric captions:    {args.testing_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
