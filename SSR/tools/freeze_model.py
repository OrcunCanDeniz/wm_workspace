# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.append('')
import argparse
import mmcv
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Build model from config and checkpoint')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def build_model_from_config(config_path, checkpoint_path, fuse_conv_bn=False, cfg_options=None, seed=0, deterministic=False):
    """
    Build model from config and checkpoint without dataset and optimization components.
    
    Args:
        config_path (str): Path to config file
        checkpoint_path (str): Path to checkpoint file
        fuse_conv_bn (bool): Whether to fuse conv and bn layers
        cfg_options (dict): Additional config options to override
        seed (int): Random seed
        deterministic (bool): Whether to set deterministic options for CUDNN
    
    Returns:
        model: Built and loaded model
    """
    # Load config
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    
    # Import modules from string list
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # Import modules from plugin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing plugin module: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config_path)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing config module: {_module_path}")
                plg_lib = importlib.import_module(_module_path)

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Set random seeds
    if seed is not None:
        set_random_seed(seed, deterministic=deterministic)

    # Build the model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Handle fp16 model
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Fuse conv-bn if requested
    if fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # Set model classes from checkpoint or default
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    # Set palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    
    print(f"Model built successfully from {config_path}")
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    return model
#model.pts_bbox_head.transformer.encoder

def main():
    args = parse_args()
    
    # Build model
    model = build_model_from_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        fuse_conv_bn=args.fuse_conv_bn,
        cfg_options=args.cfg_options,
        seed=args.seed,
        deterministic=args.deterministic
    )
    
    print("Model building completed successfully!")
    return model

if __name__ == '__main__':
    main()
