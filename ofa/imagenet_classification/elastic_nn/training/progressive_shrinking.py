# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import (
    DistributedMetric,
    list_mean,
    subset_mean,
    val2list,
    MyRandomResizedCrop,
    profile,
)
import numpy as np

from ofa.imagenet_classification.run_manager import DistributedRunManager
import horovod.torch as hvd

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
    "load_models",
    "train_elastic_depth",
    "train_elastic_expand",
    "train_elastic_width_mult",
]


def validate(
    run_manager,
    epoch=0,
    is_test=False,
    image_size_list=None,
    ks_list=None,
    expand_ratio_list=None,
    depth_list=None,
    width_mult_list=None,
    additional_setting=None,
):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = val2list(run_manager.run_config.data_provider.image_size, 1)
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if width_mult_list is None:
        if "width_mult_list" in dynamic_net.__dict__:
            width_mult_list = list(range(len(dynamic_net.width_mult_list)))
        else:
            width_mult_list = [0]

    subnet_settings = []
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for w in width_mult_list:
                    for img_size in image_size_list:
                        subnet_settings.append(
                            [
                                {
                                    "image_size": img_size,
                                    "d": d,
                                    "e": e,
                                    "ks": k,
                                    "w": w,
                                },
                                "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                            ]
                        )
    if additional_setting is not None:
        subnet_settings += additional_setting

    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ""
    for setting, name in subnet_settings:
        run_manager.write_log(
            "-" * 30 + " Validate %s " % name + "-" * 30, "train", should_print=False
        )
        run_manager.run_config.data_provider.assign_active_img_size(
            setting.pop("image_size")
        )
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, "train", should_print=False)

        run_manager.reset_running_statistics(dynamic_net)
        loss, (top1, top5) = run_manager.validate(
            epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net
        )
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        valid_log += "%s (%.3f), " % (name, top1)

    return (
        list_mean(losses_of_subnets),
        list_mean(top1_of_subnets),
        list_mean(top5_of_subnets),
        valid_log,
    )


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):

    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()

    epoch_flops = []
    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i in range(nBatch):
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)

            subnet_str = ""
            minibatch_flops = []
            for _ in range(args.dynamic_batch_size):
                # set random seed before sampling
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                subnet_settings = dynamic_net.sample_active_subnet()
                subnet_str += (
                    "%d: " % _
                    + ",".join(
                        [
                            "%s_%s"
                            % (
                                key,
                                "%.1f" % subset_mean(val, 0)
                                if isinstance(val, list)
                                else val,
                            )
                            for key, val in subnet_settings.items()
                        ]
                    )
                    + " || "
                )

                sampled_net = dynamic_net.get_active_subnet()
                flops, _ = profile(
                    sampled_net, (1, 3, 224, 224), custom_ops=None
                )
                minibatch_flops.append(flops / 1e6)
            epoch_flops.append(np.array(minibatch_flops))

            t.set_postfix(
                {
                    "R": 224,
                    "loss_type": "ce",
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                }
            )
            t.update(1)
            end = time.time()

    if hvd.rank() == 0:
        run_manager.experiment_flops.append(np.array(epoch_flops))
        np.save('ps_depth_1_flops.npy', np.array(run_manager.experiment_flops))

    return -1, (-1, -1)


def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)
    if validate_func is None:
        validate_func = validate

    for epoch in range(
        run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
        )

def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")


def train_elastic_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["depth_list"] = sorted(dynamic_net.depth_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        # validate after loading weights
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Depth: %s -> %s"
        % (depth_stage_list[: current_stage + 1], depth_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    # add depth list constraints
    if (
        len(set(dynamic_net.ks_list)) == 1
        and len(set(dynamic_net.expand_ratio_list)) == 1
    ):
        validate_func_dict["depth_list"] = depth_stage_list
    else:
        validate_func_dict["depth_list"] = sorted(
            {min(depth_stage_list), max(depth_stage_list)}
        )

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )


def train_elastic_expand(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.resume:
        validate_func_dict["expand_ratio_list"] = sorted(dynamic_net.expand_ratio_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Expand Ratio: %s -> %s"
        % (
            expand_stage_list[: current_stage + 1],
            expand_stage_list[: current_stage + 2],
        )
        + "-" * 30,
        "valid",
    )
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
        validate_func_dict["expand_ratio_list"] = expand_stage_list
    else:
        validate_func_dict["expand_ratio_list"] = sorted(
            {min(expand_stage_list), max(expand_stage_list)}
        )

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )


def train_elastic_width_mult(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    width_stage_list = dynamic_net.width_mult_list.copy()
    width_stage_list.sort(reverse=True)
    n_stages = len(width_stage_list) - 1
    current_stage = n_stages - 1

    if run_manager.start_epoch == 0 and not args.resume:
        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        if current_stage == 0:
            dynamic_net.re_organize_middle_weights(
                expand_ratio_stage=len(dynamic_net.expand_ratio_list) - 1
            )
            run_manager.write_log(
                "reorganize_middle_weights (expand_ratio_stage=%d)"
                % (len(dynamic_net.expand_ratio_list) - 1),
                "valid",
            )
            try:
                dynamic_net.re_organize_outer_weights()
                run_manager.write_log("reorganize_outer_weights", "valid")
            except Exception:
                pass
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert args.resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Width Mult: %s -> %s"
        % (width_stage_list[: current_stage + 1], width_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    validate_func_dict["width_mult_list"] = sorted({0, len(width_stage_list) - 1})

    # train
    train_func(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )
