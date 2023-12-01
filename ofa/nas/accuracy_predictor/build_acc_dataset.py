import horovod.torch as hvd
import torch
from acc_dataset import AccuracyDataset
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig

hvd.init()
ofa_network = OFAMobileNetV3(
                n_classes=1000,
                bn_param=(0.99, 1e-5),
                dropout_rate=0,
                width_mult=1.0,
                ks_list=[7],
                expand_ratio_list=[3, 4, 6],
                depth_list=[2, 3, 4],
            )

ofa_network.load_state_dict(torch.load('/home/akhare39/aditya/wsn_exps/IN_MBV3_Twarmup_150/270_ID_RW_SUM_SNetWarmup/maxnet/checkpoint/model_best.pth.tar', map_location='cpu')['state_dict'])

ofa_network.cuda()
run_config = DistributedImageNetRunConfig("acc_dataset", dataset="imagenet", test_batch_size=256,
                                            valid_size=None, image_size=224,
                                            num_replicas=1, rank=0, cifar_mode=None)


run_config.beta_decay = 0.9
run_config.n_epochs = 10
run_config.num_networks_per_batch = 4

run_manager = DistributedRunManager(
        "./",
        ofa_network,
        run_config,
        hvd_compression=hvd.Compression.none,
        backward_steps=1,
        is_root=(hvd.rank() == 0),
        init=False,
    )


accuracy_dataset = AccuracyDataset(path='/home/akhare39/aditya/once-for-all/ofa/nas/accuracy_predictor')

accuracy_dataset.build_acc_dataset(run_manager, ofa_network, n_arch=1000, image_size_list=[224])
