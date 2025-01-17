#
"""
Unit tests for conv networks.
"""
import unittest

import torch
from torch import nn

from texar.torch.core.layers import MergeLayer
from texar.torch.modules.networks.conv_networks import Conv1DNetwork


class Conv1DNetworkTest(unittest.TestCase):
    """Tests :class:`~texar.torch.modules.Conv1DNetwork` class.
    """

    def test_feedforward(self):
        """Tests feed forward.
        """
        inputs_1 = torch.ones([128, 32, 300])
        network_1 = Conv1DNetwork(in_channels=inputs_1.shape[1],
                                  in_features=inputs_1.shape[2])
        # dense layers are not constructed yet
        self.assertEqual(len(network_1.layers), 4)
        self.assertIsInstance(
            network_1.layer_by_name("MergeLayer"), MergeLayer)
        for layer in network_1.layers[0].layers:
            self.assertIsInstance(layer, nn.Sequential)

        outputs_1 = network_1(inputs_1)
        self.assertEqual(outputs_1.shape, torch.Size([128, 256]))
        self.assertEqual(outputs_1.size(-1), network_1.output_size)

        inputs_2 = torch.ones([128, 64, 300])
        hparams = {
            # Conv layers
            "num_conv_layers": 2,
            "out_channels": 128,
            "kernel_size": [[3, 4, 5], 4],
            "other_conv_kwargs": {"padding": 0},
            # Pooling layers
            "pooling": "AvgPool1d",
            "pool_size": 2,
            "pool_stride": 1,
            # Dense layers
            "num_dense_layers": 3,
            "out_features": [128, 128, 10],
            "dense_activation": "ReLU",
            "other_dense_kwargs": None,
            # Dropout
            "dropout_conv": [0, 1, 2],
            "dropout_dense": 2
        }

        network_2 = Conv1DNetwork(in_channels=inputs_2.shape[1],
                                  in_features=inputs_2.shape[2],
                                  hparams=hparams)
        # dropout-merge-dropout-(Sequential(Conv, ReLU))-avgpool-dropout-
        # flatten-(Sequential(Linear,ReLU))-(Sequential(Linear,ReLU))-dropout
        # -linear
        self.assertEqual(len(network_2.layers), 1 + 1 + 1 + 3 + 4 + 1)
        self.assertIsInstance(
            network_2.layer_by_name("MergeLayer"), MergeLayer)
        for layer in network_2.layers[1].layers:
            self.assertIsInstance(layer, nn.Sequential)

        outputs_2 = network_2(inputs_2)
        self.assertEqual(outputs_2.shape, torch.Size([128, 10]))
        self.assertEqual(outputs_2.size(-1), network_2.output_size)
        # test whether concatenation happens along channel dim when feature dim
        # has been reduced
        hparams = {
            # Conv layers
            "num_conv_layers": 1,
            "out_channels": 128,
            "kernel_size": [3, 4, 5],
            "other_conv_kwargs": {"padding": 0},
            # Pooling layers
            "pooling": "AvgPool1d",
            "pool_size": None,
            "pool_stride": 1,
            # Dense layers
            "num_dense_layers": 0,
            "out_features": [],
            "dense_activation": "ReLU",
            "other_dense_kwargs": None,
            # Dropout
            "dropout_conv": [],
            "dropout_dense": []
        }

        network_3 = Conv1DNetwork(in_channels=inputs_2.shape[1],
                                  in_features=inputs_2.shape[2],
                                  hparams=hparams)
        inputs_3 = inputs_2
        outputs_3 = network_3(inputs_3)
        num_of_kernels = len(hparams["kernel_size"])
        out_channels = hparams["out_channels"]
        self.assertEqual(outputs_3.shape,
                         torch.Size([128, num_of_kernels * out_channels]))
        self.assertEqual(outputs_3.size(-1), network_3.output_size)


if __name__ == "__main__":
    unittest.main()
