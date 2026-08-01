"""Unit test suite for Phase 3 Audio NN Building Block Layers and Encoders."""

import pytest
import torch

from app.audio.models.aasist.encoder import RawNetEncoder
from app.audio.models.layers.attention import SelfAttention1D, SqueezeExcitation
from app.audio.models.layers.conv import Conv1DBlock, Conv2DBlock, SincConv
from app.audio.models.layers.residual import ResidualBlock1D, ResidualBlock2D


class TestConvLayers:
    def test_conv1d_block_shape(self):
        block = Conv1DBlock(in_channels=1, out_channels=32, kernel_size=3)
        x = torch.randn(2, 1, 1600)
        out = block(x)
        assert out.shape == (2, 32, 1600)

    def test_conv2d_block_shape(self):
        block = Conv2DBlock(in_channels=1, out_channels=16, pool_size=(2, 2))
        x = torch.randn(2, 1, 128, 100)
        out = block(x)
        assert out.shape == (2, 16, 64, 50)

    def test_sinc_conv_shape(self):
        sinc = SincConv(out_channels=64, kernel_size=251, sample_rate=16000)
        x = torch.randn(2, 1, 16000)
        out = sinc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64


class TestResidualLayers:
    def test_residual_block_1d_same_dim(self):
        res = ResidualBlock1D(in_channels=32, out_channels=32, downsample=False)
        x = torch.randn(2, 32, 500)
        out = res(x)
        assert out.shape == (2, 32, 500)

    def test_residual_block_1d_downsample(self):
        res = ResidualBlock1D(in_channels=32, out_channels=64, downsample=True)
        x = torch.randn(2, 32, 500)
        out = res(x)
        assert out.shape == (2, 64, 250)

    def test_residual_block_2d_shape(self):
        res = ResidualBlock2D(in_channels=16, out_channels=32, stride=2)
        x = torch.randn(2, 16, 64, 64)
        out = res(x)
        assert out.shape == (2, 32, 32, 32)


class TestAttentionLayers:
    def test_squeeze_excitation_shape(self):
        se = SqueezeExcitation(channels=64, reduction=8)
        x = torch.randn(2, 64, 300)
        out = se(x)
        assert out.shape == (2, 64, 300)

    def test_self_attention_1d_shape(self):
        attn = SelfAttention1D(embed_dim=64, num_heads=4)
        x = torch.randn(2, 64, 100)
        out = attn(x)
        assert out.shape == (2, 64, 100)


class TestRawNetEncoder:
    def test_rawnet_encoder_output_shape(self):
        encoder = RawNetEncoder(
            in_channels=1,
            sinc_channels=32,
            res_channels=[32, 32, 64, 64],
            embedding_dim=128,
        )
        x = torch.randn(2, 1, 16000)
        embedding = encoder(x)
        assert embedding.shape == (2, 128)

    def test_rawnet_encoder_gradient_flow(self):
        encoder = RawNetEncoder(
            in_channels=1,
            sinc_channels=16,
            res_channels=[16, 16],
            embedding_dim=64,
        )
        x = torch.randn(2, 1, 8000, requires_grad=True)
        embedding = encoder(x)
        loss = embedding.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
