"""Unit test suite for production AASIST model architecture."""

import pytest
import torch

from app.audio.models.aasist.graph.adjacency import AdjacencyMatrixBuilder
from app.audio.models.aasist.graph.graph_builder import GraphBuilder
from app.audio.models.aasist.graph.node_features import NodeFeatureGenerator
from app.audio.models.aasist.model import AASIST
from app.audio.models.aasist.modules.backend import AASISTBackEnd, GraphAttentionLayer
from app.audio.models.aasist.modules.frontend import AASISTFrontEnd
from app.audio.models.aasist.modules.fusion import FeatureFusion
from app.audio.registry.model_registry import model_registry


class TestAASISTGraphSubsystem:
    def test_node_feature_generator_shape(self):
        gen = NodeFeatureGenerator(in_dim=128, node_dim=64, normalize=True)
        x = torch.randn(2, 128, 50)
        nodes = gen(x)
        assert nodes.shape == (2, 50, 64)
        # Check L2 norm equals 1.0
        norms = torch.norm(nodes, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_adjacency_matrix_builder_shape(self):
        builder = AdjacencyMatrixBuilder(metric="cosine")
        nodes = torch.randn(2, 20, 64)
        adj = builder(nodes)
        assert adj.shape == (2, 20, 20)
        # Check rows sum to 1.0 (Softmax normalized)
        row_sums = adj.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_graph_builder(self):
        gb = GraphBuilder(in_dim=128, node_dim=64)
        features = torch.randn(2, 128, 30)
        nodes, adj = gb(features)
        assert nodes.shape == (2, 30, 64)
        assert adj.shape == (2, 30, 30)


class TestAASISTModules:
    def test_frontend_shape(self):
        frontend = AASISTFrontEnd(in_channels=1, sinc_channels=16, embedding_dim=64)
        audio = torch.randn(2, 1, 8000)
        feat = frontend(audio)
        assert feat.shape == (2, 64)

    def test_backend_shape(self):
        backend = AASISTBackEnd(node_dim=64, num_classes=2)
        nodes = torch.randn(2, 10, 64)
        adj = torch.eye(10).unsqueeze(0).repeat(2, 1, 1)
        logits = backend(nodes, adj)
        assert logits.shape == (2, 2)

    def test_fusion_shape(self):
        fusion = FeatureFusion(feature_dim=64)
        f_feat = torch.randn(2, 64)
        g_feat = torch.randn(2, 64)
        fused = fusion(f_feat, g_feat)
        assert fused.shape == (2, 64)


class TestAASISTProductionModel:
    @pytest.mark.parametrize("batch_size", [1, 8, 16])
    def test_aasist_batch_sizes(self, batch_size):
        model = AASIST(num_classes=2, node_dim=32)
        audio = torch.randn(batch_size, 1, 8000)
        logits = model(audio)
        assert logits.shape == (batch_size, 2)

    def test_aasist_gradient_flow(self):
        model = AASIST(num_classes=2, node_dim=32)
        audio = torch.randn(2, 1, 8000, requires_grad=True)
        logits = model(audio)
        loss = logits.sum()
        loss.backward()
        assert audio.grad is not None
        assert not torch.isnan(audio.grad).any()

    def test_aasist_registry_instantiation(self):
        model_cls = model_registry.get("aasist")
        model = model_cls(num_classes=2)
        assert isinstance(model, AASIST)
        assert model.get_num_parameters() > 0

    def test_aasist_device_cpu(self):
        device = torch.device("cpu")
        model = AASIST(num_classes=2, node_dim=32).to(device)
        audio = torch.randn(2, 1, 4000, device=device)
        logits = model(audio)
        assert logits.device == device
        assert logits.shape == (2, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aasist_cuda_amp(self):
        device = torch.device("cuda")
        model = AASIST(num_classes=2, node_dim=32).to(device)
        audio = torch.randn(2, 1, 4000, device=device)
        with torch.amp.autocast("cuda"):
            logits = model(audio)
        assert logits.shape == (2, 2)
