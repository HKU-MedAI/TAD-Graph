"""CPU regressions for DenseGNN's train/eval dropout behavior."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from tadgraph.models.gnn import DenseGNN


def make_model(pooling="attention", layers=1):
    return DenseGNN(
        emb_dim=6, num_classes=3, initial_dim=8, number_of_layers=layers,
        pooling_type=pooling, atten_hidden_size=4,
    )


def make_graph():
    nodes = torch.arange(9)
    neighbors = (nodes + 1) % nodes.numel()
    return Data(
        x=torch.rand(9, 6),
        adj_t=SparseTensor(row=neighbors, col=nodes, sparse_sizes=(9, 9)),
        edge_attr=torch.full((9, 2), 0.25),
        batch=torch.zeros_like(nodes),
    )


def without_densegnn_dropout(function):
    # All other dropout rates, including in attention and preprocessing, are
    # left to the original implementation.
    dropout = F.dropout

    def disable_affected_rate(input, p=0.5, training=True, inplace=False):
        return dropout(input, p, training and p != 0.2, inplace)

    with patch.object(F, "dropout", side_effect=disable_affected_rate):
        return function()


class DenseGNNEvaluationDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def check_pooling(self, pooling, batch, reference):
        model = make_model(pooling).eval()
        features = torch.arange(1, 73, dtype=torch.float32).reshape(9, 8) / 72
        with torch.no_grad():
            expected = reference(model, features)
            actual = model.gnn_pool(features, batch)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                model.gnn_pool(features, batch), actual, rtol=0, atol=0,
            )

    def test_mean_pooling(self):
        batch = torch.tensor([0] * 4 + [1] * 5)
        self.check_pooling(
            "mean", batch,
            lambda _, x: torch.stack([x[batch == i].mean(dim=0) for i in range(2)]),
        )

    def test_add_pooling(self):
        batch = torch.tensor([0] * 4 + [1] * 5)
        self.check_pooling(
            "add", batch,
            lambda _, x: torch.stack([x[batch == i].sum(dim=0) for i in range(2)]),
        )

    def test_attention_pooling(self):
        def reference(model, features):
            logits, values = model.attention_net(features)
            weights = (logits - logits.max()).exp()
            return (weights * values).sum(dim=0, keepdim=True) / weights.sum()

        self.check_pooling("attention", torch.zeros(9, dtype=torch.long), reference)

    def test_multilayer_node_features_match_dropout_free_oracle(self):
        model = make_model("mean", layers=3).eval()
        graph = make_graph()
        with torch.no_grad():
            expected_nodes, expected_pooled = without_densegnn_dropout(
                lambda: model.gnn_forward(graph.clone()),
            )
            actual_nodes, actual_pooled = model.gnn_forward(graph.clone())
            # Node features are independent of pooling dropout and isolate
            # the dropout between convolution layers.
            self.assertEqual(actual_nodes.shape, (9, 8))
            torch.testing.assert_close(actual_nodes, expected_nodes)
            torch.testing.assert_close(actual_pooled, expected_pooled)

    def test_default_pooling_prediction_is_repeatable(self):
        model = make_model().eval()
        graph = make_graph()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            output = model(graph.clone())
            expected = without_densegnn_dropout(lambda: model(graph.clone()))
            repeated = model(graph.clone())

        embedding, logits, probabilities = output
        self.assertEqual(embedding.shape, (1, 8))
        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertTrue(probabilities.isfinite().all().item())
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))
        self.assertIn(probabilities.argmax(dim=1).item(), range(3))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def test_training_dropout_and_gradients_are_preserved(self):
        model = make_model("mean", layers=2).train()
        graph = make_graph()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        affected_calls = []
        dropout = F.dropout

        def record_dropout(input, p=0.5, training=True, inplace=False):
            output = dropout(input, p, training, inplace)
            if p == 0.2:
                affected_calls.append((tuple(input.shape), training, (input != 0) & (output == 0)))
            return output

        with patch.object(F, "dropout", side_effect=record_dropout):
            _, logits, _ = model(graph)
        self.assertEqual(
            [(shape, training) for shape, training, _ in affected_calls],
            [((1, 8), True), ((9, 8), True), ((1, 8), True)],
        )
        self.assertTrue(affected_calls[1][2].any().item())

        loss = F.cross_entropy(logits, torch.tensor([1]))
        self.assertTrue(loss.isfinite().item())
        loss.backward()
        for convolution in model.convs_list:
            gradient = convolution.lin_l.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(gradient.isfinite().all().item())
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        old_bias = model.pred_layer.bias.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(old_bias, model.pred_layer.bias))
