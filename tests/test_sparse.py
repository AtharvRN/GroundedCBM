import torch

from gcbm.sparse import threshold_weight_truncation


def test_threshold_weight_truncation_is_global_not_per_class():
    weight = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0, 3.0],
            [2.0, 1.0, 0.5, 0.25],
        ]
    )

    truncated = threshold_weight_truncation(weight, sparsity=0.25)

    expected = torch.tensor(
        [
            [10.0, 9.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(truncated, expected)


def test_threshold_weight_truncation_keeps_average_nec_budget():
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    truncated = threshold_weight_truncation(weight, sparsity=0.5)

    assert int((truncated != 0).sum()) == 6
    assert torch.equal(truncated[truncated != 0], torch.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
