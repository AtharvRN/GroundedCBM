import torch

from gcbm.target_batches import batch_targets_to_device, pad_sparse_targets


def test_pad_sparse_targets_handles_empty_batch_entries():
    indices, masks, valid = pad_sparse_targets(
        [torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)],
        [torch.zeros((0, 3, 4)), torch.zeros((0, 3, 4))],
        mask_h=3,
        mask_w=4,
    )
    assert indices.shape == (2, 1)
    assert masks.shape == (2, 1, 3, 4)
    assert valid.shape == (2, 1)
    assert not valid.any()
    assert torch.all(indices == -1)


def test_pad_sparse_targets_preserves_variable_length_masks():
    indices, masks, valid = pad_sparse_targets(
        [torch.tensor([2, 4]), torch.tensor([1])],
        [torch.ones(2, 2, 2), torch.full((1, 2, 2), 3.0)],
        mask_h=2,
        mask_w=2,
    )
    assert indices.tolist() == [[2, 4], [1, -1]]
    assert valid.tolist() == [[True, True], [True, False]]
    assert torch.allclose(masks[0, 1], torch.ones(2, 2))
    assert torch.allclose(masks[1, 0], torch.full((2, 2), 3.0))


def test_batch_targets_to_device_returns_four_target_tensors():
    batch = {
        "global_targets": torch.zeros(2, 3),
        "mask_indices": torch.zeros(2, 1, dtype=torch.long),
        "mask_targets": torch.zeros(2, 1, 4, 4),
        "mask_valid": torch.zeros(2, 1, dtype=torch.bool),
    }
    moved = batch_targets_to_device(batch, "cpu")
    assert len(moved) == 4
    assert all(tensor.device.type == "cpu" for tensor in moved)
