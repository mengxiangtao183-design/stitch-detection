import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest

def _validate_paths(rgb_dir):
    if isinstance(rgb_dir, (list, tuple)):
        for path in rgb_dir:
            assert os.path.exists(path), f"Path not found: {path}"
    else:
        assert os.path.exists(rgb_dir), f"Path not found: {rgb_dir}"

def get_training_data(
    rgb_dir,
    img_options,
    input_subdir='input',
    target_subdir='target',
    edge_subdir=None,
    max_pairs_per_root=0,
    manifest_path=None,
    manifest_data_root=None,
):
    if not manifest_path:
        _validate_paths(rgb_dir)
    return DataLoaderTrain(
        rgb_dir,
        img_options,
        input_subdir=input_subdir,
        target_subdir=target_subdir,
        edge_subdir=edge_subdir,
        max_pairs_per_root=max_pairs_per_root,
        manifest_path=manifest_path,
        manifest_data_root=manifest_data_root,
    )

def get_validation_data(
    rgb_dir,
    img_options,
    input_subdir='input',
    target_subdir='target',
    max_pairs_per_root=0,
    manifest_path=None,
    manifest_data_root=None,
):
    if not manifest_path:
        _validate_paths(rgb_dir)
    return DataLoaderVal(
        rgb_dir,
        img_options,
        input_subdir=input_subdir,
        target_subdir=target_subdir,
        max_pairs_per_root=max_pairs_per_root,
        manifest_path=manifest_path,
        manifest_data_root=manifest_data_root,
    )

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
