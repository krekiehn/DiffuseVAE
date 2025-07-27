"""Tests for the Optuna sweep configuration."""

from hydra import compose, initialize


def test_sweep_config_loads():
    """Ensure the sweep configuration can be parsed by Hydra."""
    with initialize(version_base=None, config_path="../main/configs"):
        cfg = compose(config_name="sweep_ae.yaml", return_hydra_config=True)
        assert cfg.hydra.sweeper is not None
        assert "dataset" in cfg
