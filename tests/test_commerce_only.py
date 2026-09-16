from src.config_loader import ModeConfig


def test_commerce_only_is_safe_default():
    assert ModeConfig().commerce_only is True
