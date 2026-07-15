"""Checkpoint key normalization for released AutoMoT weights."""


def normalize_automot_checkpoint_key(key: str) -> str:
    return key


def normalize_automot_state_dict(state_dict):
    return dict(state_dict)
