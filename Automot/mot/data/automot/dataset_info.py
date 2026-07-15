import os

from .pdm_lite_bev_encoder_dataset import PdmLiteBEVEncoderDataset


DATASET_REGISTRY = {
    "automot_pdm_lite": PdmLiteBEVEncoderDataset,
}

_PDM_DATA_DIR = os.environ.get("PDM_DATA_DIR", "")
_PDM_JSONL_DIR = os.environ.get("PDM_JSONL_DIR", "")
_PDM_TRAIN_JSONL = os.environ.get("PDM_TRAIN_JSONL", "")
_PDM_VAL_JSONL = os.environ.get("PDM_VAL_JSONL", "")


def _pdm_jsonl(filename):
    if filename == "pdm_lite_2hz_2tp_train_bev_encoder.jsonl" and _PDM_TRAIN_JSONL:
        return _PDM_TRAIN_JSONL
    if filename == "pdm_lite_2hz_2tp_val_bev_encoder.jsonl" and _PDM_VAL_JSONL:
        return _PDM_VAL_JSONL
    return os.path.join(_PDM_JSONL_DIR, filename)


DATASET_INFO = {
    "automot_pdm_lite": {
        "pdm_lite_traj": {
            "data_dir": _PDM_DATA_DIR,
            "jsonl_path": _pdm_jsonl("pdm_lite_2hz_2tp_train_bev_encoder.jsonl"),
            "num_total_samples": 599852,
        },
        "pdm_lite_traj_eval": {
            "data_dir": _PDM_DATA_DIR,
            "jsonl_path": _pdm_jsonl("pdm_lite_2hz_2tp_val_bev_encoder.jsonl"),
            "num_total_samples": 16077,
        },
    },
}
