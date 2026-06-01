from collections.abc import Callable


SUPPORTED_MODELS = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm")


def get_train_handler(model_name: str) -> Callable:
    if model_name == "vlg_cbm":
        from methods.vlg import train_vlg_cbm

        return train_vlg_cbm
    if model_name == "lf_cbm":
        from methods.lf import train_lf_cbm

        return train_lf_cbm
    if model_name == "salf_cbm":
        from methods.salf import train_salf_cbm

        return train_salf_cbm
    if model_name == "savlg_cbm":
        from methods.savlg import train_savlg_cbm

        return train_savlg_cbm
    raise ValueError(f"Unsupported model_name: {model_name}")
