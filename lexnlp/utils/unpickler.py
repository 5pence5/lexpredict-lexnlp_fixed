import joblib


def renamed_load(file_obj):
    """Wrapper around :func:`joblib.load` kept for backward compatibility."""

    return joblib.load(file_obj)
