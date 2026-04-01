from .model import AkimaOCVModel, TwoRCTheveninECM

__all__ = ["AkimaOCVModel", "TwoRCTheveninECM", "OfflineIdentifier"]


def __getattr__(name: str):
    if name == "OfflineIdentifier":
        from .identifier import OfflineIdentifier

        return OfflineIdentifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
