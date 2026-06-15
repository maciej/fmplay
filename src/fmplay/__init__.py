"""Audio playback and degradation profile utilities."""

__all__ = [
    "AtcCloseMicProfile",
    "GsmCodecProfile",
    "LibGsmProfile",
    "MarineVhf1993Profile",
    "PassthroughProfile",
    "Profile",
    "ProfileError",
    "ProfileSummary",
    "get_profile",
    "list_profile_summaries",
    "list_profiles",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from fmplay import profiles

        return getattr(profiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
