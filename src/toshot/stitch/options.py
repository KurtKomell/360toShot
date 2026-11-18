from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StitchOptions:
    """Advanced seam / motion options (DJI-Studio-like)."""

    optical_flow: bool = True
    selfie_mask: bool = True
    rolling_shutter: bool = True
    # Rolling-shutter readout duration as fraction of the frame period
    rs_readout_fraction: float = 0.7
    # Selfie-stick: fraction of equirect height at the nadir to inspect
    selfie_nadir_fraction: float = 0.12
    # Optical-flow analysis scale (1 = full, 0.25 = quarter)
    flow_scale: float = 0.25
    # Soft blend half-width around the carved seam (equirect pixels at full res)
    seam_blend_px: int = 8

    @property
    def advanced(self) -> bool:
        return self.optical_flow or self.selfie_mask or self.rolling_shutter

    @classmethod
    def fast(cls) -> StitchOptions:
        return cls(optical_flow=False, selfie_mask=False, rolling_shutter=False)

    @classmethod
    def from_flags(
        cls,
        *,
        optical_flow: bool = True,
        selfie_mask: bool = True,
        rolling_shutter: bool = True,
        fast: bool = False,
    ) -> StitchOptions:
        if fast:
            return cls.fast()
        return cls(
            optical_flow=optical_flow,
            selfie_mask=selfie_mask,
            rolling_shutter=rolling_shutter,
        )
