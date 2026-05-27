"""overlay_ring_fixed_hip=False must use ADS FOV in production wiring."""

from __future__ import annotations

from profiles import effective_fov_radius, effective_overlay_fov_radius


def test_overlay_fov_follows_ads_when_not_fixed_hip() -> None:
    cfg = {
        "fov_radius_pixels": 140,
        "fov_radius_ads_pixels": 200,
        "overlay_ring_fixed_hip": False,
        "fov_ads_scale": 1.3,
    }
    hip = effective_overlay_fov_radius(cfg, ads_active=False)
    ads = effective_overlay_fov_radius(cfg, ads_active=True)
    assert hip == effective_fov_radius(cfg, ads_active=False)
    assert ads == effective_fov_radius(cfg, ads_active=True)
    assert ads > hip
