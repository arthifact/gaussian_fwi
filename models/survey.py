"""Read a bundle's acquisition geometry and draw it over a velocity panel."""

import torch

LEGEND = "Sources (stars) and receivers (dots)"


def read_geometry(bundle, spacing=10.0):
    """Return source and receiver positions as (z, x) metre pairs.

    Acquisition indices are stored in tensor-axis order, so column 0 is depth
    and column 1 is the horizontal coordinate.
    """
    payload = torch.load(bundle, map_location="cpu", weights_only=True)
    acquisition = payload["acquisition"]
    sources = acquisition["source_locations"][:, 0].double().numpy() * spacing
    receivers = acquisition["receiver_locations"][0].double().numpy() * spacing
    return sources, receivers


def draw_geometry(axis, sources, receivers, *, label=False):
    """Overlay the survey on a velocity image drawn in metre coordinates."""
    axis.scatter(receivers[:, 1], receivers[:, 0], marker=".", s=5, c="black",
                 linewidths=0, zorder=3, label="Receivers" if label else None)
    axis.scatter(sources[:, 1], sources[:, 0], marker="*", s=70, c="white",
                 edgecolors="black", linewidths=.5, zorder=4,
                 label="Sources" if label else None)


def aperture(sources, receivers):
    """Maximum source-receiver offset in metres."""
    return float(max(abs(s[1] - r[1]) for s in sources for r in receivers))
