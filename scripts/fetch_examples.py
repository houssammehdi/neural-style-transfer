"""Download the public-domain / CC0 example images used by the gallery.

Every file is fetched from a pinned commit on raw.githubusercontent.com and
checked against a pinned SHA-256 digest. The originals are written to
``examples/inputs/`` (git-ignored) and are never committed; see
``docs/gallery/CREDITS.md`` for provenance and licences.

    python scripts/fetch_examples.py            # all images
    python scripts/fetch_examples.py --list     # show what would be fetched
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

JCJOHNSON = (
    "https://raw.githubusercontent.com/jcjohnson/neural-style/"
    "07c4b8299f8fbdafec0c514fc820ff1d7ff62e46/examples/inputs/"
)
SKIMAGE = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/data/"  # tag v0.25.2
)


@dataclass(frozen=True)
class Example:
    """One input image with its pinned source and digest."""

    name: str
    url: str
    sha256: str
    credit: str


EXAMPLES = (
    Example(
        "starry_night.jpg",
        JCJOHNSON + "starry_night.jpg",
        "0592ab0f7b51b3ce72608a95e53880bd93971e5b48f93a0ce8dfd3cbccf3d098",
        "Vincent van Gogh, The Starry Night (1889) - public domain",
    ),
    Example(
        "the_scream.jpg",
        JCJOHNSON + "the_scream.jpg",
        "f3075cddeb974a9ee1d6418f7f33fd05185a7ac7dcb4288bd2805d499a9dcaee",
        "Edvard Munch, The Scream (1893) - public domain",
    ),
    Example(
        "shipwreck.jpg",
        JCJOHNSON + "shipwreck.jpg",
        "4f7cf34e3cd4a5daaf3f421f059ca9a01aaa2a3afd7243d6670c9c8bc001a213",
        "J. M. W. Turner, The Shipwreck (1805) - public domain",
    ),
    Example(
        "woman-with-hat-matisse.jpg",
        JCJOHNSON + "woman-with-hat-matisse.jpg",
        "6a8b2383d9c110354a97a42c74bdcc5341be879cbb842ddb7ab5843dbdf56d53",
        "Henri Matisse, Woman with a Hat (1905) - public domain",
    ),
    Example(
        "chelsea.png",
        SKIMAGE + "chelsea.png",
        "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
        "Stefan van der Walt, Chelsea the cat - CC0",
    ),
    Example(
        "coffee.png",
        SKIMAGE + "coffee.png",
        "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
        "Rachel Michetti, coffee cup - CC0",
    ),
    Example(
        "rocket.jpg",
        SKIMAGE + "rocket.jpg",
        "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
        "SpaceX, DSCOVR launch on Falcon 9 - public domain",
    ),
)

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "examples" / "inputs"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(example: Example, directory: Path) -> Path:
    """Download ``example`` into ``directory`` unless a verified copy is already there."""
    target = directory / example.name
    if target.exists() and _sha256(target.read_bytes()) == example.sha256:
        return target
    with urllib.request.urlopen(example.url, timeout=60) as response:
        data = response.read()
    digest = _sha256(data)
    if digest != example.sha256:
        raise RuntimeError(f"{example.url}: SHA-256 {digest} does not match {example.sha256}")
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def main(argv: list[str] | None = None) -> int:
    """Fetch every example image (or list them with ``--list``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help=f"destination (default {DEFAULT_DIR})")
    parser.add_argument("--list", action="store_true", help="list the images and exit")
    args = parser.parse_args(argv)
    for example in EXAMPLES:
        if args.list:
            print(f"{example.name:28} {example.credit}")
            continue
        path = fetch(example, args.dir)
        print(f"ok  {path}  ({example.credit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
