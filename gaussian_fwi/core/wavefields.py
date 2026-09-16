"""Invocation-owned scratch lifetime for uncompressed disk wavefields."""

import logging
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

_LOG = logging.getLogger(__name__)
_TEMPORARY = re.compile(r"deepwave_tmp_[0-9]+_[0-9a-f]{32}")


def _clean_owned(path, parent, identity):
    """Remove recognized scratch only after checking absolute ownership."""
    current = path.stat()
    if (path.is_symlink() or path.is_junction() or path.resolve() != parent / "wavefields"
            or (current.st_dev, current.st_ino) != identity):
        raise OSError("Wavefield directory identity changed; scratch was preserved")
    unexpected = []
    for child in path.iterdir():
        if (not _TEMPORARY.fullmatch(child.name) or child.is_symlink()
                or child.is_junction() or not child.is_dir()):
            unexpected.append(child.name)
            continue
        target = child.resolve(strict=True)
        if target.parent != parent / "wavefields":
            raise OSError("Wavefield temporary path escaped its owned directory")
        shutil.rmtree(target)
    if unexpected:
        raise OSError(f"Unrecognized wavefield entries preserved: {unexpected}")


@contextmanager
def wavefield_directory(output: Path, storage: str):
    """Own one new scratch directory through completion, pause or failure.

    Deepwave normally removes its temporary files when the autograd graph is
    released. A retained exception traceback can keep that graph alive. This
    invocation boundary removes recognized scratch explicitly after execution
    stops, without touching checkpoints or earlier runs. It cannot handle a
    killed process or a file the operating system still refuses to release.
    """
    if storage != "disk":
        yield None
        return
    parent = Path(output).resolve(strict=True)
    path = parent / "wavefields"
    path.mkdir(exist_ok=False)
    created = path.stat()
    identity = (created.st_dev, created.st_ino)
    failure = None
    try:
        yield path
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            _clean_owned(path, parent, identity)
        except OSError as error:
            message = f"Wavefield scratch cleanup failed at {path}: {error}"
            if failure is None:
                raise RuntimeError(message) from error
            failure.add_note(message)
            _LOG.warning(message)
