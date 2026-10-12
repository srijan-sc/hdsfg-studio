"""Finding runs in a data folder and reading Andor .asc spectra.

Port of the loading part of ``v37bRetrieve.m``. A data folder holds several
runs (zqz1, zqz2, h2o2, ...), each a set of numbered files such as
``h2o2_ssp_20s_4frames_s7.asc``. Files with "bg" in the name are skipped: the
Andor software has already subtracted the background ("Counts (Bg Corrected)").

A file is either one spectrum (Full Vertical Binning: wavelength and counts) or
a full-chip image (Full Resolution Image: wavelength and one column per chip
row). An image is loaded as the sum of all its rows, which is what FVB records,
and the image itself is kept. ``slice_channel`` then cuts slices out of it: a
slice is a chip row, or the rows within a half-width of it, and what goes on to
the processing is one slice, the sum of the slices, or every slice as a scan.

``load_files`` reads a list of chosen files with any names instead: each file
is one scan, numbered 1, 2, ... in the order given, and the exposure is the one
typed in, not read from the names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

NUMBER_PATTERN = re.compile(r"_s(\d+)[_.]")
EXPOSURE_PATTERN = re.compile(r"_(\d+)s_")
FRAMES_PATTERN = re.compile(r"_(\d+)frames_")
FOOTER_START = "Date and Time:"


@dataclass
class RunInfo:
    """One run in a data folder, as listed in the GUI's run dropdowns."""
    name: str
    files: list[Path]
    numbers: list[int]
    exposure_sec: float | None     # exposure x frames, from the file names
    start_time: datetime | None

    @property
    def label(self) -> str:
        parts = [self.name, f"{len(self.files)} files"]
        if self.start_time is not None:
            parts.append(self.start_time.strftime("%H:%M"))
        return " · ".join(parts)


@dataclass
class Channel:
    """One run's spectra, [n_pixels x n_scans], in acquisition order."""
    name: str
    files: list[Path]
    numbers: np.ndarray
    wavelength_nm: np.ndarray
    counts: np.ndarray                 # what the pipeline carries forward
    raw_counts: np.ndarray             # exactly as read
    time_sec: np.ndarray               # middle of each acquisition
    file_exposure_sec: np.ndarray
    exposure_sec: float = 0.0
    spike_mask: np.ndarray | None = None
    n_spikes_replaced: int = 0
    images: np.ndarray | None = None   # full-chip files [n_files x n_pixels x n_chip_rows]
    rows_as_scans: bool = False        # each column is one slice, averaged over the files
    processed: str = "all rows"        # what the counts are, for labels and the export record
    # filled in by the transform stage
    resampled: np.ndarray | None = None
    time_domain: np.ndarray | None = None
    analytic: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_scans(self) -> int:
        return self.counts.shape[1]

    @property
    def is_image(self) -> bool:
        return self.images is not None

    @property
    def image(self) -> np.ndarray | None:
        """The whole chip, mean of the files [n_pixels x n_chip_rows]."""
        return None if self.images is None else self.images.mean(axis=0)

    @property
    def file_numbers(self) -> np.ndarray:
        """Number of each file in ``files`` (``numbers`` holds chip rows when slices are the scans)."""
        return self.extra.get("file_numbers", self.numbers)

    @property
    def prefix(self) -> str:
        """Label prefix of ``numbers``: "s" for a scan number, "px " for a slice's chip row."""
        return "px " if self.rows_as_scans else "s"


def file_belongs_to_run(path: Path, run_name: str) -> bool:
    """True when one of the name's "_"-separated tokens is exactly run_name."""
    return run_name in path.name.split("_") and "bg" not in path.name


def file_number(path: Path) -> int | None:
    match = NUMBER_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def exposure_from_name(path: Path) -> float | None:
    exposure = EXPOSURE_PATTERN.search(path.name)
    frames = FRAMES_PATTERN.search(path.name)
    if exposure and frames:
        return float(exposure.group(1)) * float(frames.group(1))
    return None


def list_runs(data_dir: str | Path) -> list[RunInfo]:
    """Every run in the folder, in acquisition order."""
    files = sorted(Path(data_dir).glob("*.asc"))
    names = sorted({path.name.split("_")[0] for path in files
                    if file_number(path) is not None and "bg" not in path.name})
    runs = []
    for name in names:
        members = sorted((path for path in files if file_belongs_to_run(path, name)
                          and file_number(path) is not None), key=file_number)
        start = None
        if members:
            times = [read_end_time(path) for path in members]
            times = [t for t in times if t is not None]
            start = min(times) if times else None
        runs.append(RunInfo(name=name, files=members, numbers=[file_number(p) for p in members],
                            exposure_sec=exposure_from_name(members[0]) if members else None,
                            start_time=start))
    runs.sort(key=lambda run: (run.start_time is None, run.start_time or datetime.min, run.name))
    return runs


def read_end_time(path: Path) -> datetime | None:
    """The "Date and Time:" footer line: when the acquisition ended."""
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith(FOOTER_START):
                text = line.split(FOOTER_START, 1)[1].strip()
                return datetime.strptime(text, "%a %b %d %H:%M:%S.%f %Y")
    return None


def read_footer_exposure(path: Path) -> tuple[float | None, int | None]:
    """"Exposure Time (secs)" and "Number of Accumulations" from the footer, if there."""
    exposure, frames = None, None
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("Exposure Time (secs):"):
                exposure = float(line.split(":", 1)[1])
            elif line.startswith("Number of Accumulations:"):
                frames = int(line.split(":", 1)[1])
    return exposure, frames


def sort_files(paths) -> list[str]:
    """Files in acquisition order (footer time), then by name; files without a time last."""
    def key(path):
        time_ = read_end_time(Path(path))
        return (time_ is None, time_ or datetime.min, Path(path).name)
    return [str(path) for path in sorted(paths, key=key)]


def _parse_table(block: str, path: Path) -> np.ndarray:
    """The numbers before the footer, one row per line.

    Fast path: every value converted by numpy at once (an image file is ~260 000 values).
    If the lines do not all hold the same number of values, the slow line-by-line reading
    decides, as before: lines with fewer than two values are skipped, and rows of
    different lengths are an error.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    tokens = block.split()
    n_columns = len(lines[0].split()) if lines else 0
    if n_columns >= 2 and len(tokens) == len(lines) * n_columns:
        return np.array(tokens, dtype=np.float64).reshape(len(lines), n_columns)
    values = [[float(part) for part in line.split()] for line in lines if len(line.split()) >= 2]
    if len({len(row) for row in values}) != 1:
        raise ValueError(f"{Path(path).name}: rows have different numbers of columns")
    return np.asarray(values)


def read_asc(path: Path, wavelength_window_nm: tuple[float, float], total_exposure: float | None = None):
    """Wavelength (nm), counts [n_pixels x n_chip_rows], and the middle of the acquisition (POSIX s).

    An FVB file has one column of counts, an image one per chip row. The
    footer's time marks the end of the acquisition, so half the total exposure
    is subtracted to place the spectrum at the middle of it. The total exposure
    is ``total_exposure`` when given, otherwise the one in the file name.
    """
    text = Path(path).read_text(errors="replace")
    start = 0 if text.startswith(FOOTER_START) else text.find("\n" + FOOTER_START) + 1   # 0: no footer
    end_time = None
    if start > 0 or text.startswith(FOOTER_START):
        line_end = text.find("\n", start)
        stamp = text[start + len(FOOTER_START):line_end if line_end >= 0 else None].strip()
        end_time = datetime.strptime(stamp, "%a %b %d %H:%M:%S.%f %Y")
    data = _parse_table(text[:start] if end_time is not None else text, path)
    wavelength, counts = data[:, 0], data[:, 1:].copy()

    # roll off outside the wavelength window, 0.9 per nm (as in v37b.py)
    low, high = min(wavelength_window_nm), max(wavelength_window_nm)
    below, above = wavelength < low, wavelength > high
    counts[below] *= 0.9 ** np.abs(wavelength[below] - low)[:, None]
    counts[above] *= 0.9 ** (1 + np.abs(wavelength[above] - high))[:, None]

    if total_exposure is None:
        total_exposure = exposure_from_name(path) or 0.0
    # POSIX seconds of a naive datetime taken as UTC; only differences matter
    mid_time = (end_time - datetime(1970, 1, 1)).total_seconds() - total_exposure / 2 \
        if end_time is not None else np.nan
    return wavelength, counts, mid_time, total_exposure


def load_channel(data_dir: str | Path, run_name: str, file_numbers, wavelength_window_nm) -> Channel:
    """All files of one run, optionally limited to some file numbers.

    Image files are summed over every chip row, like FVB, and kept in ``images``.
    """
    files = [path for path in Path(data_dir).glob("*.asc")
             if file_belongs_to_run(path, run_name) and file_number(path) is not None]
    if not files:
        raise FileNotFoundError(f'No files of run "{run_name}" in {data_dir}')
    if file_numbers:
        files = [path for path in files if file_number(path) in set(file_numbers)]
        if not files:
            raise FileNotFoundError(f'None of the chosen scan numbers exist in run "{run_name}"')
    files.sort(key=file_number)
    return _read_channel(run_name, files, [file_number(p) for p in files], wavelength_window_nm, None)


def load_files(paths, file_numbers, wavelength_window_nm, total_exposure_sec: float,
               name: str = "files") -> Channel:
    """Chosen files with any names, file k of the list as scan k (1-based).

    ``file_numbers`` keeps only those positions in the list (empty keeps all).
    Every file gets ``total_exposure_sec`` (exposure x frames, as typed in).
    """
    numbered = list(enumerate((Path(p) for p in paths), start=1))
    if not numbered:
        raise FileNotFoundError("No files chosen")
    if file_numbers:
        numbered = [(n, path) for n, path in numbered if n in set(file_numbers)]
        if not numbered:
            raise FileNotFoundError("None of the chosen file numbers are in the list")
    return _read_channel(name, [path for _, path in numbered], [n for n, _ in numbered],
                         wavelength_window_nm, total_exposure_sec)


def _read_channel(name: str, files: list[Path], numbers, wavelength_window_nm,
                  total_exposure: float | None) -> Channel:
    images, times, exposures, wavelength = [], [], [], None
    for path in files:
        wl, counts, mid_time, exposure = read_asc(path, wavelength_window_nm, total_exposure)
        if wavelength is None:
            wavelength = wl
        elif len(wl) != len(wavelength):
            raise ValueError(f"{path.name} has {len(wl)} pixels, expected {len(wavelength)}")
        if images and counts.shape[1] != images[0].shape[1]:
            raise ValueError(f"{path.name} has {counts.shape[1]} chip rows, "
                             f"{files[0].name} has {images[0].shape[1]}")
        images.append(counts)
        times.append(mid_time)
        exposures.append(exposure)

    is_image = images[0].shape[1] > 1
    counts = np.column_stack([image.sum(axis=1) for image in images])
    return Channel(name=name, files=files, numbers=np.array(numbers),
                   wavelength_nm=wavelength, counts=counts, raw_counts=counts.copy(),
                   time_sec=np.array(times, dtype=float),
                   file_exposure_sec=np.array(exposures, dtype=float),
                   images=np.stack(images).astype(np.float32) if is_image else None)


def slice_rows(centre: int, half_width: int, n_rows: int) -> np.ndarray:
    """0-based chip rows of the slice centred on ``centre`` (1-based), clipped to the chip."""
    low, high = max(1, centre - half_width), min(n_rows, centre + half_width)
    return np.arange(low, high + 1) - 1


def slice_channel(channel: Channel, slices, half_width: int, combine: str, process: str) -> Channel:
    """The counts the processing gets from an image run.

    Each slice is the chip rows within ``half_width`` of its centre, averaged
    (``combine="mean"``) or summed. ``process`` is "sum" (the slices added, one
    spectrum per file), "each" (the files averaged, one scan per slice) or a
    slice's centre row as text (that slice alone, one spectrum per file).
    Returns a new Channel; an FVB channel is returned unchanged.
    """
    if channel.images is None:
        return channel
    n_rows = channel.images.shape[2]
    centres = sorted(set(int(c) for c in slices))
    bad = [c for c in centres if not 1 <= c <= n_rows]
    if bad:
        raise ValueError(f"{channel.name} has chip rows 1-{n_rows}; slice at {bad[0]} is outside")
    if not centres:
        return replace(channel, processed="all rows")

    reduce = np.mean if combine == "mean" else np.sum
    # [n_files x n_pixels x n_slices]
    cut = np.stack([reduce(channel.images[:, :, slice_rows(c, half_width, n_rows)], axis=2)
                    for c in centres], axis=2).astype(float)
    window = f"±{half_width} px {combine}" if half_width else "1 row"
    if process == "each":
        n = len(centres)
        times = channel.time_sec
        counts = cut.mean(axis=0)
        return replace(channel, counts=counts, raw_counts=counts.copy(), numbers=np.array(centres),
                       extra=dict(channel.extra, file_numbers=channel.numbers),
                       time_sec=np.full(n, np.nanmean(times) if np.isfinite(times).any() else np.nan),
                       file_exposure_sec=np.full(n, channel.file_exposure_sec[0]), rows_as_scans=True,
                       processed=f"each slice as a scan ({window})")
    if process == "sum":
        counts = cut.sum(axis=2).T
        processed = f"sum of slices px {', '.join(map(str, centres))} ({window})"
    else:
        centre = int(process)
        if centre not in centres:
            raise ValueError(f"px {centre} is not one of the slices")
        counts = cut[:, :, centres.index(centre)].T
        processed = f"slice px {centre} ({window})"
    return replace(channel, counts=counts, raw_counts=counts.copy(), processed=processed)


def sort_by_time(reference: Channel, sample: Channel) -> None:
    """Acquisition order, times counted from the earliest file of either run."""
    all_times = np.concatenate([reference.time_sec, sample.time_sec])
    if np.all(np.isnan(all_times)):
        reference.time_sec = reference.numbers * reference.file_exposure_sec[0]
        sample.time_sec = sample.numbers * sample.file_exposure_sec[0]
        start, mean = 0.0, 0.0
    else:
        start = np.nanmin(all_times)
        mean = np.nanmean(all_times) - start
    for channel in (reference, sample):
        channel.time_sec = channel.time_sec - start
        channel.time_sec[np.isnan(channel.time_sec)] = mean
        order = np.argsort(channel.time_sec, kind="stable")
        channel.time_sec = channel.time_sec[order]
        channel.files = [channel.files[i] for i in order]
        channel.numbers = channel.numbers[order]
        channel.file_exposure_sec = channel.file_exposure_sec[order]
        channel.counts = channel.counts[:, order]
        channel.raw_counts = channel.raw_counts[:, order]
        if channel.images is not None:
            channel.images = channel.images[order]


def resolve_exposure(configured: float | None, channel: Channel) -> tuple[float, str | None]:
    """The configured exposure, or the one in the file names; plus any warning."""
    from_names = np.unique(channel.file_exposure_sec)
    if configured is None:
        if len(from_names) != 1:
            raise ValueError(f"Files of {channel.name} have different exposures: {from_names}")
        return float(from_names[0]), None
    warning = None
    if not (len(from_names) == 1 and from_names[0] == configured):
        warning = f"{channel.name}: configured {configured} s, file names say {from_names} s"
    return float(configured), warning
