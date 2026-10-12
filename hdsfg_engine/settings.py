"""Every tunable parameter of the v37b retrieval, with what the GUI needs to show it.

The values match ``v37b/v37bDefaults.m``. Each parameter also carries its group,
label, kind (choice, number, switch, range) and a one-line explanation, so the
settings panel is generated from this list rather than written by hand: adding
a parameter here is enough for it to appear in the GUI.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# refractive indices and the ag_h2o comparison spectra, shipped inside the package
# (copies of ../v37b/reference, so the app works wherever it is installed)
DEFAULT_REFERENCE_DIR = PACKAGE_DIR / "reference"


@dataclass
class Settings:
    # --- data -----------------------------------------------------------------
    data_dir: str = ""
    reference_name: str = "zqz2"
    sample_name: str = "h2o2"
    reference_file_numbers: list[int] = field(default_factory=list)   # empty = all
    sample_file_numbers: list[int] = field(default_factory=list)
    # --- chosen files (File → Open files…): any names; used as both reference and sample ---
    files: list[str] = field(default_factory=list)     # empty = the runs of data_dir
    file_exposure_sec: float = 1.0        # exposure of one frame, typed in
    file_frames: int = 1                  # frames (accumulations) per file
    file_material: str = "zqz"            # "zqz" quartz, "h2o" water, "au" gold, "diamond"
    # --- image data (Full Resolution Image files) ---------------------------------
    image_data: bool = False              # off: an image counts as the sum of all its rows
    slices: list[int] = field(default_factory=list)      # centre chip rows, set on the Chip image tab
    slice_half_width: int = 0             # rows either side of the centre
    slice_combine: str = "mean"           # "mean" or "sum" over a slice's rows
    process_slice: str = "sum"            # "sum", "each", or a slice centre such as "142"
    reference_slice: str = "same"         # "same" as the sample, "sum" of the reference slices, or a centre
    time_weighted_reference: bool = False
    reference_dir: str = str(DEFAULT_REFERENCE_DIR)

    # --- measurement ------------------------------------------------------------
    sfg_or_dfg: str = "sfg"
    polarization: str = "ssp"
    laser_wavelength_nm: float = 1031.5
    geometry: str = "53deg"               # sets the three angles, see GEOMETRIES
    reference_exposure_sec: float | None = None   # None = from the file name
    sample_exposure_sec: float | None = None
    wavelength_window_nm: tuple[float, float] = (325.0, 2805.0)

    # --- cosmic rays (not in the Python; from the v3 port) -----------------------
    remove_cosmic_rays: bool = True
    median_filter_width: int = 131
    spike_threshold_sigma: float = 5.0
    min_exposures_for_despike: int = 3

    # --- time gate --------------------------------------------------------------
    gate_shape: int = 3
    gate_range_reference_ps: tuple[float, float] = (1.9, 3.6)
    gate_range_sample_ps: tuple[float, float] = (1.9, 3.6)
    gate_edges_reference_ps: tuple[float, float] = (0.105, 0.105)
    gate_edges_sample_ps: tuple[float, float] = (0.105, 0.105)

    # --- Fresnel ------------------------------------------------------------------
    interface_model_reference: str = "lorentz"
    interface_model_sample: str = "slab"
    absolute_scale: float = 1e22

    # --- after the division ---------------------------------------------------------
    phase_correction_deg: float = 0.0
    post_division_filter: bool = False
    post_division_limit_ps: float = 4.25
    post_division_width_ps: float = 0.75
    smoothing_order: int = 0
    smoothing_length: int = 31

    # --- display ----------------------------------------------------------------------
    plot_wavenumber_lim: tuple[float, float] = (2800.0, 4000.0)
    weak_reference_fraction: float = 0.05

    # ------------------------------------------------------------------------------------
    @property
    def angles_rad(self) -> tuple[float, float, float]:
        """(mir, vis, lo) angles of incidence for the chosen geometry."""
        return GEOMETRIES[self.geometry]["angles"]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        stored = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        values = {k: (tuple(v) if isinstance(v, list) and k.endswith(("_ps", "_nm", "_lim")) else v)
                  for k, v in stored.items() if k in known}
        if values.get("geometry") not in GEOMETRIES:     # a name from before the geometries were renamed
            values["geometry"] = "53deg"
        if not Path(values.get("reference_dir", DEFAULT_REFERENCE_DIR)).is_dir():
            values["reference_dir"] = str(DEFAULT_REFERENCE_DIR)   # saved on another computer
        return cls(**values)


GEOMETRIES = {
    "53deg": {"label": "53.1° (atan 4/3)", "angles": (math.atan(4 / 3),) * 3},
    "45deg": {"label": "45°",              "angles": (math.atan(1.0),) * 3},
}

GATE_SHAPES = {
    0: "No gate",
    1: "Box-car",
    2: "Happ-Genzel rising edge",
    3: "Happ-Genzel both edges",
    4: "Raised cosine both edges",
    5: "Kaiser-Bessel derived",
}


@dataclass(frozen=True)
class ParameterSpec:
    """How one Settings field is shown in the GUI."""
    name: str
    group: str
    label: str
    kind: str                 # "choice", "float", "int", "bool", "range", "run", "scans"
    help: str = ""
    choices: tuple = ()       # (value, label) pairs for "choice"
    unit: str = ""
    minimum: float = -1e9
    maximum: float = 1e9
    step: float = 1.0
    decimals: int = 3
    advanced: bool = False
    placeholder: str = "all scans"    # for "scans": what an empty field means


PARAMETERS: tuple[ParameterSpec, ...] = (
    # data
    ParameterSpec("reference_name", "Data", "Reference run", "run",
                  "Quartz run every sample spectrum is divided by."),
    ParameterSpec("reference_file_numbers", "Data", "Reference scans", "scans",
                  "Scan numbers of the reference run to load, e.g. 1-130 or 1-10, 20, 40-50. "
                  "Empty loads every scan."),
    ParameterSpec("sample_name", "Data", "Sample run", "run",
                  "The run being measured."),
    ParameterSpec("sample_file_numbers", "Data", "Sample scans", "scans",
                  "Scan numbers of the sample run to load, e.g. 1-130 or 1-10, 20, 40-50. "
                  "Empty loads every scan."),
    ParameterSpec("image_data", "Data", "Image data", "bool",
                  "On for runs saved as full-chip images (Full Resolution Image): slices are "
                  "chosen on the Chip image tab. Off: an image counts as the sum of all its "
                  "rows, as Full Vertical Binning records, and the Chip image tab is greyed out."),
    ParameterSpec("process_slice", "Data", "Process", "slice",
                  "Which slice the processing uses: one slice, the slices added together, or "
                  "every slice as its own scan (the run's files averaged; cosmic-ray removal "
                  "skipped, since slices are not repeats)."),
    ParameterSpec("time_weighted_reference", "Data", "Time-weighted reference", "bool",
                  "Weight reference scans by closeness in time to the sample run.", advanced=True),

    # chosen files
    ParameterSpec("file_exposure_sec", "Chosen files", "Exposure per frame", "float",
                  "Exposure of one frame of the chosen files. Filled in from the first file's "
                  "footer when files are opened; change it here if that is wrong. Sets the time "
                  "of each file (Phase drift, Homodyne).",
                  unit="s", minimum=0.001, maximum=10000, step=1, decimals=3),
    ParameterSpec("file_frames", "Chosen files", "Frames", "int",
                  "Frames (accumulations) per file. Exposure per frame × frames is each file's "
                  "total exposure.", minimum=1, maximum=100000),
    ParameterSpec("file_material", "Chosen files", "Material", "choice",
                  "What the chosen files are; sets the Fresnel factors and the known χ⁽²⁾. "
                  "The chosen files are used as both reference and sample.",
                  choices=(("zqz", "Quartz"), ("h2o", "Water"), ("d2o", "D₂O"), ("au", "Gold"),
                           ("diamond", "Diamond"))),

    # measurement
    ParameterSpec("sfg_or_dfg", "Measurement", "Process", "choice",
                  "Sum- or difference-frequency generation.",
                  choices=(("sfg", "SFG"), ("dfg", "DFG"))),
    ParameterSpec("polarization", "Measurement", "Polarization", "choice",
                  "Polarization combination, SFG-visible-IR.",
                  choices=(("ssp", "ssp"), ("sps", "sps"), ("ppp", "ppp"))),
    ParameterSpec("geometry", "Measurement", "Geometry", "choice",
                  "Angles of incidence of the IR, visible and LO beams.",
                  choices=tuple((key, value["label"]) for key, value in GEOMETRIES.items())),
    ParameterSpec("laser_wavelength_nm", "Measurement", "Laser wavelength", "float",
                  "Upconversion wavelength; sets the absolute wavenumber axis.",
                  unit="nm", minimum=400, maximum=2000, step=0.1, decimals=2),

    # cosmic rays
    ParameterSpec("remove_cosmic_rays", "Cosmic rays", "Remove cosmic rays", "bool",
                  "Replace single-pixel outliers with the median across scans. Off reproduces v37b.py."),
    ParameterSpec("spike_threshold_sigma", "Cosmic rays", "Threshold", "float",
                  "Multiple of the smoothed relative scatter a pixel must exceed.",
                  unit="σ", minimum=1, maximum=50, step=0.5, decimals=1, advanced=True),
    ParameterSpec("median_filter_width", "Cosmic rays", "Median width", "int",
                  "Running-median width for drift and threshold; must be odd.",
                  unit="px", minimum=3, maximum=501, step=2, advanced=True),

    # gate
    ParameterSpec("gate_shape", "Time gate", "Gate shape", "choice",
                  "Window applied in the time domain to keep the +tau lobe.",
                  choices=tuple(GATE_SHAPES.items())),
    ParameterSpec("gate_range_reference_ps", "Time gate", "Reference gate", "range",
                  "Start and end of the reference gate.", unit="ps",
                  minimum=-20, maximum=20, step=0.05, decimals=3),
    ParameterSpec("gate_range_sample_ps", "Time gate", "Sample gate", "range",
                  "Start and end of the sample gate.", unit="ps",
                  minimum=-20, maximum=20, step=0.05, decimals=3),
    ParameterSpec("gate_edges_sample_ps", "Time gate", "Edge widths", "range",
                  "Rising and falling edge widths (both channels).", unit="ps",
                  minimum=0.0, maximum=2, step=0.005, decimals=3, advanced=True),

    # Fresnel
    ParameterSpec("interface_model_reference", "Fresnel", "Reference model", "choice",
                  "Interfacial index model; bulk quartz should be Lorentz.",
                  choices=(("lorentz", "Lorentz"), ("slab", "Slab"))),
    ParameterSpec("interface_model_sample", "Fresnel", "Sample model", "choice",
                  "Interfacial index model for the sample.",
                  choices=(("slab", "Slab"), ("lorentz", "Lorentz"))),

    # after division
    ParameterSpec("phase_correction_deg", "After division", "Extra phase", "float",
                  "Rotation on top of the one set by the two materials.",
                  unit="°", minimum=-180, maximum=180, step=0.5, decimals=2),
    ParameterSpec("smoothing_order", "After division", "Savitzky-Golay order", "choice",
                  "Smoothing of the ratio; 0 is off, as in v37b.",
                  choices=((0, "Off"), (1, "1"), (2, "2"), (3, "3"), (4, "4")), advanced=True),
    ParameterSpec("post_division_filter", "After division", "Post-division filter", "bool",
                  "Second, symmetric time gate on the ratio itself.", advanced=True),

    # display
    ParameterSpec("plot_wavenumber_lim", "Display", "Wavenumber range", "range",
                  "Range shown on every spectrum plot.", unit="cm⁻¹",
                  minimum=0, maximum=6000, step=50, decimals=0),
    ParameterSpec("weak_reference_fraction", "Display", "Weak reference below", "float",
                  "Shade where quartz falls below this fraction of its peak.",
                  unit="", minimum=0.0, maximum=0.5, step=0.01, decimals=2),
)


def parse_scan_list(text: str) -> list[int]:
    """"1-10, 20, 30-40" -> [1, ..., 10, 20, 30, ..., 40]. Empty or "all" -> [] (all scans).

    Raises ValueError on anything else, with the piece that could not be read.
    """
    text = text.strip().lower()
    if text in ("", "all"):
        return []
    numbers: set[int] = set()
    for piece in text.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        bounds = [b.strip() for b in piece.replace("–", "-").split("-")]
        if len(bounds) == 1 and bounds[0].isdigit():
            numbers.add(int(bounds[0]))
        elif len(bounds) == 2 and all(b.isdigit() for b in bounds):
            low, high = sorted(int(b) for b in bounds)
            numbers.update(range(low, high + 1))
        else:
            raise ValueError(f'cannot read "{piece}"; use e.g. 1-130 or 1-10, 20, 40-50')
    return sorted(numbers)


def format_scan_list(numbers) -> str:
    """[1, 2, 3, 5, 7, 8] -> "1-3, 5, 7-8". Empty -> "" (all scans)."""
    numbers = sorted(set(int(n) for n in numbers))
    pieces, start = [], None
    for k, n in enumerate(numbers):
        if start is None:
            start = n
        if k + 1 == len(numbers) or numbers[k + 1] != n + 1:
            pieces.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(pieces)


def spec_for(name: str) -> ParameterSpec:
    return next(spec for spec in PARAMETERS if spec.name == name)
