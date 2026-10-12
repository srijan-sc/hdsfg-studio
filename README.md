# HDSFG Studio

[MIT licence](LICENSE)

Desktop app for heterodyne-detected SFG retrieval, the v37b pipeline (50 kHz,
1031.5 nm, Andor `.asc`). It uses Python + Qt and runs without MATLAB.

## Install and run (anyone, one line)

You need [uv](https://docs.astral.sh/uv/), a small tool that fetches the right Python
and every library by itself. Install it once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then start the app with one line (open a new terminal after installing uv):

```bash
uvx --python 3.12 --from "git+https://github.com/srijan-sc/hdsfg-studio" hdsfg-studio
```

The first start downloads Python 3.12, Qt, numpy, scipy and matplotlib (about 300 MB,
once); later starts take a few seconds. Add a data folder at the end to open it
directly, e.g. `... hdsfg-studio ~/data/20261009_50kHz`. Otherwise use *Open folder…*
or *Open files…*. No data is included: bring your own Andor `.asc` files.

**To keep it as a command** (then just type `hdsfg-studio`):

```bash
uv tool install --python 3.12 "git+https://github.com/srijan-sc/hdsfg-studio"
hdsfg-studio
```

**Updates:** `uv tool upgrade hdsfg-studio` (installed), or add `--refresh` after `uvx`
(one-line start).

git must be installed ([download](https://git-scm.com/downloads); macOS has it after
`xcode-select --install`). Nothing else is needed: the repository is public.

## Development copy

```bash
git clone https://github.com/srijan-sc/hdsfg-studio.git
cd hdsfg-studio
uv run --python 3.12 hdsfg-studio            # or: double-click "HDSFG Studio.command" (macOS)
uv run --python 3.12 --extra test pytest -q  # tests
```

It reopens the last data folder you used. On first launch, use *Open folder…*
to pick a folder of `.asc` files.

## What's in the window

- **Settings panel (left).** Dropdowns and fields grouped as Data,
  Measurement, Cosmic rays, Time gate, Fresnel, After division and Display.
  Run dropdowns fill themselves from the folder ("zqz2 · 8 files · 15:08").
  *View → Show advanced settings* reveals the rarely used ones. Hover any
  label for what it does.
- **Choosing scans.** *Reference scans* and *Sample scans* under Data take scan
  numbers as text, e.g. `1-130` or `1-10, 20, 40-50`; empty loads every scan.
  The line below shows how many are loaded ("130 of 132 scans"). Text that
  cannot be read turns the field red and keeps the previous choice. Opening
  another folder resets both to all scans.
- **Full-chip images.** Runs saved in *Full Resolution Image* mode (wavelength
  plus one column per chip row) load like any other run; name the files in the
  usual pattern, e.g. `zqzimg_ssp_1s_10frames_s1.asc`. With *Data → Image data*
  off, an image counts as the sum of all its rows (what FVB records) and the
  *Chip image* tab is greyed out. With it on, slices are chosen on that tab and
  *Data → Process* picks what goes on: one slice, the sum of the slices, or each
  slice as a scan (the run's files averaged; cosmic-ray removal skipped; Export
  adds `*_per_slice.txt`). Changing slices reruns from the slice step, without
  reading the files again.
- **Chosen files, any names.** *Open files…* (toolbar, or File → Open files…,
  ⇧⌘O) loads any `.asc` files picked together, e.g. 15 repeats of quartz. Names
  are not read: the files are numbered 1, 2, ... in acquisition order (footer
  time), each file is one scan, and the set is used as both reference and
  sample. The sidebar's *Chosen files* section takes the exposure per frame and
  the frames (filled in from the first file's footer; change them if wrong) and
  the material (quartz, water, D₂O, gold, diamond), which sets the Fresnel
  factors. *Data → Files to use* picks some of them by number. *Open folder…*
  goes back to the folder's runs. The app reopens the files on the next start.
- **Nothing freezes the window.** Processing runs in the background; the status bar shows
  "⏳ Running: …" with a moving bar while anything is working (main processing, Row by row,
  the slice sub-tabs, the compare window).
- **Live update.** Changing a setting reruns only the steps it affects:
  a phase change redoes the division alone, a gate change starts at the
  transform. A full run takes about 25 ms.
- **Summary cards.** Scans, rotation, usable range (where quartz is above 5 % of
  its peak), pixels replaced, and the phase drift across the sample run.
- **Tabs:**

| tab | shows |
|---|---|
| Chip image | four sub-tabs. *Detector and slices*, image runs: the detector image (IR wavenumber or nm across, chip row up), the mean counts per row, and the spectrum of each slice with their sum. Click the image, or use Add, Add peak or Add fan, to add slices (up to 8); click a slice's chip to remove it. Averaging window ± px with Mean or Sum. *Zoom to signal* and *Full frame*; drag to zoom | *Before division per slice*: each slice of the reference and sample runs after the gate and the back-transform, as Re or Im so the fringes stay (dashed: the ±\|amplitude\| envelope that *Before division* draws), in counts or each ÷ its peak; *One slice ÷ another* divides two slices' gated spectra (px A ÷ px B, same run, nothing else applied) as Re, Im, \|ratio\| or phase; *Add* keeps a division so up to 8 are drawn together, *Every slice ÷ this* adds all slices over one | *χ⁽²⁾ per slice* runs the whole retrieval for every slice on its own and draws them in one journal figure (Im and Re χ⁽²⁾, \|χ⁽²⁾\|² or phase; absolute or relative). Each slice is divided by the same slice of the reference, or by the sum of the reference slices to compare rows of one run. *Export…* writes every slice's usual export into its own `px<row>` folder, plus `slices.svg/.pdf`, `slices_homodyne` and `slices_phase` | *Files* (chosen files): the list with time, data shape and total counts; tick the files to use, *Remove unticked*, double-click a file to show it on *Detector and slices* (also: the file picker with ◀ ▶ there, or *Mean of the files*). *Compare across files…* opens a separate window: one slice (or the sum of the slices, or the whole chip) of every file together, as counts, the gated spectrum (Re, Im, \|amplitude\|, phase) or χ⁽²⁾ (Im, Re, \|χ⁽²⁾\|², phase; absolute, or relative = file ÷ mean of the files), drawn as lines with the mean dashed or as a map (file against wavenumber); tick which files are drawn; *Export…* writes a text table, one column per file, and an SVG |
| Chip image → *Row by row (quartz ÷ sample)* | its own data, not the sidebar's runs: *Choose quartz files…* and *Choose sample files…* (any names, full-chip images, numbered 1, 2, ... in acquisition order), a number list for each (*Average files* for quartz, e.g. `1-5, 8`; *Use files* for the sample; empty = all), exposure per frame and frames (filled in from the footer) and the sample material. Chip row n of the sample is divided by chip row n of the averaged quartz, every row through the whole retrieval on its own; the rows are binned only at the end, over the *Chip rows* chosen: a list such as `140-144` or `139, 141-143, 146` (empty = whole chip), *Auto* (rows above 10 % of the brightest quartz row), *Whole chip*, or click rows on the quartz brightness plot below to add or remove them. The dashed comparison sums the same rows. Binning: weighted by quartz² (default), mean of the rows where quartz is strong, or plain mean. Three sub-tabs on the right: *Time domain and before division* shows |FFT| of quartz and sample (sum of the chosen rows or one row) with the gates, draggable as on the Time domain tab, and the gated spectra with their fringes; *Plots* draws the binned Im/Re χ⁽²⁾ (mean ± 95 % across the sample files) with the same files done the usual way (rows summed first) dashed, a map of every row's χ⁽²⁾ and a map of each row's share of the bin; *Publication* is the journal figure (Im and Re χ⁽²⁾, |χ⁽²⁾|² or phase; shaded 95 % band across the sample files; rows summed first dashed; the ag_h2o reference water in grey, absolute units; optionally only where quartz is strong), exactly what Export writes. Gates, laser, geometry, Fresnel models and cosmic rays are the sidebar's. *Export…*: binned and rows-summed-first `*_ax_and_spec.txt`, `per_row_*.txt`, `rows`, `rows_homodyne`, `rows_phase` (.svg/.pdf, also `*_relative`), `settings.json`. The choices are remembered between sessions |
| Raw spectra | counts per scan, against wavelength and resampled to wavenumber, with replaced pixels marked. Dropdowns pick the channel, the scans (all, mean, first and last, every 10th, or one scan) and the x and y scales (linear, log, each line ÷ its peak). |
| Time domain | \|FFT\| on a log scale over the whole axis. **Drag the green gates** to move them |
| Before division | amplitude of each channel's gated spectrum; grey where quartz is weak |
| χ⁽²⁾ | absolute or relative, mean ± 95 % CI or every scan, optional `ag_h2o` comparison; phase and \|χ⁽²⁾\|² below |
| Phase drift | angle(point ÷ first point) as a map and as lines, with grouping and smoothing (plain averages, no fitting) |
| Homodyne | \|E(t) ÷ first scan\|², \|E(t) ÷ mean\|² or \|E(t)\|² for every scan in one plot, light to dark blue by time, mean in red |
| Publication | the journal figure, exactly as Export writes it |

- **Zoom (every plot tab except Publication).** *Zoom box* on: drag a rectangle
  to zoom into it; off: drag pans. The mouse wheel zooms either way. A zoom
  stays through reruns and dropdown changes, per plot; changing what an axis
  means (log scale, units, divisor) starts that plot from the full view.
  *Reset zoom* returns to the full view. On *Time domain*, a drag that starts on
  a gate still moves the gate.
- **Export (⌘E)** writes the `*_ax_and_spec.txt` text files (same format as
  MATLAB), `output.svg/.pdf`, `output_homodyne.svg/.pdf`, and `settings.json`,
  which records every parameter, the files used and any warnings.
- **Save and Load settings (⌘S)** store a set of parameters as JSON, to reuse
  them as a preset.

## Layout

```
studio/
├── hdsfg_engine/       processing only, no GUI: data, processing, fresnel,
│                       pipeline (cached stages), rowwise (row by row),
│                       drift, settings; reference/ (refractive indices, ag_h2o)
├── hdsfg_studio/       the app: main_window, parameter_panel (built from
│                       settings.PARAMETERS), pages (plot tabs), files_view
│                       (Files sub-tab, compare window), row_view (Row by
│                       row sub-tab), background (worker threads), publication, theme
├── tests/              engine tests (the MATLAB comparison needs the lab's data)
├── tools/screenshot.py renders every tab to PNG without a display
├── pyproject.toml      package definition for uv / pip
└── HDSFG Studio.command
```

`pyproject.toml` makes the folder an installable package (`hdsfg-studio`, entry point
`hdsfg_studio.__main__:main`). The refractive indices and the `ag_h2o` reference water
spectra live in `hdsfg_engine/reference/` and ship with the package.

To add a parameter, add a field to `Settings` and a `ParameterSpec` in
`hdsfg_engine/settings.py`. Its control appears in the panel automatically;
then use it in `pipeline.py`.

## Validation

The engine was checked against the group's MATLAB port of the v37b analysis at every
stage, with cosmic-ray removal on and off (`tests/test_against_matlab.py`):

| stage | largest difference (relative) |
|---|---|
| file order, timestamps, gates | identical |
| cosmic-ray masks | identical |
| resampled and gated spectra | 1e-13 |
| Fresnel conversion | 1e-15 |
| relative and absolute χ⁽²⁾, every scan | 3e-11 |
| exported `*_ax_and_spec.txt` vs MATLAB's | identical at the 5 decimals written |

The MATLAB port matches the group's v37b Python script to ~1e-10 %, so the app is tied
to the original analysis. That comparison needs the lab's data and MATLAB output, which
are not in this repository, so those tests are skipped here; every other test runs on
synthetic files.

## Not done yet

- Packaging as a standalone `.app` / `.exe` (PyInstaller) for people who do not use a
  terminal; for now it is installed with uv (above).
- The v3 / 800 nm pipeline as a second mode.
- Dark theme: switchable in *View* and applied on the next start.
