"""Fresnel factors and the conversion to absolute chi2 units.

Port of ``v37bFresnel.m`` (itself calculate_fresnel, apply_fresnel and
get_reference_mat_params of the group's v37b Python script). The approximations the
Python flags are kept: the SFG beam takes the LO's angle of incidence, and the
gold and diamond constants are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Material:
    name: str
    refractive_index_file: str
    known_chi2: float
    source: str           # "interface" or "bulk"
    phase_deg: float
    degeneracy: float
    placeholder: bool = False


@dataclass
class Fresnel:
    material: Material
    coefficients: dict
    conversion_scale: np.ndarray
    molecular_frame: str


def material_parameters(name: str) -> Material:
    """Known constants, chosen by the run name's prefix."""
    if name.startswith(("h2o", "d2o")):
        return Material(name, "nk_water_hale1973.txt", 4e-22, "interface", 0, 1)
    if name.startswith("zqz"):
        # 6e-13 m/V after Shoji et al., JOSA B 14, 2268 (1997); 2 is a degeneracy
        return Material(name, "nk_qz_franta2016.txt", 6e-13, "bulk", 90, 2)
    if name.startswith("au"):
        return Material(name, "nk_au_babar2015.txt", 2.5e-10, "interface", 180, 2, placeholder=True)
    if name.startswith("diamond"):
        return Material(name, "nk_diamond_phillip1964.txt", 3e-21, "interface", 180, 2, placeholder=True)
    return Material(name, "nk_water_hale1973.txt", np.nan, "interface", 0, 1, placeholder=True)


def read_refractive_index(path: Path):
    """Tab-separated: wavelength (nm), n, and k if present. Ascending wavenumber."""
    table = np.atleast_2d(np.loadtxt(path, delimiter="\t"))
    index = table[:, 1] + 1j * (table[:, 2] if table.shape[1] >= 3 else 0)
    wavenumber = 1e7 / table[:, 0]
    order = np.argsort(wavenumber)
    return wavenumber[order], index[order]


def _interfacial_index(n2, phi):
    return np.sqrt((1 + ((np.pi - 0.5 * phi) / (3 * np.pi)) * (n2 ** 2 - 1))
                   / (1 + ((1 - n2 ** 2) / (6 * n2 ** 2)) * (1 + np.cos(0.5 * phi)) ** 2
                      * (2 - np.cos(0.5 * phi))))


def _l_factors(n1, n2, n_interface, incident, transmitted):
    lxx = 2 * n1 * np.cos(transmitted) / (n1 * np.cos(transmitted) + n2 * np.cos(incident))
    lyy = 2 * n1 * np.cos(incident) / (n1 * np.cos(incident) + n2 * np.cos(transmitted))
    lzz = (2 * n2 * np.cos(incident) * (n1 / n_interface) ** 2
           / (n1 * np.cos(transmitted) + n2 * np.cos(incident)))
    return lxx, lyy, lzz


def compute(material_name: str, interface_model: str, wavenumber: np.ndarray,
            laser_wavelength_nm: float, angles_rad, polarization: str,
            sfg_or_dfg: str, reference_dir: str | Path) -> Fresnel:
    material = material_parameters(material_name)
    phi = 0.0 if interface_model == "lorentz" else np.pi
    mir_angle, vis_angle, lo_angle = angles_rad
    sfg_angle = lo_angle

    table_wn, table_n = read_refractive_index(Path(reference_dir) / material.refractive_index_file)
    wn_mir = wavenumber
    wn_vis = 1e7 / laser_wavelength_nm
    wn_sfg = wn_mir + wn_vis

    n1 = 1.0
    n2_mir = np.interp(wn_mir, table_wn, table_n)      # np.interp clamps at the ends
    n2_vis = np.interp(wn_vis, table_wn, table_n)
    n2_sfg = np.interp(wn_sfg, table_wn, table_n)

    mir_t = np.arcsin(n1 * np.sin(mir_angle) / n2_mir)
    vis_t = np.arcsin(n1 * np.sin(vis_angle) / np.complex128(n2_vis))
    sfg_t = np.arcsin(n1 * np.sin(sfg_angle) / n2_sfg)

    kz_mir = 1e-7 * 2 * np.pi * wn_mir * np.sqrt(n2_mir ** 2 - np.sin(mir_angle) ** 2)
    kz_vis = 1e-7 * 2 * np.pi * wn_vis * np.sqrt(np.complex128(n2_vis) ** 2 - np.sin(vis_angle) ** 2)
    kz_sfg = 1e-7 * 2 * np.pi * wn_sfg * np.sqrt(n2_sfg ** 2 - np.sin(sfg_angle) ** 2)
    sign = 1 if sfg_or_dfg == "sfg" else -1
    coherence_nm = 1 / (np.abs(kz_sfg) + np.abs(kz_vis) + sign * np.abs(kz_mir))

    L = {"coherence_length_nm": coherence_nm}
    for beam, n2, incident, transmitted in (("mir", n2_mir, mir_angle, mir_t),
                                            ("vis", np.complex128(n2_vis), vis_angle, vis_t),
                                            ("sfg", n2_sfg, sfg_angle, sfg_t)):
        L[f"xx_{beam}"], L[f"yy_{beam}"], L[f"zz_{beam}"] = _l_factors(
            n1, n2, _interfacial_index(n2, phi), incident, transmitted)

    coherence_m = 1e-9 * coherence_nm
    g = material.degeneracy
    interface = material.source == "interface"
    if polarization == "ssp":
        frame = "yyz"
        scale = (g * np.sin(mir_angle) * L["yy_sfg"] * L["yy_vis"] * L["zz_mir"] if interface else
                 g * np.cos(mir_angle) * L["yy_sfg"] * L["yy_vis"] * L["xx_mir"] * coherence_m)
    elif polarization == "sps":
        frame = "yzy"
        scale = (g * np.sin(vis_angle) * L["yy_sfg"] * L["zz_vis"] * L["yy_mir"] if interface else
                 g * np.cos(vis_angle) * L["yy_sfg"] * L["xx_vis"] * L["yy_mir"] * coherence_m)
    elif polarization == "ppp":
        frame = "zzz"
        if interface:
            scale = g * (np.cos(sfg_angle) * np.cos(vis_angle) * np.sin(mir_angle)
                         * -L["xx_sfg"] * L["xx_vis"] * L["zz_mir"]
                         + np.sin(sfg_angle) * np.sin(vis_angle) * np.sin(mir_angle)
                         * L["zz_sfg"] * L["zz_vis"] * L["zz_mir"])
        else:
            scale = (g * np.cos(sfg_angle) * np.cos(vis_angle) * np.cos(mir_angle)
                     * L["xx_sfg"] * L["xx_vis"] * L["xx_mir"] * coherence_m)
    else:
        frame, scale = "invalid", np.ones_like(wavenumber, dtype=complex)

    return Fresnel(material, L, np.asarray(scale, dtype=complex), frame)
