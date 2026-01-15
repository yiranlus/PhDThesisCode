# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: abtem
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# In order to use this book, please install te required module using conda (or
# other software)
# ```
# conda install -c conda-forge abtem
# ```

# %% [markdown]
#
# # Configuration

# %%
#| code-fold: false
import numpy as np
potential_sampling = 0.05 # Angstrom
scan_sampling = 0.25 # Angstrom
slice_thickness = 0.5 # Angstrom

# %%
from ase.io import read
unit_cell = read("atom-models/WSe2_single_layer.cif")

from ase.visualize import view
#view(unit_cell)

# %% [markdown]
#
# Now, create a layer with the unit cell imported above. Here we can use
# CrystalPotential to reduce the usage of the RAM.

# %%
import abtem
from abtem.visualize import show_atoms
from abtem.structures import orthogonalize_cell
import proplot as pplt
import matplotlib.pyplot as plt

model, transform = orthogonalize_cell(unit_cell, return_transform=True)

unit_potential = abtem.Potential(
    model,
    sampling=potential_sampling,
    slice_thickness=slice_thickness,
    projection="finite",
    parametrization="kirkland")

fig, (ax1, ax2) = pplt.subplots(nrows=1, ncols=2)
show_atoms(model, legend=True, ax=ax1)

x_length = np.linalg.norm(model.cell[0])
y_length = np.linalg.norm(model.cell[1])
ax2.imshow(unit_potential.project().array.T,
           origin="lower",
           extent=[0, x_length, 0, y_length])

#pplt.show()
x_length, y_length

# %% [markdown]
#
# Now, create repeated cells to form an area.

# %%
repititions = (31, 18, 1)
potential = abtem.potentials.CrystalPotential(unit_potential, repititions)

print(f"There will be {len(potential)} slices.")
print(f"model dimension:")
print(f"    x: {potential.extent[0]}")
print(f"    y: {potential.extent[1]}")
print(f"Sampling rate:")
print(f"    x: {potential.sampling[0]} Å")
print(f"    y: {potential.sampling[1]} Å")

# %% [markdown]
#
# # Build Probe and Scan

# %%
probe = abtem.Probe(
    energy=80e3,
    semiangle_cutoff=20
)
probe.grid.match(potential)

print(f"electron beam properties:")
print(f"    wavelength: {probe.wavelength} Å")
print(f"real space sampling rate:")
print(f"    x: {probe.sampling[0]} Å")
print(f"    y: {probe.sampling[1]} Å")
print(f"reciprocal space sampling rate:")
print(f"    Rx: {probe.angular_sampling[0]} mrad")
print(f"    Ry: {probe.angular_sampling[1]} mrad")


# %%
probe.build().array.shape

# %%
#| label: probe-image
#| fig-cap: Intensity image of the probe and its profile through the center.
(extent_x, extent_y) = probe.extent
plot_range = 10 # Angstrom
probe_intensity = probe.build().intensity()
lineprofile = abtem.measure.probe_profile(probe_intensity)
lineprofile.show(ax=ax2)
ax2.format(xlim=(extent_x/2-plot_range, extent_x/2+plot_range))
pplt.show()

fwhm = abtem.measure.calculate_fwhm(lineprofile)
print(f"FWHM of Probe: {fwhm} Å")

# %% [markdown]
#
# # Summary

# %%
print(f"There will be {len(potential)} slices.")
print(f"model dimension:")
print(f"    x: {potential.extent[0]}")
print(f"    y: {potential.extent[1]}")
print(f"Sampling rate:")
print(f"    x: {potential.sampling[0]} Å")
print(f"    y: {potential.sampling[1]} Å")
print(f"electron beam properties:")
print(f"    wavelength: {probe.wavelength} Å")
print(f"real space sampling rate:")
print(f"    x: {probe.sampling[0]} Å")
print(f"    y: {probe.sampling[1]} Å")
print(f"reciprocal space sampling rate:")
print(f"    Rx: {probe.angular_sampling[0]} mrad")
print(f"    Ry: {probe.angular_sampling[1]} mrad")

detector = abtem.PixelatedDetector(max_angle=51.8223742450, resample="uniform")
grid_scan = abtem.GridScan(
    start=[0, 0],
    end=[0.25*100, 0.25*100],
    sampling=0.25
)

ax = abtem.show_atoms(model * repititions)
grid_scan.add_to_mpl_plot(ax)

# %%
measurements = probe.scan(grid_scan, potential=potential, detectors=detector, pbar=True)
measurements.write("IAMMeasurements_CrystalPotential.abtem.hdf5")
# %%
measurements.array.shape

# %%
