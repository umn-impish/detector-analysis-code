import random
import ctypes
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from pathlib import Path
from impisc.impisc.et_daqbox import daq_box_api as dba

# Get this from the `variables.env` file
NUM_BINS = 1000
NUM_DETECTORS = 4

bins = np.arange(NUM_BINS + 1)
mids = bins[:-1] + np.diff(bins) / 2

class ScienceRecord(ctypes.LittleEndianStructure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = (
        ("timestamp", ctypes.c_uint32),
        ("frame", ctypes.c_uint8),
        ("spectra", NUM_DETECTORS * (NUM_BINS * ctypes.c_uint32)),
    )
    

PACKET_SIZE = ctypes.sizeof(ScienceRecord)

def gauss(x: np.ndarray, mu, amp, wid):
    return amp * np.exp(-(x - mu)**2 / wid**2)

def energy_resolution(mu: float, width: float) -> float:
    """Return FWHM energy resolution (%)."""
    return abs((2 * np.sqrt(np.log(2)) * width / mu) * 100)

def fit_peak(mids, counts, fit_range, mu_guess, amp_guess, width_guess):
    mask = (mids >= fit_range[0]) & (mids <= fit_range[1])

    popt, pcov = sco.curve_fit(gauss, mids[mask], counts[mask], p0=[mu_guess, amp_guess, width_guess])

    mu, amp, width = popt

    return {
        "mu": mu,
        "amp": amp,
        "width": width,
        "fwhm": energy_resolution(mu, width),
        "fit": popt,
        "cov": pcov,
    }

def spec_parse(fn, PACKET_SIZE):
    spectra = []

    with open(fn, "rb") as f:
        while True:
            data = f.read(PACKET_SIZE)
            if len(data) == 0:
                # Hit end of file
                break
            spectra.append(dba.parse_spectrum_packet(data))
        
    return np.array(spectra)

def wf_parse(fn, PACKET_SIZE):
    waveforms = []

    with open(fn, "rb") as f:
        while True:
            data = f.read(PACKET_SIZE)
            if len(data) == 0:
                # Hit end of file
                break
            waveforms.append(dba.parse_waveform_packet(data))
    return waveforms

def science_spec_parse(fn, PACKET_SIZE):
    spectra = []
    timestamps = []
    frames = []


    with open(fn, "rb") as f:
        while True:
            data = f.read(PACKET_SIZE)
           
            if len(data) == 0:
                # Hit end of file
                break
            
            # Skip incomplete records
            if len(data) != PACKET_SIZE:
                print("Incomplete record skipped")
                continue

            record = ScienceRecord.from_buffer_copy(data)
            timestamps.append(record.timestamp)
            frames.append(record.frame)
            spectra.append(record.spectra)

    return np.array(spectra)

def debug_spectra_array(debug_files):
    PACKET_SIZE = 8000
    debug_spectra = []

    for fn in debug_files:
        spectra = spec_parse(fn, PACKET_SIZE)
        debug_spectra.append(spectra)

    print('one spectrum shape:', spectra.shape)
    print('There are spectra for', len(debug_spectra), 'spectra types')

    return debug_spectra

def science_spectra_array(science_files):
    PACKET_SIZE = ctypes.sizeof(ScienceRecord)
    spectra_types = []

    for fn in science_files:
        spectra = science_spec_parse(fn, PACKET_SIZE)
        spectra_types.append(spectra)

    # print('one spectrum shape:', spectra.shape)
    print('There are spectra for', len(spectra_types), 'spectra types')

    return spectra_types

def multi_row_col_spectra(n_y, n_x, x_size, y_size, range_, mu_, amp, w_, debug_spectra, types, detectors):
    fig, ax = plt.subplots(n_y,n_x, figsize=(x_size,y_size), constrained_layout=True)

    axes = np.atleast_1d(ax).ravel()

    for i, axis in enumerate(axes):
        if i >= len(debug_spectra):
            axis.set_visible(False)
            continue

        for det in range(NUM_DETECTORS):

            summation = debug_spectra[i].sum(axis=0)[det]
            fit = fit_peak(mids, summation, range_[i][det], mu_[i][det], amp[i][det], w_)


            axis.stairs(summation, bins, label=f'{types[i]}, {detectors[det]}')
            axis.plot(mids, gauss(mids, fit["mu"], fit["amp"], fit["width"]), "k--",
                      label=f"ER = {fit['fwhm']:.1f}%, $\mu$={fit['mu']:.1f}, $a$={fit['amp']:.1f}")
        axis.set_title(str(types[i]))
        axis.set_xlabel("ADC Bin")
        axis.set_ylabel("Counts")
        axis.legend(fontsize=8)

    return fig, ax

def science_multi_row_col_spectra(n_y, n_x, x_size, y_size, science_spec, temp_range, detectors, limit, title):
    fig, ax = plt.subplots(n_y, n_x, figsize=(x_size, y_size), constrained_layout=True)
    axes = np.atleast_1d(ax).ravel()

    for i, axis in enumerate(axes):
        if i >= len(science_spec):
            axis.set_visible(False)
            continue
        for det in range(NUM_DETECTORS):
            summation = science_spec[i].sum(axis=0)[det]
            axis.stairs(summation, bins, label=detectors[det])

        axis.set_title(str(temp_range[i]))
        axis.set_xlim(-10, limit)
        axis.grid()
        axis.minorticks_on()
        axis.legend(fontsize=8)
    

def fit_science_spectra(temp_range, range1, mu1, amp1, range2, mu2, amp2, width, spectra_types, channel, st, end, title):
    label = temp_range

    fig, ax = plt.subplots(figsize=(14, 8))
    fit_mu1 = []
    fit_fwhm1 = []
    fit_mu2 = []
    fit_fwhm2 = []

    for i in range(st, end):

        summation = spectra_types[i].sum(axis=0)[channel]

        fit1 = fit_peak(mids, summation, range1[i], mu1[i], amp1, width)
        fit2 = fit_peak(mids, summation, range2[i], mu2[i], amp2, width)

        fit_mu1.append(fit1["mu"])
        fit_mu2.append(fit2["mu"])
        fit_fwhm1.append(fit1["fwhm"])
        fit_fwhm2.append(fit2["fwhm"])

        ax.stairs(summation, bins)
        ax.plot(mids, gauss(mids, fit1["mu"], fit1["amp"], fit1["width"]), 'k--',
                label=f"{label[i]}, ER= {fit1['fwhm']:.1f}%, $\mu$={fit1['mu']:.1f}, $a$={fit1['amp']:.1f}")
        ax.plot(mids, gauss(mids, fit2["mu"], fit2["amp"], fit2["width"]), '--', color='gray',
                label=f"{label[i]}, ER= {fit2['fwhm']:.1f}%, $\mu$={fit2['mu']:.1f}, $a$={fit2['amp']:.1f}")
    ax.set(xlabel='integral ADC bin', ylabel=f'30s counts', title=f'{title}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_ylim(0,20000)
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    return fit_mu1, fit_fwhm1, fit_mu2, fit_fwhm2

def interpolate_exptrapolate(df, temp_range, energies, target_energies, temp_interpolate_kind=None, models=None):
    # interpolate within given temperature ranges
    if temp_interpolate_kind=='lsm':
        model30, model60, model80, model122 = models
        df_interp = pd.DataFrame({
            'Temperature': temp_range,
            'peak30_fit':  model30.predict(temp_range.reshape(-1,1)),
            'peak60_fit':  model60.predict(temp_range.reshape(-1,1)),
            'peak80_fit':  model80.predict(temp_range.reshape(-1,1)),
            'peak122_fit':model122.predict(temp_range.reshape(-1,1))
        })
        
        df_interp.set_index('Temperature', inplace=True)
    else:
        df_interp = df.reindex(temp_range).interpolate(method='linear')

    # interpolate and extrapolate for target energies
    interp_rows = []

    for temp, rows in df_interp.iterrows():
        f_energies = interp1d(energies, rows.values, kind='linear', fill_value='extrapolate')
        interp_rows.append(f_energies(target_energies))

    df_energies = pd.DataFrame(
        interp_rows,
        index = df_interp.index,
        columns = [target_energies]
    )

    return df_energies




