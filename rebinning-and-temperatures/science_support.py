import bz2
import ctypes
import pathlib

import astropy.time as atime
import numpy as np
import numpy.typing as npt

# The number of science bins lives in `/data/config/variables.env`
# onboard IMPISH
NUM_BINS = 1000
NUM_DAQBOX_CHANNELS = 4


class ScienceRecord(ctypes.LittleEndianStructure):
    """A single DAQBOX science record saved on IMPISH.
    Each record comprises of the information from the 8000B DAQBOX packet,
    as well as the current timestamp corresponding to that packet.

    The UNIX timestamp corresponds to the **previous** second.
    It is synchronized via PPS.
    To reconstruct the **left** time bin edge of the spectra in the record,
    add the frame number divided by the frame rate.

    In other words,
        - `left_time_edge = timestamp + (frame / 32)`
        - `right_time_edge = timestamp + ((frame + 1) / 32)`
    """

    _pack_ = 1
    _layout_ = "ms"
    _fields_ = (
        ("timestamp", ctypes.c_uint32),
        ("frame", ctypes.c_uint8),
        ("spectra", NUM_DAQBOX_CHANNELS * (NUM_BINS * ctypes.c_uint32)),
    )


def load_science_file(fn: str | pathlib.Path) -> dict[str, list[float] | list[int]]:
    """Take a file path (str or Path) and extract the science packets out of it.
    Returns a dict of:
        - The spectra per channel; shape = (N, 4, B) where N is number of time bins,
          and B is number of ADC bins
        - The time bins corresponding to the spectrogram edges.

    The number of ADC bins is determined by the current flight configuration,
    stored onboard IMPISH in `/data/config/variables.env` in the SCIENCE_REBIN_BINS
    environment variable.
    If that variable is empty, all 1000 DAQBOX bins are saved to science files."""
    left_edges = list()
    spectra = list()
    with bz2.open(fn, "rb") as f:
        while True:
            data = f.read(ctypes.sizeof(ScienceRecord))
            if len(data) == 0:
                break
            packet = ScienceRecord.from_buffer_copy(data)
            left_edges.append(packet.timestamp + (packet.frame / 32))
            these_spec = list()
            for s in packet.spectra:
                these_spec.append(list(int(x) for x in s))
            spectra.append(these_spec)

    left_edges = left_edges + [left_edges[-1] + 1 / 32]
    return {
        "spectra": np.array(spectra),
        "time_bins": atime.Time(left_edges, format="unix"),
    }


def rebin_time_energy(
    science_records: dict[str, np.ndarray[tuple[int, int], int] | atime.Time],
    resulting_adc_bins: int,
    time_combine_factor: int,
) -> dict[str, np.ndarray[tuple[int, int], int] | atime.Time]:
    """Given a `dict` of science records,
    combine energy and time bins along appropriate axes,
    and return a `dict` with the rebinned data.

    Doesn't do interpolation, but does use `numpy` to help speed up the array manipulations."""
    # Cut off the time bins that don't align with a full second
    num_time_bins = science_records["spectra"][:, 0, :].shape[0]
    time_limit = num_time_bins - (num_time_bins % time_combine_factor)

    # Do the bin slicing on time midpoints; they have the same shape as the
    # spectrogram's time axis.
    tb: atime.Time = science_records["time_bins"]
    dt = tb[1] - tb[0]
    time_mids: atime.Time = tb[:-1] + dt / 2
    time_mids = time_mids[:time_limit:time_combine_factor]
    reb_time_bins = np.concatenate(
        (time_mids - dt / 2, atime.Time([time_mids[-1] + dt / 2]))
    )

    rebinned_sgram = list()
    for channel in range(4):
        # Select the spectrum for this particular channel
        spec = science_records["spectra"][:, channel, :]

        # Sum the energy axis into the shape we want
        rebinned = spec.reshape(spec.shape[0], resulting_adc_bins, -1).sum(axis=2)

        # Cut off the spectrogram at an even multiple of the time binning we want
        rebinned = rebinned[:time_limit]
        rebinned = rebinned.reshape(time_combine_factor, -1, resulting_adc_bins).sum(
            axis=0
        )
        rebinned_sgram.append(rebinned)

    return {"spectra": np.array(rebinned_sgram), "time_bins": reb_time_bins}
