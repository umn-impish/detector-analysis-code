import pandas as pd
import numpy as np
from pathlib import Path

# Get the detector calibration files from this google drive link: https://drive.google.com/file/d/1onW_R556T6XyEUFtho1-hmgao08OSVOz/view?usp=drive_link

def calibrate_adc_bins(detector: int, temp: int, adc_bins: list | tuple | np.ndarray):
    """
    Converts ADC bins to energy using a linear calibration

    Parameters
    ----------
    detector : int
        Detector number (valid range: 1–4)
    temp : int
        Calibration temperature in Celsius, ranges from 40C to -10C
    adc_bins: array-like
        ADC bins to calibrate

    Returns
    -------
    numpy.ndarray
        calibrated energies (same shape as adc_bins)
    """

    csv_path = Path(f'D{detector}_cal_energies.csv')

    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration files not found: {csv_path}")
    
    df = pd.read_csv(csv_path, index_col=0)

    if temp not in df.index:
        raise KeyError(
            f"Temperature {temp} not found. "
            f"Available temperatures: {list(df.index)}"
        )
        
    temp_adc_bins = df.loc[temp].to_numpy(dtype=float)
    energies = df.columns.astype(float).to_numpy()

    slope, intercept = np.polyfit(temp_adc_bins, energies, deg=1)

    return slope * np.asarray(adc_bins) + intercept
