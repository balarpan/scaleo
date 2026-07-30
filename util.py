""" Signal processing utilities.

Metadata:
    Author: Denis Savitskiy
    Year: 2026

"""

import numpy as np
import numpy.typing as npt
import pywt
import scipy

class ScaleoError(Exception):
    pass


def ampd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Native implementation of Automatic Multiscale Peak Detection (AMPD).
    Returns indices of the detected peaks.
    """
    N = len(signal)
    L = int(np.ceil(N / 2)) - 1  # Maximum number of scales
    # Initialize the Local Maxima Scalogram (LMS) matrix
    # fill with a penalty value (uniform random number [0,1] + 1)
    LMS = np.random.uniform(0, 1, (L, N)) + 1.0 
    # Populate the Scalogram matrix
    for k in range(1, L + 1):
        for i in range(k, N - k):
            # Check if current point is a local maximum at window scale k
            if signal[i] > signal[i - k] and signal[i] > signal[i + k]:
                LMS[k - 1, i] = 0  # 0 indicates a valid local maximum
    # Row-wise summation to find the scale with the most local maxima
    row_sum = np.sum(LMS == 0, axis=1)
    gamma = np.argmax(row_sum)  # Global scaling factor (optimal scale index)
    # Slice the LMS matrix up to the optimal scale and find columns consisting only of 0s
    reduced_LMS = LMS[:gamma, :]
    peaks = np.where(np.sum(reduced_LMS, axis=0) == 0)[0]
    return peaks, LMS, gamma

def signal_denoise_DWT(signal: np.ndarray | list | tuple, wavelet_name: str, level: int | None = None) -> npt.NDArray[np.floating]:
    """Denoise signal using Discrete Wavelet Transform"""
    level = pywt.dwt_max_level(len(signal), wavelet_name) if level is None else level
    noisy_vector = np.array(signal)
    coeffs = pywt.wavedec(noisy_vector, wavelet_name, level=level, mode="symmetric")
    # Calculate a threshold to filter out noise (Universal Thresholding)
    # Uses Median Absolute Deviation (MAD) of the finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_vector)))
    # Apply soft-thresholding to all detail coefficients (leaving approximations alone)
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    # Reconstruct the clean vector
    denoised_vector = pywt.waverec(denoised_coeffs, wavelet_name, mode="symmetric")
    return denoised_vector

def signal_denoise_SWT(signal:np.ndarray | list | tuple, wavelet_name: str,
                       level: int) -> npt.NDArray[np.floating]:
    """Denoise signal using Stationary Wavelet Transform (SWT)."""
    noisy_vector = np.array(signal)
    coeffs = pywt.swt(noisy_vector, wavelet_name, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_vector)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.iswt(denoised_coeffs, wavelet_name)
    return denoised_signal

def signal_denoise_WPT(signal: np.ndarray | list | tuple, wavelet_name: str, maxlevel: int, mode: str = 'symmetric' ) -> npt.NDArray[np.floating]:
    """
    Denoise signal using Packet Wavelet Transform.

    Note: Wavelet Packet Transform (WPT) denoising naturally decreases total signal power.
    Removing small noisy pieces removes their energy from signal, which slightly lowers the overall power of the signal.
    It's trade-off: signal "volume" little quieter, but more cleaner.
    """
    noisy_vector = np.array(signal)
    # Wavelet Packet Transform Configuration
    wp = pywt.WaveletPacket(data=noisy_vector, wavelet=wavelet_name, mode=mode, maxlevel=maxlevel)
    # Calculate a global threshold (Universal Threshold / VisuShrink)
    # Sigma is estimated using the Median Absolute Deviation (MAD) of the finest detail node
    # In WPT, the last node of the highest frequency is typically named 'd' * level (e.g., 'ddd')
    finest_node = 'd' * maxlevel
    finest_coefficients = wp[finest_node].data
    sigma = np.median(np.abs(finest_coefficients)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_vector)))
    # Apply thresholding to all leaf nodes at the target level
    # Get all nodes at the bottom level of the decomposition tree
    leaf_nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    for path in leaf_nodes:
        # Extract the node data (wavelet packet coefficients)
        coef = wp[path].data
        # Apply soft or hard thresholding
        wp[path].data = pywt.threshold(coef, value=threshold, mode='soft')
    denoised_signal = wp.reconstruct(update=True)
    return denoised_signal

def signal_denoise_MRA(signal: np.ndarray | list | tuple, wavelet_name: str, level: int) -> npt.NDArray[np.floating]:
    """
    Denoise signal using Multiresolution Analysis.

    Note: MRA preserves source signal power in reconstructed signal.
    """
    noisy_vector = np.array(signal)
    # Example: for level=2 we break it down into 2 detail levels and 1 approximation
    # coeffs will contains: [Approximation_Level2, Detail_Level2, Detail_Level1]
    coeffs = pywt.mra(signal, wavelet=wavelet_name, level=level)
    # coeffs[0] is the base approximation, coeffs[1:] are details from coarse to fine
    # Denoise by keeping only smooth components (zero out high-freq details)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(noisy_vector)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    reconstructed_signal = pywt.imra(denoised_coeffs)
    return reconstructed_signal

def get_peak_bounds(signaltimeseries: np.ndarray, peak_indices: np.ndarray | list[int],
                    signal_row: int = 1, threshold_ratio: float = .40, only_boundary: bool = True) -> list[np.ndarray]:
    """
    Use this function to split signal into chunks with peak and surrounding values.

    Args:
        signaltimeseries (np.ndarray): is the array of form
                         [ [t0, t1, t2, ... tN], [value0, value1, .... valueN] ]
                         and signal_row in this case will be '1'
        peak_indices (np.ndarray | list): Index of peaks inside signaltimeseries 2D array.
        signal_row (int): Number of row with signal values. Defaults to 1 
        threshold_ratio (float): Iterate forward and backward from the peak index until 
                         the signal value drops below a set threshold (e.g., 10% of peak 
                         prominence or baseline level). Defaults to 0.20
    Returns:
        list of 2D np.ndarray with the same format as signaltimeseries.
    """
    bounds = []
    signal = signaltimeseries[signal_row,:]
    for p in peak_indices:
        peak_val = signal[p]
        threshold = peak_val * threshold_ratio
        # Scan Left
        start_idx = p
        while start_idx > 0 and signal[start_idx] > threshold:
            start_idx -= 1
        # Scan Right
        stop_idx = p
        while stop_idx < len(signal) - 1 and signal[stop_idx] > threshold:
            stop_idx += 1
        bounds.append( (start_idx, stop_idx) )
    if only_boundary:
        return [signaltimeseries[:, [start,stop]] for start,stop in bounds]
    split_points = np.array(bounds).ravel()
    all_splits = np.hsplit(signaltimeseries, split_points)
    # Extract only the targeted intervals (every odd-indexed chunk)
    # Index 0 is before the 1st start. Index 1 is start[0]:stop[0], etc.
    desired_splits = all_splits[1::2]
    return desired_splits

def kmean_timeseries(signaltimeseries: np.ndarray, window_size:int, signal_row: int = 1) -> tuple[list[float], list[np.ndarray]]:
    """
    kmean function.
    Args:
        signaltimeseries (np.ndarray): is the array of form
                         [ [t0, t1, t2, ... tN], [value0, value1, .... valueN] ]
                         and signal_row in this case will be '1'
        window_size (int): Size of sliding window. 'window_size' must be no less than the number of measurements equal to one found period.
        signal_row (int): Number of row with signal values. Defaults to 1 
    Returns:
        tuple: A tuple with list of indexes of break points and corresponding chunks of source time-series measurements.
    """
    data = signaltimeseries[signal_row,:]
    # Расчет скользящей дисперсии (окно 10 точек)
    w_size = window_size

    # Scale by data dispersion (Coefficient of Variation) and total series size
    # std_dev = data.std(ddof=1) # Sample Standard Deviation (ddof=1)
    # mean_val = data.mean()
    # cov = std_dev / mean_val if mean_val != 0 else 0
    # Formula combining cycle length, variance scaling, and series properties
    # window_size = period + int(cov * len(data) * 0.10)
    
    mean = scipy.ndimage.uniform_filter1d(data, size=w_size, mode='nearest')
    sq_mean = scipy.ndimage.uniform_filter1d(data**2, size=w_size, mode='nearest')
    rolling_var = sq_mean - mean**2
    
    # Нахождение точек изменения (градиент дисперсии)
    var_gradient = np.abs(np.gradient(rolling_var))
    peaks, _ = scipy.signal.find_peaks(var_gradient, distance=10, prominence=np.std(var_gradient))
    split_series = np.hsplit(signaltimeseries, peaks)
    
    return peaks, split_series
