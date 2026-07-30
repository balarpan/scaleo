""" Signal processing utilities.

Metadata:
    Author: Denis Savitskiy
    Year: 2026

"""

import numpy as np
import scipy
import pywt
import matplotlib.pyplot as plt
import matplotlib.patheffects as plt_pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import warnings
from . import util
from .util import ScaleoError as Excp

class CWT():
    _spectrum_text = {
        'amp': {'title': 'Amplitude Spectrum', 'cbar': r'abs(CWT)'},
        'real': {'title': 'Real Spectrum', 'cbar': r'real(CWT)'},
        'imag': {'title': 'Imaginary Spectrum', 'cbar': r'imaginary(CWT)'},
        'power': {'title': 'Power Spectrum', 'cbar': r'abs(CWT)$^2$'},
    }
    _ticktype_text = {
        'period': {'ylabel': 'Period'},
        'freq': {'ylabel': 'Frequency'},
        'scale': {'ylabel': 'Scale'},
    }
    _spectrum_values = {
        'amp': lambda coefs: np.abs(coefs),
        'real': lambda coefs: np.real(coefs),
        'imag': lambda coefs: np.imag(coefs),
        'power': lambda coefs: np.power(np.abs(coefs), 2)
    }
    IMG_MAX_ROWS = 2000
    IMG_MAX_COLS = 2000
    
    def __init__(self, signal_data: list | np.ndarray, sampling_freq:float, time_ticks: list | np.ndarray | None = None,
                 wavelet_name: str = 'cmor1.5-1.0', xlabel: str | None = None):
        """
        Initialize with signal data.

        Args:
            signal_data (list or np.ndarray): The input signal values.
            sampling_freq (float): The sampling frequency of the signal.
            time_ticks (list or np.ndarray or None, optional): The time ticks corresponding to the signal values.
                              If `None` then will be initialized as array of values [0...1]. Defaults to `None`.
            wavelet_name (str, optional): The name of the wavelet to use. Defaults to 'cmor1.5-1.0'.
                              See ```pywt.wavelist(kind='continuous')``` to get full list of wavelets.
            xlabel (str or None, optional): A custom text for x-axis (time-axis). Defaults to `None`.
        """
        if len(signal_data) < 8:
            raise Excp("Signal data must contain at least 8 samples!")
        self.signal = np.array(signal_data)
        if time_ticks is not None and len(signal_data) != len(time_ticks):
            raise Excp("Signal length and time length must be the same size!")
        self.time = np.array(time_ticks) if time_ticks is not None else np.arange(len(signal_data))
        self.fs = sampling_freq
        self.wavelet = pywt.ContinuousWavelet(wavelet_name)
        self.xlabel = xlabel
        self.scales = None
        self._cwt_result = None  # (coefs, freqs)
        self._cwt_result_params = None
        self._img = None

    def cwt(self,
            scales: np.ndarray | list | tuple[float,float] | None = None,
            periods: np.ndarray | list | tuple = None, min_max_freq: tuple[float,float] | None = None,
            method: str = 'conv') -> tuple[np.ndarray, np.ndarray]:
        """
        Proceed Continuous Wavelet Transform via pywt.cwt(...) and return results. First not `None` param will be used.
        If all parameters is `None` then default scale range will be applied.

        Args:
            scales (tuple, optional): Array of scales for CWT. If `None` and other inputs is `None`, then an attempt will be made to select scales based on fs and the length of the observation time series.
                                       Note: If all param's is `None`, then the range of frequencies between [1/time_window, fs/2] will be used.
            periods (set, optional): Array of periods to compute appropriate scales for CWT. Defaults to `None`.
            min_max_freq(tuple, optional): A tuple of two floats: the minimum and maximum frequencies to generate range between and use in computation.
                                           Applicable only if both scales=`None` and periods=`None`. Defaults to `None`.
            method(str): cwt method. Can be 'conv' or 'fft'. See https://pywavelets.readthedocs.io/en/latest/ref/cwt.html Defaults to 'conv'

        Returns:
            A tuple containing the CWT coefficients and according frequencies which is result of calling pywt.cwt(..)   
        """
        params_now = locals()
        params_now.pop('self', None)
        if self._cwt_result is None or self._cwt_result_params is not None and self._cwt_result_params != params_now:
            c_scales = self._cwt_scales_helper(scales=scales, min_max_freq=min_max_freq, periods=periods)
            self._cwt_result = self._cwt(scales=c_scales, method=method)
            self._cwt_result_params = params_now
        return self._cwt_result

    def _cwt(self, scales: np.ndarray | list[float], method: str = 'conv') -> tuple[np.ndarray, np.ndarray]:
        self.scales = scales
        # Perform Continuous Wavelet Transform
        coefs, freqs = pywt.cwt(self.signal, scales, self.wavelet, sampling_period= 1./self.fs, method=method)
        return (coefs, freqs)

    def _coi(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wavelet = self.wavelet
        time, scales = self.time, self.scales
        t0 = time[0]
        dt = 1. / self.fs
        # Get wavelet Fourier wavelengths. A wavelet scale (width) corresponds to a specific Fourier wavelength.
        # To convert a scale s into its Fourier equivalent λ
        # λ = s * dt / f_center
        f_center = pywt.central_frequency(wavelet)
        mask = t0 + (scales * dt / f_center)
        mid_idx = time.shape[0] // 2          # left half for both odd-length arrays and even-length arrays
        coi_x = mask[ mask < time[mid_idx] ]
        coi_y = y[:len(coi_x)]                # sometimes number of scales is bigger than the middle of timeline ticks
        if t0 > 0 or t0 < coi_x[0]:
            coi_x = np.concat(([t0],coi_x))
            coi_y = np.concat(([y[0]], coi_y))
        coi_x = np.concat((coi_x, time[-1] - coi_x[::-1] + t0))
        coi_y = np.concat((coi_y, coi_y[::-1]))
        return (coi_x, coi_y)
    
    def draw_signal(self, ax = None, show_xlabel: bool = False):
        """Draw the source signal."""
        time = self.time
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 1.2)) if ax is None else (None, ax)
        ax1.plot(time, self.signal)
        ax1.tick_params(axis='both', labelsize='small')
        if show_xlabel:
            ax1.set_xlabel(self.xlabel if self.xlabel else 'Time')
        if fig1 is not None:
            # ax1.set_xlim(time[0], time[-1])
            plt.tight_layout()
            plt.show()

    def _scaleo_y_vals(self, ticktype: str, scales: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        return scales if 'scale' == ticktype else 1/freqs if 'period' == ticktype else freqs

    def scalogram(self, ax=None, ticktype: str = 'period', spectrum: str = 'amp', cmap = 'PRGn', logscale: bool = False,
                   figsize: tuple = (10,5), cbar: bool = True, cbar_ax = None, coi:bool = True, coi_alpha: float = .35,
                  twinticks: str | None = None, ygrid: bool = True, vmin: float | None = None, vmax : float | None = None):
        """
        Draw a Scalogram.

        Args:
            ax (Any): Matplotlib axes. If `None`, then own Matplotlib canvas will be created. Defaults `None`.
            ticktype (str, optional): Type of Y-axes. One of the following predefined values: 'scale', 'period', 'freq'. Defaults 'period'.
            spectrum (str, optional): Type of spectrum image. One of the following predefined values: 'amp' for amplitude,
                             'real' for use real part, 'imag' for use imaginary part of values, 'power' for spectrum based on power of signal. Defaults 'amp'.
            cmap (Any, optional): Matplotlib colormap. Defaults 'PRGn'. See: https://matplotlib.org/stable/gallery/color/colormap_reference.html 
            logscale (bool, optional): If `True` logarithmic scale will be used.
            figsize (tupl, optional): Matplotlib 'figsize' parameter. Defaults `(10,5)`.
            cbar (boo, optionall): If `True` then cbar legend will be append. If 'twinticks' parameter is not `None` and 'cbar_ax' not provided, then this parameter is omitted. Defaults `True`.
            cbar_ax (Any, optional): If provided, then legend will be drawed in this matplotlib context. Defaults `None`.
            coi (bool, optional): If `True`, then the Cone of Influence will be drawn above scalogram. Defaults `True`.
            coi_alpha (floatm optional): Opacity for COI. Defaults 0.35
            twinticks (str, optional): If not `None` second Y-axes to the right will be append. Values is predefined and same as for 'ticktype' parameter. Defaults `None`.
            ygrid (bool, optional): If `True` then grid lines will be added to plot. Defaults `True`
            vmin (float or None, optional): If not `None`, then use this value as minimum for image color and colorbar. Defaults `None`.
            vmax (float or None, optional): If not `None`, then use this value as maximum for image color and colorbar. Defaults `None`. 
        Returns:
            None
        """
        self._chk_prm('ticktype', ticktype, list(CWT._ticktype_text.keys()))
        self._chk_prm('twinticks', twinticks, list(CWT._ticktype_text.keys()) + [None])
        self._chk_prm('spectrum', spectrum, list(CWT._spectrum_values.keys()))
        fig1, ax2 = plt.subplots(1, 1, figsize=figsize) if ax is None else (None, ax)
        wavelet = self.wavelet
        coefs, freqs = self.get_cwt
        time, scales = self.time, self.scales
        fs = self.fs
        ax2.locator_params(axis='y', nbins=30 if float(ax2.get_window_extent().height) > 250 else 15)
        xmesh = time
        ymesh = self._scaleo_y_vals(ticktype, scales, freqs)
        ylim = ymesh[[-1,0]]
        ax2.set_ylim(*ylim)
        ax2.set_ylabel(CWT._ticktype_text[ticktype]['ylabel'])
        ax2.tick_params(axis='both', labelsize='small')
        vals = CWT._spectrum_values[spectrum](coefs)

        if len(xmesh) > CWT.IMG_MAX_ROWS or len(ymesh) > CWT.IMG_MAX_COLS:
            warnings.warn(f"Reduced image to max allowed resolution {CWT.IMG_MAX_ROWS * 2}x{CWT.IMG_MAX_COLS * 2} pixels", RuntimeWarning)
        _img_x_steps = len(xmesh) // CWT.IMG_MAX_ROWS if len(xmesh) > CWT.IMG_MAX_ROWS else 1
        _img_y_steps = len(ymesh) // CWT.IMG_MAX_COLS if len(ymesh) > CWT.IMG_MAX_COLS else 1
        self._img = ax2.pcolormesh(
            xmesh[::_img_x_steps], ymesh[::_img_y_steps], vals[::_img_y_steps, ::_img_x_steps],
            cmap=cmap, norm=None, vmin=vmin, vmax=vmax)
        ax2.set_title(CWT._spectrum_text[spectrum]['title'] + f". Wavelet '{self.get_wavelet_name}'")
        formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
        ax2.yaxis.set_major_formatter(formatter)
        if logscale:
            self._log_y_prm(ax2)
        atwin = None
        if twinticks and twinticks in CWT._ticktype_text:
            v_new = lambda v: self._cwt_param_convert(ticktype, twinticks, v)
            iv_new = lambda v: self._cwt_param_convert(twinticks, ticktype, v)
            atwin = ax2.secondary_yaxis('right', functions=(v_new, iv_new))
            if logscale:  # set log scale first, and then call ._set_yticks.
            #     atwin.set_yscale('log')
                self._log_y_prm(atwin, False)
            atwin.tick_params(axis='y', which='both', labelsize='small', colors='darkblue')
            atwin.set_ylabel(CWT._ticktype_text[twinticks]['ylabel'], fontname='cursive', color='darkblue')
        ax2.set_xlabel(self.xlabel if self.xlabel else "Time.")
        #Grid for Y ticks
        if ygrid:
            ax2.grid(axis='y', which='both', linestyle=(0, (5, 15)), linewidth=0.5, color='gray', alpha=0.45)
        # Cone Of Influence (eliminate "bad" values near the borders of data)
        if coi:
            coi_y = scales if ticktype =='scale' else 1./freqs if ticktype == 'period' else freqs
            coi_x, coi_y = self._coi(coi_y)
            ax2.fill_between(coi_x, np.full(len(coi_x), ymesh[-1]), coi_y,
                             color='tab:gray', alpha=coi_alpha, hatch='X', facecolor='none' )
            ax2.fill_between(coi_x, np.full(len(coi_x), ymesh[-1]), coi_y,
                             color='k', alpha=coi_alpha - coi_alpha/100 )
        if cbar and twinticks is None:
            cax = cbar_ax if cbar_ax else ax2
            self.colorbar(cax, CWT._spectrum_text[spectrum]['cbar'])
        if fig1 is not None:
            fig1.tight_layout()
            plt.show()

    def colorbar(self, ax, cbarlabel: str | None = None):
        """Draw a color bar for Scalogram"""
        ax.tick_params(axis='both', labelsize='small')
        colorbar = plt.colorbar(self._img, orientation='vertical', ax=ax, aspect=30, fraction=.05)
        if cbarlabel:
            colorbar.set_label(cbarlabel)

    def complex_plot(self, title: str | None = None, figsize=(14,7), cmap: str = 'viridis', ticktype: str ='period', twinticks: str | None = None,
                     spectrum: str = 'amp', coi: bool = True, coi_alpha: float = .35, logscale: bool = False, detect_peaks: bool = False):
        """Draw a complex plot with source signal and corresponding aligned Scalogram. See description of ```scalogram()``` function."""
        fig1, axs = plt.subplots(2, 2, figsize=figsize, layout='constrained', gridspec_kw={'height_ratios': [1, 6], 'width_ratios': [15, 1]});  
        ax1 = axs[0,0]
        ax2 = axs[1,0]
        ax1.sharex(ax2)
        self.draw_signal(ax=ax1)
        if title:
            ax1.set_title(title)
        ax1.tick_params(axis='y', labelleft=False)
        # Create an aligned axis divider
        divider = make_axes_locatable(axs[1,1])
        cax = divider.append_axes("left", size="75%", pad=0.05)
        axs[0,1].axis('off')
        axs[1,1].axis('off')
        self.scalogram(ax=ax2, cbar=False, cmap=cmap, spectrum=spectrum, ticktype=ticktype, twinticks=twinticks, coi=coi, coi_alpha=coi_alpha, logscale=logscale)
        if detect_peaks and ticktype=='period':
            signal_peaks = self.detect_power_peaks_bounds()
            path_effects=[plt_pe.Stroke(linewidth=1.1, alpha=.65, foreground='k'), plt_pe.Normal()]
            line_style = {'color': 'red', 'alpha': .65, 'linestyle': (0, (5, 7)), 'linewidth':.9, 'path_effects': path_effects}
            for peak in signal_peaks:
                if len(peak[1]) == 0:
                    ax2.axhline(y=peak[0], label=f'Detected period ({round(peak[0],4) })', **line_style)
                    continue
                for bound in peak[1]:
                    ax2.hlines(y=peak[0], xmin=bound[0], xmax=bound[1], **line_style) 
            axs[0,1].text(1, 1, "Detected periods:\n" + '\n'.join([f"{x[0]:.4g}" for x in signal_peaks]),  ha='right', va='top', transform=axs[0,1] .transAxes)
        fig1.colorbar(self._img, cax=cax, label=CWT._spectrum_text[spectrum]['cbar'], aspect=30, fraction=.05)
        plt.show()

    def detect_power_peaks(self, spect_dist: int = 10) -> np.ndarray:
        """
        This function attempts to detect peaks in signal power and transform result into period values.
        Better to use after explicit call of ```cwt()```
        """
        coefs, freqs = self.get_cwt
        power = np.abs(coefs) ** 2
        global_wavelet_spectrum = np.mean(power, axis=1) # Average power over time
        # Convert to Periods. Ignore divide by zero if 0 frequency exists.
        valid_mask = freqs > 0
        periods = 1.0 / freqs[valid_mask]
        spectrum_to_search = global_wavelet_spectrum[valid_mask]
        # Autodetect Dominant Periods. Find peaks in the wavelet spectrum
        peaks, _ = scipy.signal.find_peaks(spectrum_to_search, distance=spect_dist)
        
        dominant_periods = periods[peaks]
        return dominant_periods

    def detect_power_peaks_bounds(self, threshold: float = .40, spect_dist: int = 10) -> list[float]:
        time = self.time
        coefs, freqs = self.get_cwt
        f_freqs = 1. / self.detect_power_peaks()
        # find nearest indices in unsorted array
        targets_col = np.array(f_freqs)[:, np.newaxis]
        coefs_idx = np.abs(freqs - targets_col).argmin(axis=1)
        ret = []
        for c_idx in coefs_idx:
            period_data = np.abs(coefs[c_idx])
            p_peaks, _ = scipy.signal.find_peaks(period_data, distance=spect_dist, prominence=np.std([period_data]))
            ts = np.stack((time, period_data))
            peak_boundaries = util.get_peak_bounds(ts, p_peaks, threshold_ratio=threshold, only_boundary=True)
            # peak_boundaries is 2D ndarray: first row - time, second coefs values in this scale. We need only time row
            peak_boundaries = [x[0,:] for x in peak_boundaries ] 
            ret.append( (1./freqs[c_idx], peak_boundaries))
        return ret

    def _periods2scales(self, periods: np.ndarray | list | tuple) -> np.ndarray:
        """Convert given periods to scales.

        Args:
          periods: (np.ndarray/list/tuple): Array of increasing values of periods. Minimum value should be larger than the duration of two data samples. 

        Returns:
          np.ndarray of corresponding scales.

        Raises:
          scaleo.util.ScaleoError: If smallest periods is smaller or equal of duration of two data samples.
        """
        fs = self.fs
        dt = 1 / fs
        wavelet = self.wavelet
        p = np.array(periods) if isinstance(periods, (list, tuple)) else periods
        if p.min() <= 2*dt:
            raise Excp("Minimum value of periods should be larger than the duration of two data samples.")
        # for a scale value of 's' and a wavelet Central frequency 'C', the period 'p' is: p = s / C
        return (p/dt) * pywt.central_frequency(wavelet)

    def _cwt_scales_helper(self, scales: np.ndarray | list | tuple | None = None, min_max_freq: tuple[float, float] | None = None,
                           periods: np.ndarray | list | tuple | None = None) -> np.ndarray:
        """Helper function to convert mutually exclusive input limits into proper scales array"""
        if scales is not None:
            return np.array(scales)
        if periods is not None:
            return self._periods2scales(periods)
        if min_max_freq is not None:
            return self._default_scales(min_max_freq=min_max_freq)
        return self._default_scales()

    def _default_scales(self, min_max_freq=None) -> np.ndarray:
        fs = self.fs
        signal = self.signal
        nData, window = len(signal), len(signal)/fs
        if min_max_freq:
            min_freq, max_freq = min_max_freq
        else:
            min_freq = 1 / window
            max_freq = fs / 2
        # Create a logarithmic sequence of frequencies to analyze
        # Normalize by the sampling frequency fs
        nv = 10  # Number of Voices
        num_scale = int(nv * np.log2(nData // 2))
        frequencies = np.geomspace(min_freq, max_freq, num=num_scale, endpoint=False) / fs
        # Convert these frequencies into the correct PyWavelets 'widths' 
        scales = pywt.frequency2scale(self.wavelet, frequencies)
        return scales

    def _log_y_prm(self, ax, set_logscale: bool = True):
        """Set display in log scale and adjust params for Y-axis"""
        if set_logscale:
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=12))
        ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # Adjust formatting to standard numbers instead of scientific notation
        ax.tick_params(axis='y', which='minor', labelsize='small')

    def _cwt_param_convert(self, p_from: str, p_to: str, inp_vals: np.ndarray) -> np.ndarray:
        """Convert from one CWT param value to another. E.g. from scale to period."""
        if p_from not in CWT._ticktype_text or p_to not in CWT._ticktype_text:
            raise Excp("Parameter must be one of: ", ','.join([str(x) for x in CWT._ticktype_text.keys()]))
        vals = inp_vals.astype(float)
        def one_over(x):
            x = np.array(x, float)
            near_zero = np.isclose(x, 0)
            x[near_zero] = np.inf
            x[~near_zero] = 1. / x[~near_zero]
            return x
        wavelet = self.wavelet
        fs = self.fs
        def s2f(v):
            near_zero = np.isclose(v, 0)
            v[near_zero] = np.inf
            fnl = lambda x: pywt.scale2frequency(wavelet, x) 
            fn = np.vectorize(fnl)
            v[~near_zero] = fn(v[~near_zero])
            return v
        def f2s(v):
            near_zero = np.isclose(v, 0)
            v[near_zero] = np.inf
            fnl = lambda x: pywt.frequency2scale(wavelet, x) 
            fn = np.vectorize(fnl)
            if len(near_zero) > 0:
                v[~near_zero] = fn(v[~near_zero])
            return v
        # key = transform from , index (transform to): 0='scale', 1='period', 2='freq'. Input and output is always an np.ndarray
        trans_idx = lambda t: 0 if 'scale'==t else 1 if 'period'==t else 2
        trans = {
            'scale': [lambda v: v, lambda v: one_over(s2f(v) * fs), lambda v: s2f(v) * fs],
            'period': [lambda v: f2s(one_over(v/fs)), lambda v: v, lambda v: one_over(v)],
            'freq': [lambda v: f2s(one_over(v/fs)), lambda v: one_over(v), lambda v: v]
        }
        return trans[p_from][trans_idx(p_to)](vals)

    def denoise_DWT(self, wavelet_name: str, level: int | None = None) -> np.ndarray:
        """Denoise signal using Discrete Wavelet Transform"""
        return util.signal_denoise_DWT(self.signal, wavelet_name, level)

    def denoise_SWT(self, wavelet_name: str, level: int) -> np.ndarray:
        """Denoise signal using Stationary Wavelet Transform (SWT)."""
        return util.signal_denoise_SWT(self.signal, wavelet_name, level)

    def denoise_WPT(self, wavelet_name: str, maxlevel: int, mode: str = 'symmetric' ) -> np.ndarray:
        """Denoise signal using Packet Wavelet Transform."""
        return util.signal_denoise_WPT(self.signal, wavelet_name, maxlevel, mode)
    
    def denoise_MRA(self, wavelet_name: str, level: int) -> np.ndarray:
        """Denoise signal using Multiresolution Analysis."""
        return util.signal_denoise_MRA(self.signal, wavelet_name, level)

    @property
    def get_cwt(self):
        if self._cwt_result is None:
            self._cwt_result = self.cwt()
        return self._cwt_result

    @property
    def get_wavelet(self):
        return self.wavelet

    @property
    def get_wavelet_name(self):
        return self.get_wavelet.name

    def _chk_prm(self, name: str, val, keys: list[str|None]):
        if val not in keys:
            raise Excp(f"{name} value must be one of: {', '.join([f"'{str(x)}'" for x in keys])}. Provided: '{val}'")

    def __repr__(self):
        return f"CWT(sampling_freq={self.fs!r}, wavelet_name={self.get_wavelet_name!r}, xlabel={self.xlabel!r})"

    def __str__(self):
        return f"Class CWT with {len(self.signal)} samples of signal. {f"Computed for {len(self.scales)} input scales. " if len(self.get_cwt[0]) else ''}{repr(self)}"

