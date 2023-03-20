import numpy as np
import scipy as sp

from scipy.fft import fftshift
from matplotlib import pyplot as plt

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from pdb import set_trace as keyboard

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        #cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        #cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        #ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        #ax.set_ylim([ydata - cur_yrange*scale_factor,
        #             ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

class pd_gen:
    def __init__(self, fc_MHz=0, fs_MSPS=0, dc_percent=0, prf_KHz=0, subframe_length_pulses=0, nfft=32768):
        self.fc_MHz = fc_MHz
        self.fs_MSPS = fs_MSPS
        self.dc_percent = dc_percent
        self.prf_KHz = prf_KHz
        self.pri_s = 1/(1000*prf_KHz)
        self.subframe_length_pulses = subframe_length_pulses
        self.nfft = 32768

    def get_fc(self):
        return self.fc_MHz

    def set_fc(self, fc_MHz):
        self.fc_MHz = fc_MHz

    def get_fs(self):
        return self.fs_MSPS

    def set_fs(self, fs_MSPS):
        self.fs_MSPS = fs_MSPS

    def get_dc(self):
        return self.dc_percent

    def set_dc(self, dc_percent):
        self.dc_percent = dc_percent

    def get_prf(self):
        return self.prf_KHz

    def set_prf(self, prf_KHz):
        self.prf_KHz = prf_KHz

    def get_pri(self):
        return self.pri_s

    def set_pri(self, pri_s):
        self.prs_s = pri_s

    def calculate_samples_per_pulse(self):
        samples_per_pulse = np.round(self.fs_MSPS*1e6*self.pri_s).astype(int)
        return samples_per_pulse

    def calculate_total_length(self, samples_per_pulse):
        """Caclulate total number of samples based off # of subframe pulses
        Total samples = Fs (MS/s) * Pulse time (s) * number of pulses"""
        nsamps = samples_per_pulse*self.subframe_length_pulses
        return nsamps

    def gate_cw(self, cw, samples_per_pulse):
        """To create the PD signal, we will gate the input complex exponential
        wave according to what our desired pw/DC is"""
        nsamps = len(cw)
        off_samples = np.round((1-self.dc_percent/100)*samples_per_pulse).astype(int)
        start_off_sample = samples_per_pulse-off_samples

        gated_cw = np.zeros((nsamps,)) + 1j*np.zeros((nsamps,))
        for ii in np.arange(self.subframe_length_pulses):
            gated_cw[samples_per_pulse*ii:samples_per_pulse*ii+start_off_sample] = cw[samples_per_pulse*ii:samples_per_pulse*ii+start_off_sample]
            gated_cw[samples_per_pulse*ii+start_off_sample:samples_per_pulse*(ii+1)] = np.zeros((off_samples,))
        return gated_cw


    def generate_pd(self):
        samples_per_pulse = self.calculate_samples_per_pulse()
        nsamps = self.calculate_total_length(samples_per_pulse)
        cw = np.exp(1j*2*np.pi*self.fc_MHz/self.fs_MSPS*np.arange(nsamps))
        gated_cw = self.gate_cw(cw, samples_per_pulse)
        return gated_cw, cw


    def plot_spectrogram(self, array):
        plt.figure()
        f, t, Sxx = signal.spectrogram(array, self.fs_MSPS, nperseg=self.nfft, detrend=False, return_onesided=False)
        min_x = 0
        max_x = len(array)/self.fs_MSPS
        min_y = -self.fs_MSPS/2
        max_y = self.fs_MSPS/2
        extent = [min_x, max_x, min_y, max_y]
        # pcolormesh is extremely slow compared to imshow
        #plt.imshow(fftshift(10*np.log10(np.flip(Sxx, axis=0)), axes=0), aspect='auto', extent=extent)
        plt.imshow(fftshift(10*np.log10(np.flip(Sxx, axis=0)), axes=0), aspect='auto', extent=extent, cmap='ocean')
        #plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='jet')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [usec]')
        plt.show()

    def make_psd(self, array):
        yf = fft(array, self.nfft)
        xf = fftfreq(self.nfft, 1/(self.fs_MSPS))
        return xf, yf

    def plot_psd(self, array):
        xf, yf = self.make_psd(array)
        plt.figure()
        plt.plot(fftshift(xf), 10*np.log10(np.abs(fftshift(yf))**2))
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magsqr ampltiude (dB)')
        plt.show()
        return xf, yf

    def make_detector(self, nlines):
        """Make correlation detector with n lines of detection.
        The lines will be prf_KHz apart from each other"""
        # Get FFT resolution in KHz per bin
        freq_res_kHz_bin = self.fs_MSPS*1e3/self.nfft
        bin_spacing = self.prf_KHz/freq_res_kHz_bin
        detector = np.zeros(np.round(bin_spacing*(nlines-1)+1).astype(int),)
        for ii in np.arange(nlines):
            idx = np.round(ii*bin_spacing).astype(int)
            detector[idx] = 1
        #plt.stem(detector)
        return detector

    def apply_detector(self, gated_cw, detector, title):
        """Apply the detector to the pd spectrum"""
        xf, yf = self.make_psd(gated_cw)
        corr = signal.correlate(np.abs(fftshift(yf))**2, detector, mode='same')
        lags = signal.correlation_lags(len(yf), len(detector), mode='same')

        fig, (ax_orig, ax_det, ax_corr)= plt.subplots(3, 1, figsize=(6, 8))

        ax_orig.plot(fftshift(xf),10*np.log10(np.abs(fftshift(yf))**2))
        ax_orig.set_title('Original spectrum')
        ax_orig.set_xlabel('Frequency (MHz)')

        ax_det.stem(detector)
        ax_det.set_title('generated detector')
        ax_det.set_xlabel('Sample Number')

        ax_corr.plot(fftshift(xf), corr)
        ax_corr.set_title('Cross-correlated signal')
        ax_corr.set_xlabel('Frequency (MHz)')

        fig.suptitle(title)
        ax_orig.margins(0, 0.1)
        ax_det.margins(0, 0.1)
        ax_corr.margins(0, 0.1)
        fig.tight_layout()

if __name__ == "__main__":
    fs_MSPS = 50
    nfft = 32768
    pd = pd_gen(fc_MHz = 2.12254, fs_MSPS = fs_MSPS, dc_percent = 18, prf_KHz = 147, subframe_length_pulses=100, nfft=nfft)
    nl_gated_cw, nl_cw = pd.generate_pd()

    noise_std = 12
    noise = noise_std/np.sqrt(2)*(np.random.randn(nl_gated_cw.size) + 1j*np.random.randn(nl_gated_cw.size))

    gated_cw = nl_gated_cw + noise
    cw = nl_cw + noise

    detector = pd.make_detector(10)
    pd.apply_detector(gated_cw, detector, 'Noisy signal')
    pd.apply_detector(nl_gated_cw, detector, 'Noiseless signal')

    #pd.plot_psd(gated_cw)

    # Plot these together
    scale = 1.5
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(12, 12))
    fig.tight_layout()
    ax1.plot(np.real(nl_gated_cw))
    ax1.plot(np.real(nl_cw), alpha=0.2, linestyle='dashed')
    ax1.axvline(x=nfft, color='r', linestyle='dashed')
    ax1.set_title('Real part, noiseless')
    f1 = zoom_factory(ax1, base_scale=scale)

    ax2.plot(np.imag(nl_gated_cw))
    ax2.plot(np.imag(nl_cw), alpha=0.2, linestyle='dashed')
    ax2.axvline(x=nfft, color='r', linestyle='dashed')
    ax2.set_title('Imag part, noiseless')

    ax3.plot(np.real(gated_cw))
    ax3.plot(np.real(cw), alpha=0.2, linestyle='dashed')
    ax3.axvline(x=nfft, color='r', linestyle='dashed')
    ax3.set_title(f'Real part, noise std = {noise_std}')

    ax4.plot(np.imag(gated_cw))
    ax4.plot(np.imag(cw), alpha=0.2, linestyle='dashed')
    ax4.axvline(x=nfft, color='r', linestyle='dashed')
    ax4.set_title(f'Imag part, noise std = {noise_std}')

    fig.canvas.manager.toolbar.pan()
    plt.show()

