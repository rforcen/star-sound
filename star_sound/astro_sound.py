'''
generate sound wav files from
http://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/lib.html
.dat star spectrum files
'''

import re
from math import pi, sin

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.io.wavfile import write as wav_write

from musicFreq import MusicFreq
from peaks import peakdetect


# fits
def test_fits():
    hdulist = fits.open('stars/uko5v.fits')

    print(hdulist.info())

    hdu = hdulist[0]

    print(hdu.header)
    print(hdu.data.shape)
    print(hdu.data)

    plt.plot(hdu.data)
    plt.show()


def _read_dat(fh):  # read from fh
    irms, lrms = 0, 0  # isotope-ratio mass spectrometer (IRMS)
    header, data = [], []

    for ln, line in enumerate(fh):
        if line.startswith('#'):
            if ln == 0:
                l = line[1:].strip().split(' ')
                for _l in l:
                    # regex w/2 groups 1=[il]RMS, 2=value
                    s_il_rms = re.search('([il]RMS)=([-+]?(([0-9]*[.]?[0-9]+([ed][-+]?[0-9]+)?)))', _l)
                    if s_il_rms:
                        if s_il_rms.group(1) == 'iRMS':
                            irms = float(s_il_rms.group(2))
                        elif s_il_rms.group(1) == 'lRMS':
                            lrms = float(s_il_rms.group(2))

            elif ln == 1:
                header += list(filter(None, line[1:].strip().split(' ')))
        else:
            lst = filter(None, line.strip().split(' '))
            data += [list(map(float, lst))]
    return irms, lrms, header, np.asarray(data)


def read_dat(fnme):  # usage: irms, lrms, header, data = read_dat('stars/uko5v.dat')
    with open(fnme, 'r') as fh:
        return _read_dat(fh)


def star_sound(fnme, wavenme=None, rate=44100, secs=5):  # from data[1]
    def sound_func(amp, hz, t):  # sound generation
        return amp * sin(hz * t) * sin(MusicFreq.freq2octave(hz, -7) * t)

    def mean(l):
        return sum(l) / len(l)

    irms, lrms, header, data = read_dat(fnme)

    y = data.T[1]
    x = data.T[0]
    dif = (y.max() - y.min())

    pk = peakdetect(y, x, lookahead=1, delta=dif / 10)
    max_peaks = pk[0]  # [ [x0,y0] , ...., [xn,yn] ]

    waves = [[(_y - y.min()) / dif, MusicFreq.freq2octave(_x, 0)] for _x, _y in max_peaks]  # amp(0..1), freq oct '0'
    waves.sort(reverse=True)  # get <= 10 most powerful
    waves = waves[:10]
    pi2 = pi * 2  # -> evaluate waves average for each sample
    datawav = np.asarray(
        [mean([sound_func(amp, hz, t) for amp, hz in waves]) for t in np.arange(0, secs * pi2, pi2 / rate)],
        dtype=np.float32)

    if wavenme is None or not wavenme:
        wavenme = fnme.replace('.dat', '.wav')

    wav_write(wavenme, rate, datawav)


def test_read_dat():
    irms, lrms, header, data = read_dat('stars/ukb0i.dat')

    print('iRMS=', irms, 'lRMS=', lrms)
    for h in header:
        print(h, end='\t')
    print()

    for d in data:
        for c in d:
            print(c, end='\t')
        print()


star_sound('stars/ukb0i.dat')
