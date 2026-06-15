import numpy as np
from numpy import fft

def compute(frames):
    """
    Compute the bidirectional phase offset between odd and even scan lines

    Estimates pixel offset between alternating lines that occur in bidirectional line
    scanning, using phase correlation along x axis

    PARAMETERS
        frames: np.ndarray
            Random subsample of frames of shape (n_frames, Ly, Lx)

    OUTPUT
        bidiphase: int
            Bidirectional phase offest in pixels
    """
    _, Ly, Lx = frames.shape

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, 1::2, :], axis=2)
    d1 /= np.abs(d1) + 1e-5

    d2 = np.conj(fft.fft(frames[:, ::2, :], axis=2))
    d2 /= np.abs(d2) + 1e-5
    d2 = d2[:, :d1.shape[1], :]

    cc = np.real(fft.ifft(d1 * d2, axis=2))
    cc = cc.mean(axis=1).mean(axis=0)
    cc = fft.fftshift(cc)

    bidiphase = -(np.argmax(cc[-10 + Lx // 2:11 + Lx // 2]) - 10)
    return bidiphase

def shift(frames, bidiphase):
    """
    Shift odd scan lines by the bidirectional phase offset
    """
    if bidiphase > 0:
        frames[:, 1::2, bidiphase:] = frames[:, 1::2, :-bidiphase]
    elif bidiphase < 0:
        frames[:, 1::2, :bidiphase] = frames[:, 1::2, -bidiphase:]
    return frames