import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.signal import resample, savgol_filter


class PPG_test(Dataset):
    def __init__(self, txt, filetype='csv', siglength=30):
        self.filetype = filetype

        if self.filetype == 'npy':
            self.data = np.load(txt).astype(np.float32)
        else:
            self.data = open(txt, 'r').read().strip().split('\n')

        self.siglength = siglength

    def expand10to30(self, sig):
        res = np.tile(sig, 3)
        smoother = savgol_filter(res, 101, 7)
        res = resample(res, 7201)

        smoother = resample(smoother, 7201)

        N = int(7201 / 3)
        for i in range(2):
            sf = N * (i + 1) - 50
            ef = N * (i + 1) + 50

            res[sf:ef] = smoother[sf:ef]

        return res

    def expand25to30(self, sig):

        N = int(len(sig) * 0.2)
        res = np.concatenate((sig, sig[:N]), axis=None)

        smoother = savgol_filter(res, 101, 7)
        res = resample(res, 7201)
        smoother = resample(smoother, 7201)

        N = int(7201 * 0.8)
        sf = N - 50
        ef = N + 50

        res[sf:ef] = smoother[sf:ef]

        return res

    def normalize(self, sig):
        smin = np.min(sig)
        smax = np.max(sig)
        return (sig - smin) / (smax - smin)

    def __getitem__(self, index):
        fn = self.data[index]
        data = []

        if self.filetype == 'csv':
            f = open(fn).read().split(',')
            for item in f:
                data.append(float(item))
            data = np.array(data).astype(np.float32)
        elif self.filetype == 'npy':
            data = self.data[index]
            if self.siglength == 25:
                data = self.expand25to30(data)
            elif len(data) != 7201:
                data = resample(data, 7201)
        else:
            mat = sio.loadmat(fn)
            data = mat['val'][1].astype(np.float32)
            if self.siglength == 10:
                data = self.expand10to30(data)
                data = self.normalize(data)

        return data.reshape(1, len(data))

    def __len__(self):
        return len(self.data)


class PPG_np_test(Dataset):
    def __init__(self, fn, filetype):
        self.filetype = filetype
        self.data = np.load(fn) if filetype == 'npy' else np.load(fn, allow_pickle = True)

    def expand25to30(self, sig):

        N = int(len(sig) * 0.2)
        res = np.concatenate((sig, sig[:N]), axis=None)

        smoother = savgol_filter(res, 101, 7)
        res = resample(res, 7201)
        smoother = resample(smoother, 7201)

        N = int(7201 * 0.8)
        sf = N - 50
        ef = N + 50

        res[sf:ef] = smoother[sf:ef]

        return res

    def normalize(self, sig):
        smin = np.min(sig)
        smax = np.max(sig)
        return (sig - smin) / (smax - smin)

    def __getitem__(self, index):

        if self.filetype == 'npy':
            data = self.data[index]

        else:
            data = self.data['signal'][index]
            data = self.expand25to30(data)

        data = data.astype(np.float32)

        return data.reshape(1, 7201)

    def __len__(self):
        return len(self.data)
