import numpy as np
from torch.utils.data import Dataset


class PPG(Dataset):
    def __init__(self, txt):
        fns = open(txt, 'r')
        data = []
        for line in fns:
            fn, label = line.strip().split()
            data.append((fn, label))
        fns.close()
        self.data = data

    def __getitem__(self, index):
        fn, labelind = self.data[index]
        data = []

        label = np.zeros(2)
        label[int(labelind)] = 1
        label = label.astype(np.float32)

        f = open(fn).read().split(',')
        for item in f:
            data.append(float(item))

        data = np.array(data).astype(np.float32).reshape(1, len(data))

        return data, label

    def __len__(self):
        return len(self.data)
