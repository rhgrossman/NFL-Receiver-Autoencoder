import os

import numpy as np
from data_frame import DataFrame
import threading

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
GLOBAL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def spawn(f):
    def fun(pipe, x):
        pipe.send(f(x))
        pipe.close()

    return fun


def parmap(f, X):
    pipe = [Pipe() for x in X]
    proc = [Process(target=spawn(f), args=(c, x)) for x, (p, c) in zip(X, pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p, c) in pipe]


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
      '"""
    if len(image.shape) != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = np.split(image, 3, axis=-1)
    for i in range(num_channels):
        channels[i] -= means[i]
    return np.concatenate(channels, axis=2)


def thread_decorator(function):
    def wrapper_func(*args, **kwargs):
        thread = threading.Thread(target=function, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper_func


class DataReader(object):

    def __init__(self, data_dir, train_data_dir, test_data_dir, val_size, num_classes,
                 train_inx, val_inx, seed, percent_augmented=.98, band=0):

        data_cols = ['all_ids',
                     'all_band',
                     'all_mjd_diff',
                     'all_flux',
                     ]
        train_data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)))[train_inx] for i in data_cols]
        val_data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)))[val_inx] for i in data_cols]

        self.train_df = DataFrame(columns=data_cols, data=train_data)
        self.val_df = DataFrame(columns=data_cols, data=val_data)
        self.test_df=self.train_df
        # seed=np.random.randint(0, 1000000)
        # seed = 99 + seed
        # self.train_df, self.val_df = self.full_train.train_test_split(train_size=0.95, random_state=seed)
        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.percent_augmented = percent_augmented
        self.val_size=val_size
        self.num_classes=num_classes
        self.max_frames = 72
        self.GLOBAL_IS_VAL = False
        self.GLOBAL_IS_TEST = False
        self.band=band

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size, shuffle=True, num_epochs=10000, test_aug=False):
        is_test = False
        if num_epochs == 1:
            is_test = True
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=num_epochs,
            is_val=True,
            is_test=is_test,
            test_aug=test_aug,
        )

    def test_batch_generator(self, batch_size, test_aug):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            is_test=True,
            test_aug=test_aug,
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False, is_val=False,
                        test_aug=False):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )

        self.GLOBAL_IS_VAL = is_val
        self.GLOBAL_IS_TEST = is_test
        band=self.band
        print('batch_size', batch_size)
        print(is_test)

        id_col= 'all_ids'
        band_col = 'all_band'
        mjd_col = 'all_mjd_diff'
        flux_col = 'all_flux'

        #print(id_col)
        for batch in batch_gen:
            batch_size = len(batch[id_col])

            flux = np.zeros([batch_size, 200])
            mjd = np.zeros([batch_size, 200])
            nan = np.zeros([batch_size, 200])
            band = np.zeros([batch_size, 200])
            ids=np.zeros([batch_size]).astype(str)

            for cur_inx, (id_batch, flux_batch, mjd_batch, band_batch) in enumerate(zip(batch[id_col],
                batch[flux_col], batch[mjd_col], batch[band_col])):


                flux_batch=np.log1p(np.abs(np.nan_to_num(flux_batch)))*np.sign(flux_batch)
                flux_batch=flux_batch[::-1]
                mjd_batch=np.nan_to_num(mjd_batch[::-1])
                band_batch=np.nan_to_num(band_batch[::-1])
                nan_batch=(flux_batch!=0).astype(int)
                flux[cur_inx, :]=flux_batch[-200:]
                mjd[cur_inx, :] = mjd_batch[-200:]
                nan[cur_inx, :] = nan_batch[-200:]
                band[cur_inx, :] = band_batch[-200:]
                ids[cur_inx]=str(id_batch)

            np.nan_to_num(flux, copy=False)
            np.nan_to_num(mjd, copy=False)
            #print(np.isnan(flux).sum(), np.isnan(mjd).sum(), np.isnan(nan).sum())
            batch['ids'] = ids
            batch['values'] =  flux
            batch['differences'] = mjd
            batch['is_nan']=nan
            batch['band']= band

            #print(flux[0], mjd[0], nan[0], band[0])
            #print(batch['seqlengths'])
            #print(seq_lengths.shape)
            #print(flux)
            yield batch
