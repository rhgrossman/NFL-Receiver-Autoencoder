import os
import sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cuda_device = str(sys.argv[2])  # see issue #
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

import numpy as np
import tensorflow as tf

from tf_base_model import TFBaseModel
from sklearn.model_selection import KFold
from data_iterator_autoencoder_v2 import DataReader
from tf_utils import shape, sequence_mean

slim = tf.contrib.slim
tf.reset_default_graph()
class rnn(TFBaseModel):

    def __init__(
            self,
            backbone_scope='resnet_v2_50',
            target_shape=[513, 513, 3],
            num_anchors=1161,
            anchors=None,
            hidden_dim=128,
            initialized=False,
            feature_size=245,
            num_classes=37,
            **kwargs
    ):
        self.backbone_scope = backbone_scope
        self.target_shape = target_shape
        self.block_outputs = None
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.initialized = initialized
        self.hidden_dim=hidden_dim
        self.feature_size=feature_size
        self.num_classes=num_classes

        super(rnn, self).__init__(**kwargs)

    def transform(self, x):
        return tf.log(x + 1) - tf.expand_dims(self.log_x_encode_mean, 1)

    def inverse_transform(self, x):
        return tf.exp(x + tf.expand_dims(self.log_x_encode_mean, 1)) - 1

    def get_input_tensors(self):
        self.ids=tf.placeholder(tf.string, [None])

        self.values = tf.placeholder(tf.float32, [None, self.feature_size])
        self.differences = tf.placeholder(tf.float32, [None, self.feature_size])
        self.is_nan = tf.placeholder(tf.float32, [None, self.feature_size])
        self.band=tf.placeholder(tf.int32, [None, self.feature_size])

        self.is_training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])



    def rnn_encoder(self):
        #print(self.log_x_encode.shape().as_list())

        inputs=tf.concat([tf.expand_dims(self.values, axis=-1),
                          tf.expand_dims(self.differences, axis=-1),
                          tf.one_hot(self.band, 6)], axis=-1)
        with tf.variable_scope('forward'):
            fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs, dtype=tf.float32
            ) # Set `time_major` accordingly


        intermediate=tf.concat([outputs[0], outputs[1]], axis=-1)
        with tf.variable_scope('encoder'):
            fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(12)
            bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(12)
            #lstm_cell_fwd = tf.contrib.rnn.DropoutWrapper(lstm_cell_fwd, output_keep_prob=self.keep_prob)
            outputs, self.state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, intermediate, dtype=tf.float32)
        self.state=self.state[0]
        #intermediate = tf.concat([outputs[0], outputs[1]], axis=0)
        #self.state=outputs[:,-1:,:]

    def rnn_decoder(self):
        state_adder =tf.tile(tf.expand_dims(self.state, axis=1), (1, self.feature_size, 1))
        #state_adder=tf.cast(state_adder, tf.float32)
        features=tf.concat([tf.expand_dims(self.differences, axis=-1),
                            tf.one_hot(self.band, 6),
                            state_adder], axis=-1)

        with tf.variable_scope('decoder'):
            fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, features, dtype=tf.float32
            ) # Set `time_major` accordingly

        intermediate = tf.concat([outputs[0], outputs[1]], axis=-1)
        self.classification_outputs=tf.contrib.layers.conv1d(intermediate, 1, 1, activation_fn=None)
    #def dense(self):
    #    self.classification_outputs=tf.layers.dense(self.log_x_encode, 90, activation='linear')

    def calculate_loss(self):
        self.get_input_tensors()
        #self.dense()
        self.rnn_encoder()
        self.rnn_decoder()
        self.loss=tf.reduce_mean(tf.multiply(
            tf.math.abs(tf.squeeze(self.classification_outputs) - self.values), self.is_nan))

        self.metric=self.loss
        self.prediction_tensors = {
            'state': self.state,
            'ids': self.ids,

        }
        return self.loss



if __name__ == '__main__':
    base_dir = './'
    run_no="v2"
    run_no = 'ae_allflux_fixed_16_{}'.format(run_no)
    n_splits = 4

    kf=KFold(n_splits=5, shuffle=True, random_state=117)
    band=int(sys.argv[3])
    train_ids=np.load('./data/processed3/all_ids.npy')
    print(train_ids.shape)
    fold_no=0
    fold_run=int(sys.argv[1])

    for train_inx, val_inx in kf.split(train_ids):
        fold_no+=1
        if fold_no!=fold_run:
            continue
        print('device:', cuda_device)
        checkpoint_dir = 'checkpoints/checkpoints_run_{}_fold_{}'.format(run_no, fold_no)
        prediction_dir = 'predictions/predictions_run_{}_fold_{}'.format(run_no, fold_no)


        batch_size = 256
        dr = DataReader(data_dir='./data/processed3', train_data_dir='./data/train/',
                        test_data_dir='./data/test/', val_size=96,
                        num_classes=14, percent_augmented=1., train_inx=train_inx,
                        val_inx=val_inx, seed=fold_no,  band=band)
        # with tf.device('/device:GPU:1'):
        nn = rnn(
            reader=dr,
            log_dir=os.path.join(base_dir, 'logs'),
            checkpoint_dir=os.path.join(base_dir, checkpoint_dir),
            prediction_dir=os.path.join(base_dir, prediction_dir),
            optimizer='adamw',
            learning_rate=.01,
            batch_size=batch_size,
            num_training_steps=1200000,
            num_cycles=2,
            early_stopping_steps= 3000,
            warm_start_init_step=0,
            regularization_constant=0,  # .000002,
            keep_prob=1,
            enable_parameter_averaging=True,
            num_restarts=2,
            min_steps_to_checkpoint=4000,
            log_interval=10,
            num_validation_batches=1,
            grad_clip=20,
            loss_averaging_window=int(1000),
            backbone_scope='resnet_v2_50',  # 'InceptionResnetV2', #
            initialized=False,
            use_metric_for_eval= False,
            use_adaptive_lr=False,
            hidden_dim=64,
            restore_saver_flag=False,
            num_classes=14,
            feature_size=200,

        )

        rest_dir_name = 'name_to_restore'
        rest_dir = './checkpoints/{}/'.format(rest_dir_name)
        rest_dir = None

        nn.fit(restore_pretrain=False, restore_dir=rest_dir)
        nn.restore(averaged=False)
        nn.predict(chunk_size=4, is_val=True)
        nn.predict(chunk_size=4)
        #nn.restore(averaged=False)

        #nn.predict(chunk_size=4, append=str(test_chunk))
        #if test_chunk<1:
        #    nn.predict(chunk_size=4, is_val=True)

        # for i in range(8):
        #    nn.predict(chunk_size=16, append='_{}'.format(i), test_aug=i)
        #    nn.predict(chunk_size=16, is_val=True, append='_{}'.format(i), test_aug=i)
        # print(cuda_device)

        nn.close_session()
        break



