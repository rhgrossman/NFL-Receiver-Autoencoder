import os
import sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cuda_device = '0'  # see issue #
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
        self.ids=tf.placeholder(tf.string, [None, 3])
        self.values = tf.placeholder(tf.float32, [None, self.feature_size, 4])
        self.target = tf.placeholder(tf.int32, [None, self.feature_size])
        self.targets =tf.one_hot(self.target, 10)
        self.is_training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])



    def rnn_encoder(self):

        
        inputs=tf.concat([self.values, self.targets], axis=-1)           
        
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
        features=state_adder

        with tf.variable_scope('decoder'):
            fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hidden_dim)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, features, dtype=tf.float32
            ) # Set `time_major` accordingly

        intermediate = tf.concat([outputs[0], outputs[1]], axis=-1)
        self.classification_outputs=tf.layers.conv1d(intermediate, 10, 1)
        self.regression_outputs=tf.layers.conv1d(intermediate, 2, 1)
    #def dense(self):
    #    self.classification_outputs=tf.layers.dense(self.log_x_encode, 90, activation='linear')

    def calculate_loss(self):
        self.get_input_tensors()
        #self.dense()
        self.rnn_encoder()
        self.rnn_decoder()
        weights=tf.ones_like(self.classification_outputs)
        loss=tf.reduce_sum(
              tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.classification_outputs)*weights)/tf.reduce_sum(weights)
        
        loss2=tf.reduce_mean(
            tf.abs(tf.squeeze(self.regression_outputs) - self.values[:, :, 1:3]))
        self.loss=loss+.1*loss2
                
        self.metric=self.loss
        self.prediction_tensors = {
            'state': self.state,
            'ids': self.ids,

        }
        return self.loss



if __name__ == '__main__':
    base_dir = './'
    run_no="v6"
    run_no = 'direction_targets_{}'.format(run_no)
    n_splits = 4
    fold_no=0
    checkpoint_dir = 'checkpoints/checkpoints_run_{}_fold_{}'.format(run_no, fold_no)
    prediction_dir = 'predictions/predictions_run_{}_fold_{}'.format(run_no, fold_no)


    batch_size = 128
    dr = DataReader(data_dir='./data/processed',

                    seed=fold_no)
    # with tf.device('/device:GPU:1'):
    nn = rnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, checkpoint_dir),
        prediction_dir=os.path.join(base_dir, prediction_dir),
        optimizer='adamw',
        learning_rate=.001,
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
        num_classes=10,
        feature_size=34,

    )

    rest_dir_name = 'name_to_restore'
    rest_dir = './checkpoints/{}/'.format(rest_dir_name)
    rest_dir = None

    nn.fit(restore_pretrain=False, restore_dir=None)
    nn.restore(averaged=False)
    nn.predict(chunk_size=128, is_val=True)
    nn.predict(chunk_size=128)
    #nn.restore(averaged=False)

    #nn.predict(chunk_size=4, append=str(test_chunk))
    #if test_chunk<1:
    #    nn.predict(chunk_size=4, is_val=True)

    # for i in range(8):
    #    nn.predict(chunk_size=16, append='_{}'.format(i), test_aug=i)
    #    nn.predict(chunk_size=16, is_val=True, append='_{}'.format(i), test_aug=i)
    # print(cuda_device)

    nn.close_session()
    



