from collections import deque
from datetime import datetime
import logging
import os
import pprint as pp
import math
import numpy as np
import tensorflow as tf
import threading
import queue
import gc
from tf_utils import shape
'''
class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(args=args, kwargs=kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    '''


def thread_decorator(function):
    def wrapper_func(*args, **kwargs):
        thread = threading.Thread(target=function, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper_func


class TFBaseModel(object):
    """Interface containing some boilerplate code for training tensorflow models.
    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.
    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adamw' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_wi11ndow:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
        **additionalargs: u know what they are
    """

    def __init__(
            self,
            reader,
            batch_size=32,
            num_training_steps=20000,
            learning_rate=.01,
            optimizer='adam',
            optimizer_initialized_flag=False,
            grad_clip=5,
            regularization_constant=0.0,
            keep_prob=1.0,
            early_stopping_steps=3000,
            warm_start_init_step=0,
            num_restarts=None,
            enable_parameter_averaging=False,
            use_metric_for_eval=False,
            min_steps_to_checkpoint=100,
            log_interval=20,
            loss_averaging_window=100,
            num_validation_batches=1,
            q_size=10,
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            step_at_initialization=0,
            train_q=queue.Queue(maxsize=10),
            val_q=queue.Queue(maxsize=10),
            test_q=queue.Queue(maxsize=10),
            timeout=10,
            restore_saver_flag=True,
            pretrained_dir='/hdd-1/pretrained_models/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
            use_adaptive_lr=False,
            lr_reduction=10,
            num_cycles=1,

    ):

        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_initialized_flag = optimizer_initialized_flag
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.early_stopping_steps = early_stopping_steps
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.num_restarts = num_restarts
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.num_validation_batches = num_validation_batches
        self.loss_averaging_window = loss_averaging_window
        self.q_size = q_size
        self.log_dir = log_dir
        self.step_at_initialization = step_at_initialization
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        self.train_q = train_q
        self.val_q = val_q
        self.test_q = test_q
        self.timeout = timeout
        self.use_adaptive_lr = use_adaptive_lr
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'
        self.use_metric_for_eval = use_metric_for_eval
        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))
        self.is_train = True
        self.restore_saver_flag=restore_saver_flag
        self.graph = self.build_graph()
        self.session = tf.InteractiveSession(graph=self.graph)
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.pretrained_dir = pretrained_dir
        self.lr_reduction=lr_reduction
        self.num_cycles=num_cycles
        print('built graph')

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    @thread_decorator
    def test_data_loader(self):
        for i, test_batch_df in enumerate(self.test_generator):
            test_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in test_batch_df if hasattr(self, placeholder_name)
            }
            if hasattr(self, 'keep_prob'):
                test_feed_dict.update({self.keep_prob: 1.0})
            if hasattr(self, 'is_training'):
                test_feed_dict.update({self.is_training: False})
            self.test_q.put(test_feed_dict, True, self.timeout)

    @thread_decorator
    def val_data_loader(self):
        still_training = True
        while still_training == True:
            val_batch_df = next(self.val_generator)
            val_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in val_batch_df if hasattr(self, placeholder_name)
            }
            # print(val_feed_dict)

            val_feed_dict.update({self.learning_rate_var: self.learning_rate})
            if hasattr(self, 'keep_prob'):
                val_feed_dict.update({self.keep_prob: 1.0})
            if hasattr(self, 'is_training'):
                val_feed_dict.update({self.is_training: False})
            try:
                self.val_q.put(val_feed_dict, True, self.timeout)
            except:
                still_training = False

    @thread_decorator
    def train_data_loader(self):
        still_training = True
        while still_training == True:
            train_batch_df = next(self.train_generator)
            train_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)
            }
            # print(train_feed_dict)

            train_feed_dict.update({self.learning_rate_var: self.learning_rate})
            if hasattr(self, 'keep_prob'):
                train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
            if hasattr(self, 'is_training'):
                train_feed_dict.update({self.is_training: True})
            try:
                self.train_q.put(train_feed_dict, True, self.timeout)
            except:
                still_training = False

    def fit(self, restore_pretrain=True, restore_dir=None):

        self.train_generator = self.reader.train_batch_generator(self.batch_size)
        self.val_generator = self.reader.val_batch_generator(self.batch_size)

        with self.session.as_default():
            t1 = self.train_data_loader()
            t2 = self.val_data_loader()
            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                if restore_dir is not None:
                    self.restore(restore_dir=restore_dir)
                    step = 0
                else:
                    self.session.run(self.init)
                    step = 0
                    
                    # time.sleep(1)
                    # print(self.restore_list)
                    if restore_pretrain:
                        print('restoring')
                        self.restore_backbone(self.pretrained_dir, self.backbone_scope)

            if self.use_adaptive_lr:
                self.base_learning_rate = self.learning_rate

            self.step_at_initialization = step
            step = self.step_at_initialization
            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)
            train_metric_history = deque(maxlen=self.loss_averaging_window)
            val_metric_history = deque(maxlen=self.loss_averaging_window)
            best_validation_loss, best_validation_tstep = float('inf'), 0
            restarts = 0
            print('training')
            # sys.stdout.flush()
            while step < self.num_training_steps:

                # validation evaluation
                if (step % self.num_validation_batches == 0):
                    for validation_batch_num in range(self.num_validation_batches):
                        self.is_train = False

                        # val_batch_df = self.val_q.get(True, self.timeout)
                        val_feed_dict_loop = self.val_q.get(True, self.timeout)

                        val_loss, val_metric = self.session.run(
                            fetches=[self.loss, self.metric],
                            feed_dict=val_feed_dict_loop
                        )
                        val_loss_history.append(val_loss)
                        val_metric_history.append(val_metric)
                        if hasattr(self, 'monitor_tensors'):
                            for name, tensor in self.monitor_tensors.items():
                                [np_val] = self.session.run([tensor], feed_dict=val_feed_dict_loop)
                                print(name)
                                # print('min', np_val.min())
                                # print('max', np_val.max())
                                print('mean', np_val.mean())
                                # print('std', np_val.std())
                                print('nans', np.isnan(np_val).sum())
                                print('shape', np_val.shape)

                # train step
                self.is_train = True
                train_feed_dict_loop = self.train_q.get(True, self.timeout)

                if self.use_adaptive_lr:
                    alpha=.05
                    global_step = step
                    anneal_steps = self.num_training_steps/self.num_cycles
                    if (step % anneal_steps) == 0:
                        self.learning_rate=self.base_learning_rate
                    cosine_decay = 0.5 * (1 + np.cos(math.pi * (global_step % anneal_steps) / anneal_steps))
                    decayed = (1 - alpha) * cosine_decay + alpha
                    self.learning_rate = self.base_learning_rate * decayed

                train_loss, _, train_metric = self.session.run(
                    fetches=[self.loss, self.step, self.metric],
                    feed_dict=train_feed_dict_loop
                )

                train_loss_history.append(train_loss)
                train_metric_history.append(train_metric)

                if step % self.log_interval == 0:
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    avg_train_metric = sum(train_metric_history) / len(train_metric_history)
                    avg_val_metric = sum(val_metric_history) / len(val_metric_history)
                    metric_log = (
                        "[[step {:>8}]]     "
                        "[[train]]     loss: {:<12}     metric: {:<12}"
                        "[[val]]     loss: {:<12}      metric: {:<12}     learning_rate: {:<12}"
                    ).format(step, round(avg_train_loss, 8), round(avg_train_metric, 8), round(avg_val_loss, 8),
                             round(avg_val_metric, 8), round(self.learning_rate, 8))
                    logging.info(metric_log)
                    if self.use_metric_for_eval:
                        update_condition = avg_val_metric < best_validation_loss
                    else:
                        update_condition = avg_val_loss < best_validation_loss

                    if update_condition:
                        if self.use_metric_for_eval:
                            best_validation_loss = avg_val_metric
                        else:
                            best_validation_loss = avg_val_loss
                        best_validation_tstep = step
                        if step <= self.min_steps_to_checkpoint:
                            best_validation_loss = float('inf')
                        if step > self.min_steps_to_checkpoint:
                            self.save(step)
                            if self.enable_parameter_averaging:
                                self.save(step, averaged=True)

                    if step - best_validation_tstep > self.early_stopping_steps:
                        #self.optimizer_initialized_flag = False
                        if self.num_restarts is None or restarts >= self.num_restarts:
                            logging.info('best validation loss of {} at training step {}'.format(
                                best_validation_loss, best_validation_tstep))
                            logging.info('early stopping - ending training.')
                            return

                        if restarts < self.num_restarts:
                            self.restore(best_validation_tstep)
                            logging.info(
                                '1/2 learning rate-restart {}-learning rate {}'.format(restarts, self.learning_rate))
                            self.learning_rate /= self.lr_reduction
                            step = best_validation_tstep
                            #self.optimizer_initialized_flag = False
                            restarts += 1

                step += 1

            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            logging.info('num_training_steps reached - ending training')

            t1.join()
            t2.join()

    def predict(self, chunk_size=512, is_val=False, append=None, test_aug=False):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)
        self.is_train = False
        print('predicting')
        if hasattr(self, 'prediction_tensors'):
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            self.test_generator = self.reader.test_batch_generator(chunk_size, test_aug=test_aug)
            if is_val:
                self.test_generator = self.reader.val_batch_generator(chunk_size, num_epochs=1, shuffle=False, test_aug=test_aug)
            self.done_testing = False
            t1 = self.test_data_loader()
            while self.done_testing == False:
                try:
                    test_feed_dict = self.test_q.get(True, self.timeout)
                except:
                    self.done_testing = True
                if self.done_testing == False:

                    tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                    np_tensors = self.session.run(
                        fetches=tf_tensors,
                        feed_dict=test_feed_dict
                    )
                    for tensor_name, tensor in zip(tensor_names, np_tensors):
                        prediction_dict[tensor_name].append(tensor)

            for tensor_name, tensor in prediction_dict.items():
                np_tensor = np.concatenate(tensor, 0)
                if append is not None and is_val:
                    join_name = tensor_name + 'val'+append
                elif append:
                    join_name = tensor_name + append
                elif is_val:
                    join_name = tensor_name + 'val'
                else:
                    join_name = tensor_name
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(join_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        t1.join()
        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                np_tensor = tensor.eval(self.session)

                sarestore_saverve_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

    def save(self, step, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}'.format(model_path))
        saver.save(self.session, model_path, global_step=step)

    def export_meta_graph(self, step, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')

        saver.export_meta_graph(model_path + '/meta_graph-{}.meta'.format(step))

    def restore(self, step=None, averaged=False, restore_dir=None):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if restore_dir is not None:
            checkpoint_dir = restore_dir
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            logging.info('restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)

    def restore_backbone(self, model_path, backbone_scope):

        self.restore_saver.restore(self.session, model_path)

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        # reload(logging)  # bad
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def update_parameters(self, loss):
        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([(tf.reduce_sum(tf.nn.l2_loss(param))) \
                                                    for param in tf.trainable_variables()])
            loss2 = loss + self.regularization_constant * l2_norm

        if not self.optimizer_initialized_flag:
            optimizer = self.get_optimizer(self.learning_rate_var)
        else:
            optimizer = self.optimizer
        if self.regularization_constant != 0:
            grads = optimizer.compute_gradients(loss2)
        else:
            grads = optimizer.compute_gradients(loss)
        if self.grad_clip == -1:
            clipped=grads
        else:
            clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) if g is not None else (g, v_)
                       for g, v_ in grads]
        step = optimizer.apply_gradients(clipped, global_step=self.global_step)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        # logging.info('all parameters:')
        # logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        # logging.info('trainable parameters:')
        # logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')
        logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adamw':
            return tf.contrib.opt.AdamWOptimizer(weight_decay=.01*self.learning_rate_var, learning_rate=learning_rate)
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate_var = tf.Variable(0.0, trainable=False)

            self.loss = self.calculate_loss()
            self.restore_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.backbone_scope)
            if self.restore_saver_flag==True:
                self.restore_saver = tf.train.Saver(self.restore_list)

            self.extra_update_op_c = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.extra_update_op_c):
                self.update_parameters(self.loss)

            self.saver = tf.train.Saver(max_to_keep=1)
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()

            return graph

    def close_session(self):
        self.session.close()
        del self.session
        tf.reset_default_graph()
        gc.collect()