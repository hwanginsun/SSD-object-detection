"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K


def SSDLoss(alpha=1., pos_neg_ratio=3., verbose=True):
    """
    Original Loss Function Using Hard Negative Sampling

    :param alpha:
    :param pos_neg_ratio:
    :return:
    """
    def ssd_loss(y_true, y_pred):
        num_classes = tf.shape(y_true)[2] - 4
        y_true = tf.reshape(y_true, [-1, num_classes + 4])
        y_pred = tf.reshape(y_pred, [-1, num_classes + 4])
        eps = K.epsilon()

        # Split Classification and Localization output
        y_true_clf, y_true_loc = tf.split(y_true, [num_classes, 4], axis=-1)
        y_pred_clf, y_pred_loc = tf.split(y_pred, [num_classes, 4], axis=-1)

        # split foreground & background
        neg_mask = y_true_clf[:, -1]
        pos_mask = 1 - neg_mask
        num_pos = tf.reduce_sum(pos_mask)
        num_neg = tf.reduce_sum(neg_mask)
        num_neg = tf.minimum(pos_neg_ratio * num_pos, num_neg)

        # softmax loss
        y_pred_clf = K.clip(y_pred_clf, eps, 1. - eps)
        clf_loss = -tf.reduce_sum(y_true_clf * tf.log(y_pred_clf),
                                  axis=-1)
        pos_clf_loss = tf.reduce_sum(clf_loss * pos_mask) / (num_pos + eps)
        neg_clf_loss = clf_loss * neg_mask
        values, indices = tf.nn.top_k(neg_clf_loss,
                                      k=tf.cast(num_neg, tf.int32))
        neg_clf_loss = tf.reduce_sum(values) / (num_neg + eps)

        clf_loss = pos_clf_loss + neg_clf_loss
        # smooth l1 loss
        l1_loss = tf.abs(y_true_loc - y_pred_loc)
        l2_loss = 0.5 * (y_true_loc - y_pred_loc) ** 2
        loc_loss = tf.where(tf.less(l1_loss, 1.0),
                            l2_loss,
                            l1_loss - 0.5)
        loc_loss = tf.reduce_sum(loc_loss, axis=-1)
        loc_loss = tf.reduce_sum(loc_loss * pos_mask) / (num_pos + eps)

        # total loss
        if verbose:
            clf_loss = K.print_tensor(clf_loss, message='clf_loss : ')
            loc_loss = K.print_tensor(loc_loss, message='loc_loss : ')
        return clf_loss + alpha * loc_loss
    return ssd_loss
