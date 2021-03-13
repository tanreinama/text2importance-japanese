import argparse
import json
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
import matplotlib.pyplot as plt

from model import modeling
from model.modeling import BertConfig, BertModel

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='text2importance-ja_small', help='pretrained model directory.')
parser.add_argument('--context', type=str, default='', help='input text.')
parser.add_argument('--input_file', type=str, default='', help='input text file.')
parser.add_argument('--output_file', type=str, default='', help='result csv file to write.')
parser.add_argument('--gpu', default='0', help='visible gpu number.')

sep_txt = lambda text: text.replace('\n', '。').replace('．', '。').replace('｡', '。').split('。')

def get_masked_regression_output(bert_config, input_tensor, positions,
                         label_values, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/regression"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        output_weights = tf.get_variable(
                "output_weights",
                shape=[1, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
                "output_bias",
                shape=[1],
                initializer=tf.zeros_initializer())
        outputs = tf.matmul(input_tensor, output_weights, transpose_b=True)
        outputs = tf.nn.bias_add(outputs, output_bias)

        output_values = tf.reshape(outputs, [-1])
        label_values = tf.reshape(label_values, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        per_example_loss = (output_values - label_values) ** 2
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, outputs)


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                                                        [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def main():
    args = parser.parse_args()

    if os.path.isfile(args.model+'/hparams.json'):
        with open(args.model+'/hparams.json') as f:
            bert_config_params = json.load(f)
    else:
        raise ValueError('invalid model name.')

    if not (len(args.input_file) > 0 or len(args.context) > 0):
        raise ValueError('--input_file or --context required.')
    if (not os.path.isfile(args.input_file)) and len(args.context) ==0:
        raise ValueError('invalid input file name.')
    if len(args.input_file) > 0 and os.path.isfile(args.input_file):
        with open(args.input_file) as f:
            args.context = f.read()

    vocab_size = bert_config_params['vocab_size']
    max_seq_length = bert_config_params['max_position_embeddings']
    batch_size = 1
    EOT_TOKEN = vocab_size - 4
    MASK_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 1

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')

    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    bert_config = BertConfig(**bert_config_params)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu

    with tf.Session(config=config) as sess:
        input_ids = tf.placeholder(tf.int32, [None, None])
        input_mask = tf.placeholder(tf.int32, [None, None])
        segment_ids = tf.placeholder(tf.int32, [None, None])
        masked_lm_positions = tf.placeholder(tf.int32, [None, None])
        masked_lm_ids = tf.placeholder(tf.int32, [None, None])
        masked_lm_weights = tf.placeholder(tf.float32, [None, None])
        next_sentence_labels = tf.placeholder(tf.int32, [None])

        model = BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output = model.get_sequence_output()
        (_,_,_) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (_,_,_) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        saver = tf.train.Saver()

        masked_lm_values = tf.placeholder(tf.float32, [None, None])

        with tf.variable_scope("loss"):
            (_,outputs) = get_masked_regression_output(
                 bert_config, model.get_sequence_output(),
                 masked_lm_positions, masked_lm_values, masked_lm_weights)

            saver = tf.train.Saver(var_list=tf.trainable_variables())
            ckpt = tf.train.latest_checkpoint(args.model)
            saver.restore(sess, ckpt)

            _input_ids = []
            _lm_positions = []
            tokens = [enc.encode(p.strip()) for p in sep_txt(args.context)]
            tokens = [t for t in tokens if len(t) > 0]
            for t in tokens:
                _lm_positions.append(len(_input_ids))
                _input_ids.extend([CLS_TOKEN]+t)
            _input_ids.append(EOT_TOKEN)
            _input_masks = [1] * len(_input_ids)
            _segments = [1] * len(_input_ids)
            _input_ids = _input_ids[:max_seq_length]
            _input_masks = _input_masks[:max_seq_length]
            _segments = _segments[:max_seq_length]
            while len(_segments) < max_seq_length:
                _input_ids.append(0)
                _input_masks.append(0)
                _segments.append(0)
            _lm_positions = _lm_positions[:max_seq_length]
            _lm_lm_weights = [1] * len(_lm_positions)
            while len(_lm_positions) < max_seq_length:
                _lm_positions.append(0)
                _lm_lm_weights.append(0)
            _lm_ids = [0] * len(_lm_positions)
            _lm_vals = [0] * len(_lm_positions)

            regress = sess.run(
                outputs,
                feed_dict={
                    input_ids:[_input_ids],
                    input_mask:[_input_masks],
                    segment_ids:[_segments],
                    masked_lm_positions:[_lm_positions],
                    masked_lm_ids:[_lm_ids],
                    masked_lm_weights:[_lm_lm_weights],
                    next_sentence_labels:[0],
                    masked_lm_values:[_lm_vals]
                })
            regress = regress.reshape((-1,))
            if args.output_file == '':
                for tok,value in zip(tokens, regress):
                    print(f'{value}\t{enc.decode(tok)}')
            else:
                sent = []
                impt = []
                for tok,value in zip(tokens, regress):
                    sent.append(enc.decode(tok))
                    impt.append(value)
                df = pd.DataFrame({'sentence':sent,'importance':impt})
                df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()
