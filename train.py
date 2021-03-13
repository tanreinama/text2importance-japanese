import argparse
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time
from tqdm import tqdm
from multiprocessing import Pool
from copy import copy
from tensorflow.core.protobuf import rewriter_config_pb2

from model import modeling
from model.modeling import BertConfig, BertModel

from text2importance import get_masked_regression_output,get_masked_lm_output,get_next_sentence_output

from encode_bpe import BPEEncoder_ja

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RoBERTa-ja_small', help='pretrained model directory.')
parser.add_argument('--input_dir', type=str, required=True, help='input texts.')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size.')
parser.add_argument('--run_name', type=str, default='text2importance-ja_small', help='save model name.')
parser.add_argument('--save_every', type=int, default=2, help='save every N epoch.')
parser.add_argument('--num_epochs', type=int, default=2, help='training epochs.')
parser.add_argument('--num_encode_process', type=int, default=12, help='encode processes.')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate.')
parser.add_argument('--gpu', default='0', help='visible gpu number.')
parser.add_argument('--do_eval', action='store_true', help='make evaluate data.')
parser.add_argument('--eval_rate', default=0.1, type=float, help='evaluate data rate.')

CHECKPOINT_DIR = 'checkpoint'

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def encode_one(f):
    with open(f, encoding='utf-8') as fn:
        lm_tokens = []
        lm_positions = []
        lm_imprtances = []
        for p in fn.readlines():
            p = p.split('\t')
            tokens = enc.encode(p[1].strip())
            lm_positions.append(len(lm_tokens))
            lm_imprtances.append(float(p[0]))
            lm_tokens.extend([CLS_TOKEN]+tokens)
        lm_tokens.append(EOT_TOKEN)
    return (lm_tokens,lm_positions,lm_imprtances)

def main():
    global EOT_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN, enc
    args = parser.parse_args()

    if os.path.isfile(args.model+'/hparams.json'):
        with open(args.model+'/hparams.json') as f:
            bert_config_params = json.load(f)
    else:
        raise ValueError('invalid model name.')

    vocab_size = bert_config_params['vocab_size']
    max_seq_length = bert_config_params['max_position_embeddings']
    batch_size = args.batch_size
    save_every = args.save_every
    num_epochs = args.num_epochs
    EOT_TOKEN = vocab_size - 4
    MASK_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 1

    with open('ja-bpe.txt', encoding='utf-8') as f:
        bpe = f.read().split('\n')

    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())

    enc = BPEEncoder_ja(bpe, emoji)

    fl = [f'{args.input_dir}/{f}' for f in os.listdir(args.input_dir)]
    with Pool(args.num_encode_process) as pool:
        imap = pool.imap(encode_one, fl)
        input_contexts = list(tqdm(imap, total=len(fl)))
    input_indexs = np.random.permutation(len(input_contexts))

    if args.do_eval:
        eval_num = int(args.eval_rate * len(input_indexs))
        eval_input_indexs = input_indexs[:eval_num]
        input_indexs = input_indexs[eval_num:]

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
            is_training=True,
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
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)
        train_vars = tf.trainable_variables()
        restored_weights = {}
        for i in range(len(train_vars)):
            restored_weights[train_vars[i].name] = sess.run(train_vars[i])

        labels = tf.placeholder(tf.float32, [None, ])

        output_layer = model.get_pooled_output()

        if int(tf.__version__[0]) > 1:
            hidden_size = output_layer.shape[-1]
        else:
            hidden_size = output_layer.shape[-1].value

        masked_lm_values = tf.placeholder(tf.float32, [None, None])

        with tf.variable_scope("loss"):
            (loss,_) = get_masked_regression_output(
                 bert_config, model.get_sequence_output(),
                 masked_lm_positions, masked_lm_values, masked_lm_weights)

            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_vars = tf.trainable_variables()
            opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summaries = tf.summary.scalar('loss', loss)
            summary_log = tf.summary.FileWriter(
                os.path.join(CHECKPOINT_DIR, args.run_name))

            counter = 1
            counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
            if os.path.exists(counter_path):
                # Load the step number if we're resuming a run
                # Add 1 so we don't immediately try to save again
                with open(counter_path, 'r') as fp:
                    counter = int(fp.read()) + 1

            hparams_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'hparams.json')
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            with open(hparams_path, 'w') as fp:
                fp.write(json.dumps(bert_config_params))

            sess.run(tf.global_variables_initializer()) # init output_weights
            restored = 0
            for k,v in restored_weights.items():
                for i in range(len(train_vars)):
                    if train_vars[i].name == k:
                        assign_op = train_vars[i].assign(v)
                        sess.run(assign_op)
                        restored += 1
            assert restored == len(restored_weights), 'fail to restore model.'
            saver = tf.train.Saver(var_list=tf.trainable_variables())

            def save():
                maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
                print(
                    'Saving',
                    os.path.join(CHECKPOINT_DIR, args.run_name,
                                 'model-{}').format(counter))
                saver.save(
                    sess,
                    os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                    global_step=counter)
                with open(counter_path, 'w') as fp:
                    fp.write(str(counter) + '\n')
            avg_loss = (0.0, 0.0)
            start_time = time.time()

            def sample_feature(i, eval=False):
                indexs = eval_input_indexs if eval else input_indexs
                last = min((i+1)*batch_size,len(indexs))
                _input_ids = []
                _input_masks = []
                _segments = []
                _lm_positions = []
                _lm_vals = []
                _lm_lm_weights = []
                _lm_ids = []
                for j in range(i*batch_size,last,1):
                    (lm_tokens,lm_positions,lm_imprtances) = input_contexts[indexs[j]]
                    ids = copy(lm_tokens)[:max_seq_length]
                    seg = [1] * len(ids)
                    while len(ids) < max_seq_length:
                        ids.append(0)
                        seg.append(0)
                    _input_ids.append(ids)
                    _input_masks.append(seg)
                    _segments.append(seg)
                    pos = copy(lm_positions)[:max_seq_length]
                    val = copy(lm_imprtances)[:max_seq_length]
                    wei = [1] * len(pos)
                    while len(ids) < max_seq_length:
                        pos.append(0)
                        val.append(0)
                        wei.append(0)
                    _lm_positions.append(pos)
                    _lm_ids.append([0]*max_seq_length)
                    _lm_lm_weights.append(wei)
                    _lm_vals.append(val)

                return {
                    input_ids:_input_ids,
                    input_mask:_input_masks,
                    segment_ids:_segments,
                    masked_lm_positions:_lm_positions,
                    masked_lm_ids:_lm_ids,
                    masked_lm_weights:_lm_lm_weights,
                    next_sentence_labels:[0]*len(_input_ids),
                    masked_lm_values:_lm_vals
                }

            try:
                for ep in range(num_epochs):
                    if ep % args.save_every == 0:
                        save()

                    prog = tqdm(range(0,len(input_indexs)//batch_size,1))
                    for i in prog:
                        (_, v_loss, v_summary) = sess.run(
                            (opt_apply, loss, summaries),
                            feed_dict=sample_feature(i))

                        summary_log.add_summary(v_summary, counter)

                        avg_loss = (avg_loss[0] * 0.99 + v_loss,
                                    avg_loss[1] * 0.99 + 1.0)

                        prog.set_description(
                            '[{ep} | {time:2.0f}] loss={loss:.4f} avg={avg:.4f}'
                            .format(
                                ep=ep,
                                time=time.time() - start_time,
                                loss=v_loss,
                                avg=avg_loss[0] / avg_loss[1]))

                        counter += 1

                    if args.do_eval:
                        eval_losses = []
                        for i in tqdm(range(0,len(eval_input_indexs)//batch_size,1)):
                            eval_losses.append(sess.run(loss, feed_dict=sample_feature(i, True)))
                        print("eval loss:",np.mean(eval_losses))

            except KeyboardInterrupt:
                print('interrupted')
                save()

            save()


if __name__ == '__main__':
    main()
