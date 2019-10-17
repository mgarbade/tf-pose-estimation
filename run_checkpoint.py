import argparse
import logging
import os

import tensorflow as tf
# tf.disable_v2_behavior() # TODO-MAYBE needed for "MACE" (XIAOMI) in the future
from tf_pose.networks import get_network, model_wh, _get_base_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num-stages', type=int, default=7)
    parser.add_argument('--out-graph', type=str, default='frozen_graph')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    # w, h = args.input_width, args.input_height
    if w <= 0 or h <= 0:
        w = h = None
    print(w, h)
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')

    net, pretrain_path, last_layer = get_network(args.model,
                                                 input_node,
                                                 sess_for_load=None,
                                                 trainable=False,
                                                 num_stages=args.num_stages)
    print("Last layer: ")
    print(last_layer)
    if args.quantize:
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)

    with tf.Session(config=config) as sess:
        loader = tf.train.Saver(net.restorable_variables())
        loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        tf.train.write_graph(sess.graph_def, args.checkpoint, 'graph_def_binary.pb', as_text=False)
        # tf.train.write_graph(sess.graph_def, './tmp', args.out_graph, as_text=False)
        # tf.train.write_graph(sess.graph_def, './tmp', "text_" + args.out_graph, as_text=True)

        flops = tf.profiler.profile(None, cmd='graph', options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP = ', flops.total_float_ops / float(1e6))

        graph = tf.get_default_graph()
        for n in tf.get_default_graph().as_graph_def().node:
             if 'concat_stage' not in n.name:
                 continue
             print(n.name)


