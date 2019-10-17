import argparse
import logging
import os

import tensorflow as tf
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
    parser.add_argument('--out-graph', type=str, default='frozen_graph')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    # w, h = args.input_width, args.input_height
    if w <= 0 or h <= 0:
        w = h = None
    print(w, h)
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')

    net, pretrain_path, last_layer = get_network(args.model, input_node, None, trainable=False)
    print("Last layer: ")
    print(last_layer)

    if args.quantize:
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)

    with tf.Session(config=config) as sess:
        loader = tf.train.Saver(net.restorable_variables())
        #loader.restore(sess, pretrain_path)
        if args.checkpoint and os.path.isdir(args.checkpoint):
            loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))


            # Convert network to inference only graph
            # is_training = tf.placeholder (tf.bool, name = 'is_training')
            # sess.run (net, feed_dict= {x: x, is_training: is_training})
            # is_training = False
            # sess.run (net, feed_dict= {x: x, is_training: is_training})


            tf.train.write_graph(sess.graph_def, './tmp', args.out_graph + '_text.pb', as_text=True)
            tf.train.write_graph(sess.graph_def, './tmp', args.out_graph + '_binary.pb', as_text=False)

            flops = tf.profiler.profile(None, cmd='graph', options=tf.profiler.ProfileOptionBuilder.float_operation())
            print('FLOP = ', flops.total_float_ops / float(1e6))

            graph = tf.get_default_graph()
            for n in tf.get_default_graph().as_graph_def().node:
                 if 'concat_stage' not in n.name:
                     continue
                 print(n.name)

            # saver = tf.train.Saver(max_to_keep=100)
            # saver.save(sess, './tmp/m2_1.4_chk', global_step=1)

        else:
            print("Error: Checkpoint  or checkpoint dir not found")

