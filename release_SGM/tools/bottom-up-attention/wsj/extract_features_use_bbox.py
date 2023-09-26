import _init_paths
from fast_rcnn.test import test_net_with_gt_boxes_without_eval, test_net_with_gt_union_boxes_without_eval
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
from wsj_imdb import wsj
import time

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    # parser.add_argument('--wait', dest='wait',
    #                     help='wait until net file exists',
    #                     default=True, type=bool)
    # parser.add_argument('--imdb', dest='imdb_name',
    #                     help='dataset to test',
    #                     default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--vis', dest='vis', help='visualize detections',
    #                     action='store_true')
    # parser.add_argument('--num_dets', dest='max_per_image',
    #                     help='max number of detections per image',
    #                     default=400, type=int)
    # parser.add_argument('--rpn_file', dest='rpn_file',
    #                     default=None, type=str)
    parser.add_argument('--imageSet', type=str, default='mscoco')
    parser.add_argument('--split', type=str, default='train',
                        help='train|val|test')
    parser.add_argument('--version', type=str, default='2014')
    parser.add_argument('--imgRootDir', type=str)
    parser.add_argument('--annFile', type=str)
    parser.add_argument('--dist_file', type=str)
    parser.add_argument('--mode', type=str, default='obj',
                        help='extract object feature or relationship feature')



    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = wsj(args.imageSet, args.split, args.version, args.imgRootDir, args.annFile)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    print(time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime()))
    if args.mode =='obj':
        print('extracting obj  features')
        test_net_with_gt_boxes_without_eval(net, imdb, args.dist_file)
    if args.mode =='rel':
        print('extracting relationship features')
        test_net_with_gt_union_boxes_without_eval(net, imdb, args.dist_file)

    print(time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime()))