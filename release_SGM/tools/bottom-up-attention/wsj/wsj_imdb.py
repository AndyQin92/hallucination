import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import utils.cython_bbox
import cPickle
import gzip
import PIL
import json
from fast_rcnn.config import cfg

class wsj(imdb):
    def __init__(self, imageSet, split, version, imgRootDir, annFile):
        imdb.__init__(self, imageSet + '_' + split + version)
        self._data_path = annFile
        self._img_path = imgRootDir

        # sg information
        with open(self._data_path, 'r') as f:
            sg_data= json.load(f)
            self.object_num = sg_data['object_num']
            self.predicate_num = sg_data['predicate_num']
            img_to_sg = sg_data['img_to_sg']
            self.img_nums = len(img_to_sg)

            self.obj_num_per_img = [ sg['sg']['object'].index(-1) if -1 in sg['sg']['object'] else len(sg['sg']['object']) for sg in img_to_sg]
            self.predicate_num_per_img = [ sg['sg']['relationship'].index([-1,-1,-1]) if [-1,-1,-1] in sg['sg']['relationship'] else len(sg['sg']['relationship']) for sg in img_to_sg]

            self.obj_list_per_img = []
            self.rel_list_per_img = []
            self.obj_box_list_per_img = []
            for i, sg in enumerate(img_to_sg):
                obj_num = self.obj_num_per_img[i]
                self.obj_list_per_img.append( sg['sg']['object'][:obj_num])
                self.obj_box_list_per_img.append(sg['sg']['bbox'][:obj_num])

                pred_num  =self.predicate_num_per_img[i]
                self.rel_list_per_img.append(sg['sg']['relationship'][:pred_num])

        if imageSet == 'flickr30k':
            self.img_path_list = [os.path.join(self._img_path ,img['filename']) for img in img_to_sg]
        else:
            self.img_path_list = [os.path.join(self._img_path, img['filepath'], img['filename']) for img in img_to_sg]
        self._img_ids = [img['imageid'] for img in img_to_sg]
        
    def img_ids(self):
        return self._img_ids
    def image_path_at(self, index):
        image_path = self.img_path_list[index]
        return image_path
    def gt_roidb(self):
        gt_roidb =[ self._load_obj_annotation(index)
                    for index in range(self.img_nums)]
        return gt_roidb

    def _load_obj_annotation(self, index):
        width, height = self._get_size(index)
        obj_num = self.obj_num_per_img[index]
        boxes = np.zeros((obj_num, 4), dtype=np.float)
        box_list = self.obj_box_list_per_img[index]

        for i, box in enumerate(box_list):
            x1 = max(0, float(int(box[0])))
            y1 = max(0, float(int(box[1])))
            x2 = min(width-1, float(int(box[2])))
            y2 = min(height-1, float(int(box[3])))
            if x2 < x1 or y2 < y1:
                    #print 'Failed bbox in %s, object %s' % (filename, obj_name)
                    print('Failed bbox in %dth img', index)
                    x1 = 0
                    y1 = 0
                    x2 = width-1
                    y2 = height-1
            boxes[i,:] = [x1,y1,x2,y2]
        return {'boxes' : boxes}


    def _get_size(self, index):
        return PIL.Image.open(self.image_path_at(index)).size

    def gt_reldb(self):
        gt_reldb = [self._load_rel_annotation(index)
                    for index in range(self.img_nums)]
        return gt_reldb

    def _load_rel_annotation(self, index):
        width, height = self._get_size(index)
        obj_num = self.obj_num_per_img[index]
        pred_num = self.predicate_num_per_img[index]
        rel_boxes = np.zeros((pred_num,4), dtype= np.float)

        box_list = self.obj_box_list_per_img[index]
        rel_list = self.rel_list_per_img[index]

        for i, rel in enumerate(rel_list):
            s,p,o =rel
            s_box = box_list[s]
            o_box = box_list[o]

            x1 = max(0, min(float(int(s_box[0])), float(int(o_box[0]))))
            y1 = max(0, min(float(int(s_box[1])), float(int(o_box[1]))))
            x2 = min(width-1, max(float(int(s_box[2])), float(int(o_box[2]))))
            y2 = min(height-1, max(float(int(s_box[3])), float(int(o_box[3]))))

            if x2 < x1 or y2 < y1:
                    #print 'Failed bbox in %s, object %s' % (filename, obj_name)
                    print('Failed bbox in %dth img', index)
                    x1 = 0
                    y1 = 0
                    x2 = width-1
                    y2 = height-1
            rel_boxes[i,:] = [x1,y1,x2,y2]

        return {'boxes':rel_boxes}










