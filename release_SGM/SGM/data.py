import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json
import h5py
import ipdb as pdb


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, vocab, image_sg_file, caption_data_file, feature_name):
        self.vocab = vocab
        

        # Captions
        self.captions = []
        self.caption_rel = []
        with open(os.path.join(data_path, caption_data_file), 'rb') as f:
            caption_data = json.load(f)
            for ann in caption_data:
                for cap in ann['captions']:
                    self.captions.append(cap['sent'])
                    rel_list = []
                    for rel in cap['rels']:
                        if len(rel)==3:
                            rel_list.append(rel)
                    self.caption_rel.append(rel_list)

        # sg information
        with open(os.path.join(data_path, image_sg_file), 'r') as f:
            sg_data= json.load(f)
            self.obj_vocab = sg_data['object_dict']
            self.rel_vocab = sg_data['predicate_dict']
            self.object_num = sg_data['object_num']
            self.predicate_num = sg_data['predicate_num']
            img_to_sg = sg_data['img_to_sg']

            self.obj_num_per_img = [ sg['sg']['object'].index(-1) if -1 in sg['sg']['object'] else len(sg['sg']['object']) for sg in img_to_sg]
            self.predicate_num_per_img = [ sg['sg']['relationship'].index([-1,-1,-1]) if [-1,-1,-1] in sg['sg']['relationship'] else len(sg['sg']['relationship']) for sg in img_to_sg]

            self.obj_list_per_img = []
            self.rel_list_per_img = []
            for i, sg in enumerate(img_to_sg):
                obj_num = self.obj_num_per_img[i]
                self.obj_list_per_img.append( sg['sg']['object'][:obj_num])

                pred_num  =self.predicate_num_per_img[i]
                self.rel_list_per_img.append(sg['sg']['relationship'][:pred_num])


        if 'Flickr30k' in data_path:
            self.image_feature_root = os.path.join(data_path, 'npys', 'flickr30k_'+ data_split +'_image_obj_features_by_' +feature_name+'_36')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys', 'flickr30k_'+ data_split +'_image_rel_features_by_' +feature_name+'_25')
        elif feature_name == 'NM_BUA':
            self.image_feature_root = os.path.join(data_path, 'npys','NM_vgg_mscoco_'+ data_split +'2014_image_obj_features_by_BUA_36')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys','NM_vgg_mscoco_'+ data_split +'2014_image_rel_features_by_BUA_25')

        else:
            self.image_feature_root = os.path.join(data_path, 'npys','mscoco_'+ data_split +'2014_image_obj_features_by_' +feature_name+'_36')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys','mscoco_'+ data_split +'2014_image_rel_features_by_' +feature_name+'_25')

        

        
        self.length = len(self.captions)
        self.im_div = 5
        if data_split == 'val':
            self.length = 5000


    def __getitem__(self, index):
        # handle the image redundancy
        # Get obj features of an image
        img_id = index/self.im_div
        obj_num = self.obj_num_per_img[img_id]
        image = np.load(os.path.join(self.image_feature_root, str(img_id)+'.npy'))[:obj_num]
        image = torch.from_numpy(image)
        obj_list = self.obj_list_per_img[img_id]
        
        
        # Get relation features and rel list of an image
        pred_num = self.predicate_num_per_img[img_id]
        image_predicate = np.load(os.path.join(self.image_predicate_feature_root, str(img_id)+'.npy'))[:pred_num]
        image_predicate = torch.from_numpy(image_predicate)
        rel_list = self.rel_list_per_img[img_id]
        
        
        # Get the corresponding caption
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        # Get the relation list of caption
        caption_rel = self.caption_rel[index]
        cap_rel_list = []
        for rel in caption_rel:
            rel = rel[0] +' '+rel[2]+' ' +rel[1]
            tokens = nltk.tokenize.word_tokenize(str(rel).lower().decode('utf-8'))
            rel = []
            rel.append(vocab('<start>'))
            rel.extend([vocab(token) for token in tokens])
            rel.append(vocab('<end>'))
            rel = torch.Tensor(rel).long()
            cap_rel_list.append(rel)

        cap_rel_num = len(cap_rel_list)



        return image, obj_num, target, index, img_id, image_predicate, pred_num, rel_list, obj_list, cap_rel_list, cap_rel_num

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, obj_nums, captions, ids, img_ids, image_predicates, pred_nums, rel_lists, obj_lists, cap_rel_lists, cap_rel_nums = zip(*data)

    # Renumber all object in a batch
    objs = []
    for obj_list in obj_lists:
        objs = objs + obj_list
    objs = torch.tensor(objs, dtype=torch.int64)

    obj_offset = 0
    rels = []
    for i, rel_list in enumerate(rel_lists):
        obj_num = obj_nums[i]
        img_id = img_ids[i]
        pred_num = pred_nums[i]

        for s,p,o in rel_list:
            rels.append([s+obj_offset, p, o+obj_offset])
        obj_offset+= obj_num
    rels = torch.tensor(rels, dtype=torch.int64)
    images = torch.cat(images,0)
    image_predicates = torch.cat(image_predicates,0)
    assert len(rels) == len(image_predicates)


    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    cap_rels = []
    for cap_rel_list in cap_rel_lists:
        cap_rels += cap_rel_list

    
    return images, obj_nums, targets, lengths, ids, image_predicates, pred_nums, rels, objs, cap_rels, cap_rel_nums


def get_precomp_loader(data_path, data_split, file_list,feature_name, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, file_list[0], file_list[1], feature_name)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers = num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader, dset.obj_vocab, dset.rel_vocab


def get_loaders(data_name, train_file_list, val_file_list, feature_name, vocab, batch_size, workers, opt):

    dpath = opt.data_path
    if opt.debug:
        train_loader, obj_vocab, rel_vocab = get_precomp_loader(dpath, 'val', val_file_list, feature_name, vocab, opt,                                      batch_size, True, workers)
    else:
        train_loader, obj_vocab, rel_vocab = get_precomp_loader(dpath, 'train', train_file_list, feature_name, vocab, opt,
                                      batch_size, True, workers)
    val_loader,_,_ = get_precomp_loader(dpath, 'val', val_file_list, feature_name, vocab, opt,
                                    batch_size, False, workers)

    return train_loader, val_loader, obj_vocab, rel_vocab




def get_test_loader(split_name, data_name, test_file_list, feature_name, vocab, batch_size,
                    workers, opt):
    
    dpath = opt.data_path
    test_loader,_,_ = get_precomp_loader(dpath, split_name, test_file_list, feature_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader








    





















            
            

          