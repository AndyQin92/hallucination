import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import numpy as np
from collections import OrderedDict
from graph import GraphTripleConv, GraphTripleConvNet, FusionLayer



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def parse_glove_with_split(vocab, file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print len(lines)
    embed_dim =0 
    word2vec = dict()
    for line in lines:
        line = line.strip('\n').split(' ')
        word = line[0]
        vec = [float(line[i]) for i in range(1,len(line))]
        vec = np.array(vec, dtype = float)
        embed_dim = vec.shape[0]
        word2vec[word] = vec
    idx2vec = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    
    for i in range(len(vocab)):
        word = vocab.idx2word[int(i)]
        if word in word2vec:
            idx2vec[i] = word2vec[word]
        elif word =='<start>':
            idx2vec[i] = np.random.uniform(-0.1, 0.1, embed_dim)
        elif word == '<end>':
            idx2vec[i] = np.zeros(embed_dim, dtype= np.float32)
        elif '_' in word:
            word = word.split('_')
            tmp = np.zeros((len(word), embed_dim), dtype=np.float32)
            for k,w in enumerate(word):
                if w in word2vec:
                    tmp[k] = word2vec[w]
            idx2vec[i] = np.mean(tmp, axis=0)
        elif '-' in word:
            word = word.split('-')
            tmp = np.zeros((len(word), embed_dim), dtype=np.float32)
            for k,w in enumerate(word):
                if w in word2vec:
                    tmp[k] = word2vec[w]
            idx2vec[i] = np.mean(tmp, axis=0)

        else:
            idx2vec[i] = np.random.uniform(-0.25, 0.25, embed_dim)

    return idx2vec


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, obj_nums, captions, cap_lens, opt):
    """
    Images: (n_image, n_objs, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_word = captions.size(1)

    ## padding caption use 0
    # for i in range(n_caption):
    #     n_word = cap_lens[i]
    #     captions[i,n_word:, :] = torch.zeros(max_n_word-n_word, captions.size(2), dtype= captions.dtype).cuda()

    cap_lens = torch.tensor(cap_lens, dtype=captions.dtype)
    cap_lens = Variable(cap_lens).cuda()
    captions = torch.transpose(captions, 1, 2)
    for i in range(n_image):
        n_obj = obj_nums[i]
        img_i = images[i, : n_obj, :].unsqueeze(0).contiguous()
        # --> (n_caption , n_region ,d)
        img_i_expand = img_i.repeat(n_caption, 1, 1)
        # --> (n_caption, d, max_n_word)
        dot = torch.bmm(img_i_expand, captions)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=1, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        cap_lens = cap_lens.contiguous().view(-1,1)
        dot = dot/cap_lens
        # dot = dot.mean(dim=1, keepdim=True)
        dot = torch.transpose(dot, 0, 1)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)
    
    return similarities


def xattn_score_i2t(images, obj_nums, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_obj = images.size(1)

    # # padding iamge use 0
    # for i in range(n_image):
    #     n_obj = obj_nums[i]
    #     images[i, n_obj:,:]  = torch.zeros(max_n_obj-n_obj, images.size(2), dtype=images.dtype).cuda()

    obj_nums = torch.tensor(obj_nums, dtype=images.dtype)
    obj_nums = Variable(obj_nums).cuda()
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        cap_i_expand = cap_i_expand.contiguous()
        cap_i_expand = torch.transpose(cap_i_expand, 1,2)
        dot = torch.bmm(images, cap_i_expand)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=2, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        obj_nums = obj_nums.contiguous().view(-1,1)
        dot = dot/obj_nums
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, im_l, s, s_l, pred, pred_l, s_pred, s_pred_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores1 = xattn_score_t2i(im, im_l, s, s_l, self.opt)
            scores2 = xattn_score_t2i(pred, pred_l, s_pred, s_pred_l, self.opt)
            scores = scores1 + self.opt.predicate_score_rate*scores2 

        elif self.opt.cross_attn == 'i2t':
            scores1 = xattn_score_i2t(im, im_l, s, s_l, self.opt)
            scores2 = xattn_score_i2t(pred, pred_l, s_pred, s_pred_l, self.opt)
            scores = scores1 + self.opt.predicate_score_rate*scores2 
            
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]


        return cost_s.sum() + cost_im.sum()




""" The Architecture of the Visual Graph Encoder """
class EncoderImageSg(nn.Module):
    def __init__(self, img_dim, gconv_dim, word_dim, obj_vocab, rel_vocab, 
    			 glove=None, no_imgnorm=True, is_fusion=True,
                 fusion_activation='relu', fusion_method='concatenate', 
                 gconv_hidden_dim=1024, gconv_num_layers=5, alpha=1.0,
                 mlp_normalization='none',activation=None,
               **kwargs):
        super(EncoderImageSg, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.glove = glove
        self.no_imgnorm = no_imgnorm
        self.obj_vocab = obj_vocab
        self.rel_vocab = rel_vocab
        
        # Embedding layers -- word2vec
        self.obj_embed = nn.Embedding(len(obj_vocab), word_dim)
        self.rel_embed = nn.Embedding(len(rel_vocab), word_dim)
 
        # Multi-modal fusion Layer
        if is_fusion:
        	self.obj_fusion = FusionLayer(img_dim, word_dim, fusion_activation, fusion_method)
        	self.rel_fusion = FusionLayer(img_dim, word_dim, fusion_activation, fusion_method)
        else:
        	self.obj_fusion = None
        	self.rel_fusion = None
     	# GCN Net
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(img_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
            'input_dim': img_dim,
            'output_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'alpha':alpha,
            'mlp_normalization': mlp_normalization,
            'activation': activation,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'alpha':alpha,
            'num_layers': gconv_num_layers - 1,
            'mlp_normalization': mlp_normalization,
            'activation': activation
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        self.init_weights()

    def init_weights(self):
        if self.glove is None:
            self.obj_embed.weight.data.uniform_(-0.1, 0.1)
            self.rel_embed.weight.data.uniform_(-0.1, 0.1)
        else:
            idx2vec = parse_glove_with_split(self.obj_vocab, self.glove)
            self.obj_embed.weight.data.copy_(torch.from_numpy(idx2vec))

            idx2vec2 = parse_glove_with_split(self.rel_vocab, self.glove)
            self.rel_embed.weight.data.copy_(torch.from_numpy(idx2vec2))
        


    def forward(self, obj_embs, obj_nums, pred_embs, pred_nums, rels, objs):
        O, T = obj_embs.size(0), rels.size(0)
        s, p, o = rels.chunk(3, dim=1)           # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

        obj_label_embs = self.obj_embed(objs)
        if self.obj_fusion is not None:
            obj_vecs = self.obj_fusion(obj_embs, obj_label_embs)
        else:
            obj_vecs = obj_embs
        
        pred_label_embs = self.rel_embed(p)
        if self.rel_fusion is not None:
            pred_vecs = self.rel_fusion(pred_embs, pred_label_embs)
        else:
            pred_vecs = pred_embs
        

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        if not self.no_imgnorm:
            obj_vecs = l2norm(obj_vecs, dim=-1)
            pred_vecs = l2norm(pred_vecs, dim=-1)

        return obj_vecs, pred_vecs




""" The Architecture of the Text Graph Encoder """
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, glove=None, vocab=None,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.glove = glove
        self.vocab = vocab

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.wrnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.prnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        if self.glove is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            idx2vec = parse_glove_with_split(self.vocab, self.glove)
            self.embed.weight.data.copy_(torch.from_numpy(idx2vec))

    def forward(self, x, lengths, cap_rels, cap_rel_nums):
        """Handles variable size captions
        """

        """ Embed the word level feature """
        # Embed word ids to vectors
        total_length = x.size(1)
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.wrnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_len = cap_len.cuda()

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)



        """ Embed the relation level feature"""
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        cxt = torch.gather(cap_emb, 1, I).squeeze(1)

        # normalization in the joint embedding space
        #cxt = l2norm(cxt, dim=1)

        if len(cap_rels)>0:
            # sort the cap rels by the len of rels with ids
            data = zip(cap_rels, range(len(cap_rels)))
            data.sort(key=lambda x: len(x[0]), reverse=True)


            cap_triplets, cap_triplets_ids = zip(*data)
            cap_triplet_lens = [len(rel) for rel in cap_triplets]
            cap_triplet_targets = torch.zeros(len(cap_triplets), max(cap_triplet_lens)).long()
            for i, rel in enumerate(cap_triplets):
                end = cap_triplet_lens[i]
                cap_triplet_targets[i, :end] = rel[:end]
            cap_triplet_targets = cap_triplet_targets.cuda()
            cap_triplets_ids = torch.tensor(cap_triplets_ids).long().cuda()

            cap_rel_emb, cap_rel_nums = self.forward_prnn(cap_triplet_targets, cap_triplet_lens,cap_triplets_ids, cap_rel_nums, cxt)
        else:
            cap_rel_emb = torch.unsqueeze(cxt,dim=1) 
            cap_rel_nums = torch.ones(len(cxt), dtype=torch.int64).cuda()


        return cap_emb, cap_len, cap_rel_emb, cap_rel_nums


    def forward_prnn(self, cap_rels, cap_rel_lens, cap_rels_ids, cap_rel_nums, cxt):
        max_num = max(cap_rel_nums)
        assert len(cap_rels)== sum(cap_rel_nums)
        
        cap_rels = self.embed(cap_rels)
        packed = pack_padded_sequence(cap_rels, cap_rel_lens, batch_first=True)
        out, _ = self.prnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        sq_cap_rel_emb, sq_cap_rel_lens = padded
        if self.use_bi_gru:
            sq_cap_rel_emb = (sq_cap_rel_emb[:,:,:sq_cap_rel_emb.size(2)/2] + sq_cap_rel_emb[:,:,sq_cap_rel_emb.size(2)/2:])/2
        if not self.no_txtnorm:
            sq_cap_rel_emb = l2norm(sq_cap_rel_emb, dim=-1)

        I = torch.LongTensor(cap_rel_lens).view(-1, 1, 1)
        I = Variable(I.expand(cap_rels.size(0), 1, self.embed_size)-1).cuda()
        rel_cxt = torch.gather(sq_cap_rel_emb, 1, I).squeeze(1)

        
        ### sort back
        cap_rels_ids_arg = torch.argsort(cap_rels_ids)
        rel_cxt = rel_cxt[cap_rels_ids_arg]
        cap_rel_emb = torch.zeros(len(cap_rel_nums), max_num+1, self.embed_size, dtype=rel_cxt.dtype).cuda()

        rel_offset=0
        for i, l in enumerate(cap_rel_nums):
            cap_rel_emb[i][:l] = rel_cxt[rel_offset : rel_offset+l]
            cap_rel_emb[i][l] = cxt[i]
            rel_offset += l


        cap_rel_nums = [rel_num+1 for rel_num in cap_rel_nums]
        cap_rel_nums = torch.tensor(cap_rel_nums, dtype=torch.int64).cuda()
        return cap_rel_emb, cap_rel_nums



                 
class SGM(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImageSg(opt.img_dim, opt.embed_size, opt.word_dim, opt.obj_vocab, opt.rel_vocab, 
                                      glove = opt.glove, no_imgnorm = opt.no_imgnorm, is_fusion = opt.is_fusion, 
                                      fusion_activation = opt.fusion_activation, fusion_method = opt.fusion_method, 
                                      gconv_hidden_dim = 1024, gconv_num_layers = opt.gcn_num_layers,
                                      alpha = opt.alpha,
                                      activation = opt.activation)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   opt.glove, opt.vocab, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)


        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())


        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()


    def forward_emb(self, images, obj_nums, captions, lengths, 
    	            image_predicates, pred_nums, rels, objs, 
    	            cap_rels, cap_rel_nums,
    	            max_obj_n=36, max_pred_n = 25, volatile=False):
        
        images = Variable(images, volatile=volatile)
        image_predicates = Variable(image_predicates, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_predicates = image_predicates.cuda()
            rels = rels.cuda()
            objs = objs.cuda()
            

        img_vecs, pred_vecs = self.img_enc(images, obj_nums, image_predicates, pred_nums, rels, objs)
        cap_emb, cap_lens,cap_rel_emb, cap_rel_nums = self.txt_enc(captions, lengths, cap_rels, cap_rel_nums)

        img_emb = torch.zeros(len(obj_nums), max_obj_n, img_vecs.shape[1]).cuda()
        pred_emb = torch.zeros(len(pred_nums), max_pred_n, pred_vecs.shape[1]).cuda()

        obj_offset = 0
        for i, obj_num in enumerate(obj_nums):
        	img_emb[i][:obj_num] = img_vecs[obj_offset:obj_offset+obj_num]
        	obj_offset+= obj_num
        
        pred_offset = 0
        for i, pred_num in enumerate(pred_nums):
        	pred_emb[i][:pred_num] = pred_vecs[pred_offset : pred_offset + pred_num]
        	pred_offset+= pred_num



        return img_emb, cap_emb, cap_lens, pred_emb, cap_rel_emb, cap_rel_nums

    def forward_loss(self, img_emb, obj_nums, cap_emb, cap_len, pred_emb, pred_nums, cap_rel_emb, cap_rel_lens):
        
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, obj_nums, cap_emb, cap_len, pred_emb, pred_nums, cap_rel_emb, cap_rel_lens)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, obj_nums, captions, lengths, ids, image_predicates, pred_nums, rels, objs, cap_rels, cap_rel_nums):
    	self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, pred_emb, cap_rel_emb, cap_rel_lens = \
                     self.forward_emb(images, obj_nums, captions, lengths, image_predicates, pred_nums, rels, objs, cap_rels, cap_rel_nums)

        self.optimizer.zero_grad()

        loss = self.forward_loss(img_emb, obj_nums, cap_emb, cap_lens, pred_emb, pred_nums, cap_rel_emb, cap_rel_lens)
    
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()