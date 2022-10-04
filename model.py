""" CIBN model"""
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from sim_Reason import Reason_i2t, Reason_t2i
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from relation_Network import ImageRelationNTN, TextRelationNTN
from graph_Convolution import VisualConvolution, TextualConvolution


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


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features  # features[:,:,:-64]

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)

class WordEmbeddings(nn.Module):
    def __init__(self):
        super(WordEmbeddings, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)[0]
        x = x[-4:]
        x = torch.stack(x, dim=1)
        x = torch.sum(x, dim=1)
        return x

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # bert embedding
        self.embedd = WordEmbeddings()

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # bert_embedding
        x = self.embedd(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                       cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.Rank_Loss = opt.Rank_Loss

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
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

        # keep the DynamciTopK, maximum or all violating negative for each query

        if self.Rank_Loss == 'DynamicTopK_Negative':
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.Rank_Loss == 'Hardest_Negative':
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()

class Balance_information(nn.Module):

    def __init__(self, opt):
        super(Balance_information, self).__init__()
        self.opt = opt
        self.add_textual_features = nn.Parameter(torch.FloatTensor(opt.batch_size, 1, opt.Add_features))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.add_textual_features, 0, 1)

    def forward(self, cap_emb, img_emb):

        batch_size, textual_len = cap_emb.size(0), cap_emb.size(1)

        if batch_size == self.opt.batch_size:
            add_features = self.add_textual_features.expand(batch_size, textual_len,
                                                            self.opt.Add_features).contiguous()
        else:
            add_features = self.add_textual_features[:batch_size, :, :]
            add_features = add_features.expand(batch_size, textual_len,
                                               self.opt.Add_features).contiguous()

        cap_emb = torch.cat((cap_emb, add_features), dim=-1)

        # visual features
        img_emb = img_emb[:, :, :self.opt.embed_size]

        return cap_emb, img_emb

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Balance_information, self).load_state_dict(new_state)


class CIBN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        # Encoding
        self.img_enc = EncoderImagePrecomp(
            opt.img_dim, opt.embed_size + opt.Reduce_features, opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.word_dim,
                                   opt.embed_size - opt.Add_features, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)

        # Convolution
        self.visual_convolution = VisualConvolution(opt.embed_size + opt.Reduce_features,
                                                    opt.embed_size + opt.Reduce_features, n_kernels=8, bias=False)
        self.textual_convolution = TextualConvolution(opt.embed_size - opt.Add_features,
                                                      opt.embed_size - opt.Add_features, n_kernels=8, bias=False)

        # Convolution weights
        self.visual_weights = ImageRelationNTN(opt.image_K, opt.embed_size + opt.Reduce_features, bias=True)
        self.textual_weights = TextRelationNTN(opt.text_K, opt.embed_size - opt.Add_features, bias=True)


        # Balance information
        self.balance_information = Balance_information(opt)

        # Matching
        self.i2t_match = Reason_i2t(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)
        self.t2i_match = Reason_t2i(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.visual_convolution.cuda()
            self.textual_convolution.cuda()
            self.visual_weights.cuda()
            self.textual_weights.cuda()
            self.i2t_match.cuda()
            self.t2i_match.cuda()
            self.balance_information.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.visual_convolution.parameters())
        params += list(self.textual_convolution.parameters())
        params += list(self.visual_weights.parameters())
        params += list(self.textual_weights.parameters())
        params += list(self.i2t_match.parameters())
        params += list(self.t2i_match.parameters())
        params += list( self.balance_information.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

        self.opt = opt

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.visual_convolution.state_dict(),
                      self.textual_convolution.state_dict(),
                      self.visual_weights.state_dict(),
                      self.textual_weights.state_dict(),
                      self.i2t_match.state_dict(),
                      self.t2i_match.state_dict(),
                      self.balance_information.state_dict()
                      ]
        return state_dict

    def load_state_dict(self, state_dict):

        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.visual_convolution.load_state_dict(state_dict[2])
        self.textual_convolution.load_state_dict(state_dict[3])
        self.visual_weights.load_state_dict(state_dict[4])
        self.textual_weights.load_state_dict(state_dict[5])
        self.i2t_match.load_state_dict(state_dict[6])
        self.t2i_match.load_state_dict(state_dict[7])
        self.balance_information.load_state_dict((state_dict[8]))


    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.visual_convolution.train()
        self.textual_convolution.train()
        self.visual_weights.train()
        self.textual_weights.train()
        self.i2t_match.train()
        self.t2i_match.train()
        self.balance_information.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.visual_convolution.eval()
        self.textual_convolution.eval()
        self.visual_weights.eval()
        self.textual_weights.eval()
        self.i2t_match.eval()
        self.t2i_match.eval()
        self.balance_information.eval()

    def neighbor_matrx(self, lens):
        adj = np.zeros((lens, lens), dtype=np.int)
        for i in range(lens):
            for j in range(lens):
                if abs(i - j) <= self.opt.windows_size:
                    adj[i, j] = 1
        return torch.from_numpy(adj).cuda().float()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward

        img_emb = self.img_enc(images)

        if self.opt.Visual_convolution == True:
            visual_weights = self.visual_weights(img_emb)
            # visual_weights = l2norm(visual_weights, dim=2)
            img_emb = self.visual_convolution(img_emb, visual_weights)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        # batch_size, textual_len = cap_emb.size(0), cap_emb.size(1)

        if self.opt.Textual_convolution == True:
            # adj_matrix = self.neighbor_matrx(textual_len).repeat(batch_size, 1, 1)
            textual_weights = l2norm(self.textual_weights(cap_emb), dim=2)
            # textual_weights = l2norm(textual_weights * adj_matrix, dim=2)
            cap_emb = self.textual_convolution(cap_emb, textual_weights)

        cap_emb, img_emb = self.balance_information(cap_emb, img_emb)

        # balance information
        # textual features
        # if batch_size == self.opt.batch_size:
        #     add_features = self.add_textual_features.expand(batch_size, textual_len,
        #                                                     self.opt.Add_features).contiguous().cuda()
        # else:
        #     add_features = self.add_textual_features[:batch_size, :, :]
        #     add_features = add_features.expand(batch_size, textual_len,
        #                                        self.opt.Add_features).contiguous().cuda()
        # cap_emb = torch.cat((cap_emb, add_features), dim=-1)
        #
        # # visual features
        # img_emb = img_emb[:, :, :self.opt.embed_size]

        return img_emb, cap_emb, cap_lens

    def forward_sim(self, img_emb, cap_emb, cap_lens):
        if self.opt.Matching_direction == 'i2t':
            i2t_scores = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return i2t_scores
        elif self.opt.Matching_direction == 't2i':
            t2i_scores = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores
        else:
            t2i_scores = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            i2t_scores = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores + i2t_scores

    def forward_loss(self, scores, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths)

        scores = self.forward_sim(img_emb, cap_emb, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(scores)

        # # log Loss
        # file = open(self.opt.model_name + '/' + self.opt.region_relation + '/' + '%s_%s_Loss.txt' %(self.opt.region_relation, self.opt.windows_size), 'a' )
        # file.write(str(self.Eiters) + "    " + str(loss.item()) + "\n")
        # file.close()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
