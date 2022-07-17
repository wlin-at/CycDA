import torch
from torch import nn
from ..builder import RECOGNIZERS
from .base import BaseRecognizer

import warnings
from mmcv.cnn import normal_init
from mmaction.models.model_da.model_da import DA_clsfr, Cls_Head, LinearWeightedAvg
from mmaction.models.vit.temptransformer import TempTransformer
from ..model_da.model_da import adapt_weight, prepare_domain_labels, accuracy
import torch.nn.functional as F
# a new recognizer
@RECOGNIZERS.register_module()
class RecognizerI3dDA(BaseRecognizer): # todo   a recognizer consists of a backbone + a classification head
    """
    Recognizer with I3D inceptionv1  as backbone

    no temporal aggregation for clip-level representation !!!!!


    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 spatial_type = 'avg',
                 spatial_dropout_ratio = 0.5,
                 feat_dim = None,
                 n_class = None,
                 init_std = 0.01,


                 embed_dim_t = None,

                 # action classifiers, domain discriminators
                 d_clsfr_mlp_dim = None,
                 clsfr_mlp_dim = None,


                 tmp_agg_type = 'transformer',

                 # temporal transformer
                 n_transformer_blocks = None,
                 n_heads = None,
                 intermediate_dim = None,
                 n_clips = None,
                 token_pool_type = None,
                 head_dim=None,
                 transformer_dropout = None,
                 emb_dropout = None,


                 ):
        assert cls_head is None

        super(RecognizerI3dDA, self).__init__(backbone, cls_head, neck, train_cfg, test_cfg)
        self.spatial_type = spatial_type
        self.spatial_dropout_ratio = spatial_dropout_ratio
        self.feat_dim = feat_dim
        self.n_class = n_class
        self.init_std = init_std

        if self.spatial_type:
            self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1 ))
        else:
            self.avg_pool = None

        self.spatial_dropout  = nn.Dropout(p = self.spatial_dropout_ratio)

        self.dim_reduct = nn.Linear(feat_dim, embed_dim_t)

        # define our own classification head
        # self.clip_clsfr = nn.Linear(self.feat_dim, self.n_class)  #  one layer for classifier ??
        self.clip_clsfr = Cls_Head(embed_dim=embed_dim_t, mlp_dim=clsfr_mlp_dim, n_class=n_class)
        self.clip_d_clsfr = DA_clsfr(embed_dim= embed_dim_t, mlp_dim=  d_clsfr_mlp_dim)
        # self.vid_clsfr = Cls_Head(embed_dim=embed_dim_t, mlp_dim=clsfr_mlp_dim, n_class=n_class)
        # self.vid_d_clsfr = DA_clsfr(embed_dim=embed_dim_t, mlp_dim=d_clsfr_mlp_dim)

        # initialize the classification head
        normal_init(self.clip_clsfr, std=self.init_std)
        normal_init(self.clip_d_clsfr, std=self.init_std)
        # normal_init(self.vid_clsfr, std=self.init_std)
        # normal_init(self.vid_d_clsfr, std= self.init_std)

        # self.tmp_agg_type = tmp_agg_type
        #
        # # ## Temporal transformer
        # if self.tmp_agg_type == 'transformer':
        #     self.tmp_agg_model = TempTransformer(dim=embed_dim_t, depth=n_transformer_blocks, heads= n_heads,
        #                                          intermediate_dim=intermediate_dim, n_patches= n_clips, pool=token_pool_type, dim_head= head_dim,
        #                                          dropout= transformer_dropout, emb_dropout=emb_dropout)
        # elif self.tmp_agg_type =='weighted_avg':
        #     self.tmp_agg_model = LinearWeightedAvg(n_clips)





    def forward_domain(self, imgs, domain_label = None, beta = None,   ):
        # todo  inputs:  source data, domain_label, beta value
        #  outputs: pred_clip_cls, pred_vid_cls, pred_clip_d, pred_vid_d, feat

        # each video might consist of several views ( temporal clips x spatial crops), here we vectorize all the views of a video
        # (batch, n_views, 3, 32, 224,224 ) -> (batch * n_views, 3, 32, 224,224 )

        # todo check the reshape is row-wise or column-wise
        batch_sz, n_views = imgs.shape[0], imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:]) # (batch, n_views, 3, 32, 224,224 ) -> (batch * n_views, 3, 32, 224,224 )


        x = self.extract_feat(imgs)  # (batch * n_views, 3, 32, 224,224 ) -> (batch * n_views, 2048, 4, 7, 7  )
        if self.avg_pool is not None:
            x = self.avg_pool(x) # (batch * n_views, 2048, 4, 7, 7  ) -> (batch * n_views, 2048, 1, 1, 1  )
        x = self.spatial_dropout(x)
        x = x.view(x.shape[0], -1)  # ( batch*n_views, 2048, 1, 1, 1)  ->  ( batch*n_views, 2048 )

        # dimension reduction for clip-level representation before temporal aggregation
        x = self.dim_reduct(x) # ( batch*n_views, 2048 ) -> ( batch*n_views, embed_dim_t  )

        # clip-level classification
        pred_clip_cls = self.clip_clsfr(x)  # (batch * n_views, embed_dim_t ) ->  (batch * n_views, n_class  )

        # # clip-level domain discrimination
        # pred_clip_d = self.clip_d_clsfr(feat= x, beta = beta)  # (batch * n_views, embed_dim_t ) ->  (batch * n_views, 2   )

        # decompose into clip-level representation
        x = torch.unsqueeze( x, 0 ) # (batch * n_views, embed_dim_t  ) -> (1, batch * n_views, embed_dim_t )
        x = x.reshape(    (-1, n_views,  ) + x.shape[2: ]    )  # (1, batch * n_views, embed_dim_t) -> (batch,  n_views, embed_dim_t ), clip-level representation

        # #  temporal aggregation :
        # if self.tmp_agg_type == 'transformer':
        #     vid_rep = self.tmp_agg_model(x, domain_label)  # (batch, embed_dim_t  )
        # elif self.tmp_agg_type == 'weighted_avg':
        #     clip_rep_list = [  x[:, idx_, :]   for idx_ in range(n_views)] # a list of (batch, embed_dim_t)
        #     vid_rep = self.tmp_agg_model( clip_rep_list )
        # elif self.tmp_agg_type == 'avg':
        #     # temporal aggregation : averaging
        #     vid_rep = torch.mean(x, dim=1)
        #
        # # vid-level classification
        # pred_vid_cls = self.vid_clsfr(vid_rep)
        # # vid-level domain classification
        # pred_vid_d = self.vid_d_clsfr(feat= vid_rep, beta= beta)

        return pred_clip_cls

    def forward_train(self ,  data_batch,   **kwargs):
        DA_config = kwargs['DA_config']
        # batch_size = DA_config.batch_size
        w_pseudo = DA_config.w_pseudo
        adv_DA = DA_config.adv_DA

        experiment_type = DA_config.experiment_type
        use_target = DA_config.use_target
        weight_clip_clspred = DA_config.weight_clip_clspred  #  clip classification on source
        # weight_clip_domainpred = DA_config.weight_clip_domainpred
        # weight_vid_clspred = DA_config.weight_vid_clspred   #  video classification on source
        # weight_vid_domainpred = DA_config.weight_vid_domainpred

        if w_pseudo:
            # weight_vid_clspred_vid_ps = DA_config.weight_vid_clspred_vid_ps
            weight_clip_clspred_vid_ps = DA_config.weight_clip_clspred_vid_ps

        # beta_high = DA_config.beta_high
        n_clips = DA_config.n_clips

        ce = DA_config.ce
        # ce_d = DA_config.ce_d

        # iter_now = kwargs['iter']
        # max_iters = kwargs['max_iters']

        # beta = adapt_weight(iter_now, max_iters, weight_value=10.0, high_value=beta_high, low_value=0.0)

        loss = torch.tensor(0).float().cuda()

        #  data batches
        data_dict_source, data_dict_target = data_batch
        if_use_source_batch = not ( experiment_type == 'DA' and weight_clip_clspred == 0  )
        if_use_target_batch = not experiment_type in ['source_only', 'target_only']
        if if_use_source_batch:
            # todo perform forward pass on source batch
            imgs_s, label_s, vid_idx_s = data_dict_source['imgs'], data_dict_source['label'], data_dict_source['vid_idx']
            #  imgs_s  (batch, n_clips, 3, 32, 224,224)   imgs_s (batch, 1)    vid_idx_s (batch, 1)
            label_s = label_s.squeeze().long()
            #  (batch * n_views, n_class  ), (batch * n_views, 2 ),   (batch , n_class  ), (batch, 2 ),  (batch, dim)
            pred_clip_cls_s = self.forward_domain(imgs_s)


            label_s = label_s.repeat_interleave(n_clips)  # (batch_s * n_clips )
            # label_s = label_s.unsqueeze(0) if label_s.shape == torch.Size([]) else label_s

            # clip-level classification loss  on source
            loss_cls_clip_s = torch.tensor(0) if weight_clip_clspred == 0 else ce(pred_clip_cls_s, label_s)
            loss += loss_cls_clip_s * weight_clip_clspred
            # # video-level classification loss  on source
            # loss_cls_vid_s = torch.tensor(0) if weight_vid_clspred == 0 else ce(pred_vid_cls_s, label_s)
            # loss += loss_cls_vid_s * weight_vid_clspred

        if if_use_target_batch :
            # todo perform forward pass on target batch
            imgs_t, label_t, vid_idx_t = data_dict_target['imgs'], data_dict_target['label'], data_dict_target['vid_idx']
            if w_pseudo:
                ps_label_t = data_dict_target['ps_label']
            label_t = label_t.squeeze().long()
            pred_clip_cls_t  =self.forward_domain(imgs_t)
            label_t = label_t.repeat_interleave(n_clips)  # (batch_t * n_clips )
            # label_t = label_t.unsqueeze(0) if label_t.shape == torch.Size([]) else label_t
            #  clip-level classification loss with target ground truth,    only for logging purpose
            loss_cls_clip_t = ce(pred_clip_cls_t, label_t)
            # video-level classification loss with target ground truth,    only for logging purpose
            # loss_cls_vid_t = ce(pred_vid_cls_t, label_t)

            if w_pseudo:
                if weight_clip_clspred_vid_ps:
                    ps_label_t = ps_label_t.repeat_interleave(n_clips)  # (batch_s * n_clips )
                # loss_cls_vid_t_ps = torch.tensor(0) if weight_vid_clspred_vid_ps == 0 else ce(pred_vid_cls_t, ps_label_t)
                # loss += loss_cls_vid_t_ps * weight_vid_clspred_vid_ps
                loss_cls_clip_t_ps = torch.tensor(0) if weight_clip_clspred_vid_ps == 0 else ce(pred_clip_cls_t, ps_label_t)
                loss += loss_cls_clip_t_ps * weight_clip_clspred_vid_ps

            # if adv_DA and use_target != 'none':
            #     # loss_adv = torch.tensor(0).float().cuda()
            #     if weight_clip_domainpred != 0:
            #         label_clip_d = prepare_domain_labels(pred_clip_d_s, pred_clip_d_t)
            #         pred_clip_d = torch.cat((pred_clip_d_s, pred_clip_d_t), 0)
            #     loss_d_clp = torch.tensor(0) if weight_clip_domainpred == 0 else ce_d(pred_clip_d, label_clip_d)
            #
            #     if weight_vid_domainpred != 0:
            #         label_vid_d = prepare_domain_labels(pred_vid_d_s, pred_vid_d_t)
            #         pred_vid_d = torch.cat((pred_vid_d_s, pred_vid_d_t), 0)
            #     loss_d_vid = torch.tensor(0) if weight_vid_domainpred == 0 else ce_d(pred_vid_d, label_vid_d)
            #
            #     loss_adv = loss_d_clp * weight_clip_domainpred + loss_d_vid * weight_vid_domainpred
            #     loss += loss_adv

        losses = dict()  # for the purpose of logging, contains all the items that need to be logged

        if if_use_source_batch:
            if weight_clip_clspred != 0:
                top1_acc_clip_s, _ = accuracy(pred_clip_cls_s.data, label_s, topk=(1, 5))
                losses.update({
                    'loss_cls_clip_s': loss_cls_clip_s,
                    'top1_acc_clip_s': top1_acc_clip_s,
                    # 'top5_acc_clip_s': top5_acc_clip_s,
                })

            # if weight_vid_clspred != 0:
            #     top1_acc_vid_s, _ = accuracy(pred_vid_cls_s.data, label_s, topk=(1, 5))
            #     losses.update({
            #         'loss_cls_vid_s': loss_cls_vid_s,
            #         'top1_acc_vid_s': top1_acc_vid_s,
            #         # 'top5_acc_vid_s': top5_acc_vid_s,
            #     })
        if if_use_target_batch:
            # only for logging purpose
            top1_acc_clip_t, _ = accuracy(pred_clip_cls_t.data, label_t, topk=(1, 5))
            losses.update({
                'loss_cls_clip_t': loss_cls_clip_t,
                'top1_acc_clip_t': top1_acc_clip_t,
                # 'top5_acc_clip_t': top5_acc_clip_t,
            })

            # # only for logging purpose
            # top1_acc_vid_t, _ = accuracy(pred_vid_cls_t.data, label_t, topk=(1, 5))
            # losses.update({
            #     'loss_cls_vid_t': loss_cls_vid_t,
            #     'top1_acc_vid_t': top1_acc_vid_t,
            #     # 'top5_acc_vid_t': top5_acc_vid_t,
            # })

            if w_pseudo:
                losses.update({
                    # 'loss_cls_vid_t_ps': loss_cls_vid_t_ps,
                    'loss_cls_clip_t_ps': loss_cls_clip_t_ps,
                })

            # if adv_DA and use_target != 'none':
            #     losses.update({'beta': torch.tensor(beta).cuda()})
            #     if weight_clip_domainpred != 0:
            #         losses.update({'loss_d_clp': loss_d_clp})
            #     if weight_vid_domainpred != 0:
            #         losses.update({'loss_d_vid': loss_d_vid})

        # todo notice that not all the losses that we log  will be used in back propagation
        # _, log_vars = self._parse_losses(losses)

        # log_vars.update({ 'loss' : loss.item() } )
        # log_vars['loss'] = loss.item()  #

        # outputs = dict(
        #     loss=loss,  # for back propagation
        #     log_vars=log_vars,
        #     num_samples=batch_size
        # )
        return loss, losses


    def _do_test(self, imgs, domain_label = None, ):
        pass


    def forward_test(self, imgs, domain_label = None, **kwargs):
        batch_sz, n_views = imgs.shape[0], imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:]) # (batch, n_views, 3, 32, 224,224 ) -> (batch * n_views, 3, 32, 224,224 )
        x = self.extract_feat(imgs)  # (batch * n_views, 3, clip_len, 224,224 ) -> (batch * n_views, 2048,  clip_len/8, 7, 7  )
        if self.avg_pool is not None:
            x = self.avg_pool(x) # (batch * n_views, 2048, clip_len/8, 7, 7  ) -> (batch * n_views, 2048, 1, 1, 1  )
        x = self.spatial_dropout(x)
        x = x.view(x.shape[0], -1)  # ( batch*n_views, 2048, 1, 1, 1)  ->  ( batch*n_views, 2048 )

        # dimension reduction for clip-level representation before temporal aggregation
        x = self.dim_reduct(x) # ( batch*n_views, 2048 ) -> ( batch*n_views, embed_dim_t  )

        # if hasattr( kwargs['DA_config'], 'compute_clip_ps' ):
        #     if kwargs['DA_config'].compute_clip_ps:
        pred_clip_cls = self.clip_clsfr(x)  # (batch * n_views, embed_dim_t ) ->  (batch * n_views, n_class  )
        pred_clip_cls = torch.unsqueeze(pred_clip_cls, 0)  # (batch * n_views,  n_class  ) -> (1, batch * n_views, n_class )
        pred_clip_cls = pred_clip_cls.reshape((-1, n_views,) + pred_clip_cls.shape[ 2:])  # (1, batch * n_views, n_class) -> (batch,  n_views, n_class )
        # todo video-level score is the average of the clip-level softmax score
        vid_cls_score = F.softmax(  pred_clip_cls, dim=2).mean(dim=1)  # (batch, n_class )

        # # decompose into clip-level representation
        # x = torch.unsqueeze( x, 0 ) # (batch * n_views, embed_dim_t  ) -> (1, batch * n_views, embed_dim_t )
        # x = x.reshape(    (-1, n_views,  ) + x.shape[2: ]    )  # (1, batch * n_views, embed_dim_t) -> (batch,  n_views, embed_dim_t ), clip-level representation
        #
        # # #  temporal aggregation : temporal transformer
        # # vid_rep = self.tmp_agg_model(x, domain_label)  # (batch, embed_dim_t  )
        #
        # #  temporal aggregation :
        # if self.tmp_agg_type == 'transformer':
        #     vid_rep = self.tmp_agg_model(x, domain_label)  # (batch, embed_dim_t  )
        # elif self.tmp_agg_type == 'weighted_avg':
        #     clip_rep_list = [  x[:, idx_, :]   for idx_ in range(n_views)] # a list of (batch, embed_dim_t)
        #     vid_rep = self.tmp_agg_model( clip_rep_list )
        # elif self.tmp_agg_type == 'avg':
        #     # temporal aggregation : averaging
        #     vid_rep = torch.mean(x, dim=1)
        #
        # # vid-level classification
        # pred_vid_cls = self.vid_clsfr(vid_rep)  # (batch , n_class  )
        #
        #
        # if hasattr( kwargs['DA_config'], 'compute_clip_ps' ):
        #     if kwargs['DA_config'].compute_clip_ps:
        #         return pred_vid_cls.cpu().numpy(), pred_clip_cls.cpu().numpy()
        #
        # return pred_vid_cls.cpu().numpy()
        #  todo here softmax score is returned
        if hasattr(kwargs['DA_config'], 'compute_features'):
            if kwargs['DA_config'].compute_features:
                return vid_cls_score.cpu().numpy(), x.cpu().numpy()

        return vid_cls_score.cpu().numpy()

    def forward_dummy(self, imgs, softmax= False):
        """Used for computing network FLOPs.

                See ``tools/analysis/get_flops.py``.

                Args:
                    imgs (torch.Tensor): Input images.

                Returns:
                    Tensor: Class score.
                """
        pass

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
                utils."""