import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class DSLN(nn.Module):
    def __init__(self,  embed_dim = None, eps= 1e-12,   n_domains = 2,  ):
        super(DSLN, self).__init__()
        self.lns = nn.ModuleList( [ nn.LayerNorm( embed_dim, eps= eps  ) for _ in range(n_domains) ] )
    # def reset_running_stats(self):
    #     for ln in self.lns:
    #         ln.reset_running_stats()
    def forward(self, x, domain_label):
        ln = self.lns[domain_label]
        return ln(x)



class LinearWeightedSum(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedSum, self).__init__()
        self.weights = nn.ParameterList( [nn.Parameter(torch.randn(1))  for i in range(n_inputs)  ] )
    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res

class PreNorm(nn.Module):
    # apply laynorm before the specific function
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.dsln = DSLN( embed_dim= dim, )
        self.fn = fn
    def forward(self, x, domain_label,   **kwargs):
        # return self.fn(self.norm(x), **kwargs)
        return self.fn(self.dsln(x, domain_label), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) #  #  (batch, 1 + n_patches, dim ) ->   (batch, 1 + n_patches, inner_dim*3 ) -> a tuple of   (batch, 1 + n_patches, inner_dim )
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) #  a tuple of   (batch, 1 + n_patches, inner_dim )  ->  a tuple of       (batch,  heads,   1  + n_patches, head_dim  )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  #   (batch,  heads,   1  + n_patches, head_dim  )  matmul  (batch,  heads, head_dim,   1  + n_patches,   )   - >   (batch,  heads,   1  + n_patches,  1+n_patches  )

        attn = self.attend(dots)  # (batch,  heads,   1  + n_patches,  1+n_patches  )

        out = torch.matmul(attn, v) #  (batch,  heads,   1  + n_patches,  1+n_patches  ) matmul    (batch,  heads,   1  + n_patches, head_dim  )  ->   (batch,  heads,   1  + n_patches, head_dim  )
        out = rearrange(out, 'b h n d -> b n (h d)') #   (batch,  heads,   1  + n_patches, head_dim  ) ->   (batch,  1  + n_patches, inner_dim  )
        return self.to_out(out)   #  (batch,  1  + n_patches, inner_dim  ) ->  (batch,  1  + n_patches, dim   )

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, intermediate_dim, dropout = 0.):
        """

        :param dim:
        :param depth:    number of transformer blocks
        :param heads:
        :param dim_head:
        :param intermediate_dim:
        :param dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # apply the laynorm before
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, intermediate_dim, dropout = dropout))
            ]))
    def forward(self, x, domain_label):
        for attn, ff in self.layers:
            x = attn(x, domain_label) + x  #  (batch,  1  + n_patches, dim  )
            x = ff(x, domain_label) + x  #  (batch,  1  + n_patches, dim   )
        return x

class TempTransformer(nn.Module):
     #  *  is used to force the caller to use named arguments
    def __init__(self, *,  dim, depth, heads, intermediate_dim, n_patches,
                 pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        """
        Temporal transformer

        :param dim:
        :param depth:   # transformer blocks
        :param heads:
        :param intermediate_dim:
        :param pool:
        :param dim_head:
        :param dropout:
        :param emb_dropout:
        """
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # n_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean', 'w_sum'}, 'pool type must be either cls (cls token) or mean (mean pooling) or w_sum (weighted sum)'

        # self.to_patch_embedding = nn.Sequential(
        #
        #     # decompose and compose  b c (h p1) (w p2)  ->  b c h p1 w p2 ->  b (h w) (p1 p2 c)
        #     # flatten each patch,   batch,  n_patches, flattend_vec
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )


        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, intermediate_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),   #  todo    change  LayerNorm into domain-specific layer norm    dsln
        #     nn.Linear(dim, num_classes)
        # )
        self.dsln = DSLN(embed_dim=dim)

        if self.pool == 'w_sum':
            self.weighted_sum_layer = LinearWeightedSum( n_patches +1 )

    def forward(self, x, domain_label):
        """

        :param x:   (batch, n_patches, dim )
        :return:
        """
        # x = self.to_patch_embedding(img)  # (batch, c, ori_w, ori_h) ->  (batch, n_patches, patch_dim ) ->  (batch, n_patches, dim )

        b, n, _ = x.shape  # n number of patches

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # (batch, 1, dim )
        x = torch.cat((cls_tokens, x), dim=1)  #  (batch, 1 + n_patches, dim )

         # position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, domain_label)  #  #  (batch, 1 + n_patches, dim )

        # if pool is 'mean', the average of all the tokens are taken
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  #  #  (batch,  dim )

        if self.pool == 'cls':
            x = x [:, 0]   # take the class token
        elif self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'w_sum':
            token_list = [  x[:, idx, : ]   for idx in range( x.shape[1] )   ]
            x = self.weighted_sum_layer( token_list) # (batch, dim )

        x = self.dsln(x, domain_label)  #  video-level representation

        # x = self.to_latent(x)
        # return self.mlp_head(x)
        return x
