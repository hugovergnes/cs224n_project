from audioop import bias
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from util import masked_softmax


class Conv1dNonlin(nn.Module):
    """Generic class for 1d convolutions"""

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv1d,
        # nonlin=nn.LeakyReLU, # Also GeLU ?
        nonlin=nn.ReLU,
        groups=1,
        bias=True,
        use_non_lin=True,
    ):
        super(Conv1dNonlin, self).__init__()
        self.conv_kwargs = {
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "groups": groups,
            "bias": bias,
        }
        # Use that for LeakyReLU
        # self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': False}
        self.nonlin_kwargs = {"inplace": False}
        self.norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}

        self.use_non_lin = use_non_lin

        self.conv = conv_op(input_channels, output_channels, **self.conv_kwargs)
        if use_non_lin:
            self.non_linearity = nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        if self.use_non_lin:
            return self.non_linearity(self.conv(x))
        else:
            return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, number_of_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        self.mem_conv = Conv1dNonlin(
            hidden_size, 2 * hidden_size, bias=False, use_non_lin=False
        )
        self.query_conv = Conv1dNonlin(
            hidden_size, hidden_size, bias=False, use_non_lin=False
        )

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.number_of_heads)
        K, V = [
            self.split_last_dim(tensor, self.number_of_heads)
            for tensor in torch.split(memory, self.hidden_size, dim=2)
        ]

        key_depth_per_head = self.hidden_size // self.number_of_heads
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = logits * mask.type(torch.float32) + (
                1 - mask.type(torch.float32)
            ) * (-1e30)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


# For encoding block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channel, output_channel, conv_kernel_size, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=input_channel,
            out_channels=input_channel,
            kernel_size=conv_kernel_size,
            groups=input_channel,
            padding=conv_kernel_size // 2,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.device)).transpose(1, 2)


# Key component of QAnet. Only conv and self attentions.
# Default are value from the paper
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        number_of_convolutions=4,
        depthwise_separable_convs_kernel=7,
        number_of_heads=8,
        drop_prob=0.1,
    ):
        super().__init__()
        self.encoding_convs = nn.ModuleList(
            [
                DepthwiseSeparableConv(
                    hidden_size, hidden_size, depthwise_separable_convs_kernel
                )
                for _ in range(number_of_convolutions)
            ]
        )

        self.self_att = SelfAttention(hidden_size, number_of_heads, drop_prob)
        self.FFN_1 = Conv1dNonlin(hidden_size, hidden_size, use_non_lin=True, bias=True)
        self.FFN_2 = Conv1dNonlin(
            hidden_size, hidden_size, use_non_lin=False, bias=True
        )
        self.norm_C = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(number_of_convolutions)]
        )
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.number_of_convolutions = number_of_convolutions
        self.drop_prob = drop_prob

    def forward(self, x, mask, layer_number, number_of_blocks):
        total_number_of_layers = (self.number_of_convolutions + 1) * number_of_blocks
        out = PosEncoder(x)

        for i, conv in enumerate(self.encoding_convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out)
            out = self.layer_dropout(
                out, res, self.drop_prob * float(layer_number) / total_number_of_layers
            )
            layer_number += 1

        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(
            out, res, self.drop_prob * float(layer_number) / total_number_of_layers
        )
        layer_number += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(
            out, res, self.drop_prob * float(layer_number) / total_number_of_layers
        )
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = Conv1dNonlin(2 * hidden_size, 1, use_non_lin=False, bias=False)
        self.w2 = Conv1dNonlin(2 * hidden_size, 1, use_non_lin=False, bias=False)

    def forward(self, M1, M2, M3, mask):
        logits_1 = self.w1(torch.cat([M1, M2], dim=1))
        logits_2 = self.w2(torch.cat([M1, M3], dim=1))
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2


class QAnet(nn.Module):
    def __init__(
        self,
        word_vectors,
        char_vectors,
        hidden_size,
        drop_prob=0.0,
        number_of_heads=1,
        number_of_encoder_blocks=7,
    ):
        super().__init__()
        # Embed
        self.emb = layers.Embedding(
            word_vectors=word_vectors,
            char_vectors=char_vectors,
            hidden_size=hidden_size,
            drop_prob=drop_prob,
        )

        # Input Encode
        self.encoder = Encoder(hidden_size=hidden_size, number_of_heads=number_of_heads)
        self.cq_attention = layers.BiDAFAttention(
            hidden_size=hidden_size, drop_prob=drop_prob
        )

        # Encoder Blocks
        self.encoder_blocks = nn.ModuleList(
            [
                Encoder(
                    number_of_convolutions=2,
                    hidden_size=hidden_size,
                    number_of_heads=number_of_heads,
                    depthwise_separable_convs_kernel=5,
                    drop_prob=0.1,
                )
                for _ in range(number_of_encoder_blocks)
            ]
        )

        # Decoder
        self.decoder = Decoder(hidden_size)

        # Utils
        self.cq_resizer = Conv1dNonlin(
            4 * hidden_size, hidden_size, bias=False, use_non_lin=False
        )
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs, cc_idxs).transpose(1, 2) # bs, h, c_len
        q_emb = self.emb(qw_idxs, qc_idxs).transpose(1, 2)

        c_enc = self.encoder(c_emb, c_mask, 1, 1)
        q_enc = self.encoder(q_emb, q_mask, 1, 1)

        out = self.cq_attention(
            c_enc.transpose(1, 2), q_enc.transpose(1, 2), c_mask, q_mask
        ).transpose(1, 2)

        out = F.dropout(self.cq_resizer(out), p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)
        M1 = out

        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)

        M2 = out
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)
        M3 = out

        log_p1, log_p2 = self.decoder(M1, M2, M3, c_mask)
        return log_p1, log_p2
