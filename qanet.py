import math

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
        nonlin=nn.GELU,
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
        self.nonlin_kwargs = {}

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
    def __init__(self, hidden_size, number_of_heads, drop_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.drop_prob = drop_prob
        self.mem_conv = Conv1dNonlin(
            hidden_size, 2 * hidden_size, bias=False, use_non_lin=False
        )
        self.query_conv = Conv1dNonlin(
            hidden_size, hidden_size, bias=False, use_non_lin=False
        )

    def forward(self, x, mask):
        bs, num_channels, T = x.size()
        memory = x

        memory = self.mem_conv(memory)
        query = self.query_conv(x)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = query.view(bs, T, self.number_of_heads, num_channels // self.number_of_heads).transpose(1, 2)
        K, V = [tensor.view(bs, T, self.number_of_heads, num_channels // self.number_of_heads).transpose(1, 2)
            for tensor in torch.split(memory, self.hidden_size, dim=2)]

        x = self.make_attention(Q, K, V, mask=mask)
        return x.transpose(1, 2).contiguous().view(bs, T, num_channels).transpose(1, 2)

    def make_attention(self, q, k, v, mask=None):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = mask.view(att.shape[0], 1, 1, att.shape[-1])
        weights = masked_softmax(att, mask, dim=-1, log_softmax=False)
        weights = F.dropout(weights, p=self.drop_prob, training=self.training)
        return torch.matmul(weights, v)


class DepthConv(nn.Module):
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


class Encoder(nn.Module):
    """Key component of QAnet. Only conv and self attentions. Default are value from the paper
    """
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
                DepthConv(
                    hidden_size, hidden_size, depthwise_separable_convs_kernel
                )
                for _ in range(number_of_convolutions)
            ]
        )

        self.self_att = SelfAttention(hidden_size, number_of_heads, drop_prob)
        self.feed_forward_networks = nn.ModuleList(
            [
                Conv1dNonlin(hidden_size, hidden_size, use_non_lin=True, bias=True),
                Conv1dNonlin(hidden_size, hidden_size, use_non_lin=False, bias=True),
            ]
        )
        self.encoding_layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(number_of_convolutions)]
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.number_of_convolutions = number_of_convolutions
        self.drop_prob = drop_prob

    @staticmethod
    def add_positional_encoding(x, denominator_power=1.0e4):
        """Add positional encoding to x as described in attention is all you need.

        Args:
            x (torch.Tensor): (bs, hidden_size, context_length)
            denominator_power (float, optional): _description_. Defaults to 1.0e4.

        Returns:
            torch.Tensor: (bs, hidden_size, context_length)
        """
        x = x.transpose(1, 2)
        _, context_length, hidden_size = x.size()
        position = torch.arange(context_length).type(torch.float32)
        num_timescales = hidden_size / 2
        log_timescale_increment = (
            torch.log(torch.tensor(denominator_power)) / (num_timescales - 1)
        )
        inv_timescales = torch.exp(
            -1
            * torch.arange(num_timescales).type(torch.float32)
            * log_timescale_increment
        )
        scaled_time = torch.matmul(position.unsqueeze(1), inv_timescales.unsqueeze(0))
        positional_encoding = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        ).unsqueeze(0)
        return (x + positional_encoding.to(x.device)).transpose(1, 2)

    def forward(self, x, mask, layer_number, number_of_blocks):
        total_number_of_layers = (self.number_of_convolutions + 1) * number_of_blocks
        out = self.add_positional_encoding(x)

        for i, conv in enumerate(self.encoding_convs):
            res = out
            out = self.encoding_layer_norms[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out)
            out = self.make_layer_dropout(
                out, res, self.drop_prob * float(layer_number) / total_number_of_layers
            )
            layer_number += 1

        res = out
        out = self.layer_norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.self_att(out, mask)
        out = self.make_layer_dropout(
            out, res, self.drop_prob * float(layer_number) / total_number_of_layers
        )
        layer_number += 1
        res = out

        out = self.layer_norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        for network in self.feed_forward_networks:
            out = network(out)
        out = self.make_layer_dropout(
            out, res, self.drop_prob * float(layer_number) / total_number_of_layers
        )
        return out

    def make_layer_dropout(self, inputs, residual, drop_prob):
        if self.training:
            active = torch.empty(1).uniform_(0, 1) < drop_prob
            if active:
                return residual
            else:
                return F.dropout(inputs, drop_prob, training=self.training) + residual
        else:
            return inputs + residual


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = Conv1dNonlin(2 * hidden_size, 1, use_non_lin=False, bias=False)
        self.w2 = Conv1dNonlin(2 * hidden_size, 1, use_non_lin=False, bias=False)

    def forward(self, output_1, output_2, output_3, mask):
        logits_1 = self.w1(torch.cat([output_1, output_2], dim=1))
        logits_2 = self.w2(torch.cat([output_1, output_3], dim=1))
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

        c_emb = self.emb(cw_idxs, cc_idxs).transpose(1, 2)
        q_emb = self.emb(qw_idxs, qc_idxs).transpose(1, 2)

        c_enc = self.encoder(c_emb, c_mask, 1, 1)
        q_enc = self.encoder(q_emb, q_mask, 1, 1)

        out = self.cq_attention(
            c_enc.transpose(1, 2), q_enc.transpose(1, 2), c_mask, q_mask
        ).transpose(1, 2)

        out = F.dropout(self.cq_resizer(out), p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)
        output_1 = out

        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)

        output_2 = out
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.encoder_blocks):
            out = blk(out, c_mask, i * (2 + 2) + 1, 7)
        output_3 = out

        log_p1, log_p2 = self.decoder(output_1, output_2, output_3, c_mask)
        return log_p1, log_p2
