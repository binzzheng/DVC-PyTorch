import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *

#import torchac

from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

def save_model(model, iter, config):
    if not os.path.isdir("./snap/{}".format(config)):
        os.mkdir("./snap/{}".format(config))

    torch.save(model.state_dict(), "./snap/{}/iter{}.model".format(config, iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0

class VideoCompressor(CompressionModel):
    def __init__(self, **kwargs):
        super().__init__(entropy_bottleneck_channels=out_channel_N, **kwargs)
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.mvDecoder = Synthesis_mv_net()
        self.mvpriorEncoder = Analysis_mvprior_net()
        self.mvpriorDecoder = Synthesis_mvprior_net()
        self.warpnet = Warp_net()

        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.warp_weight = 0
        
        self.entropy_hyper_mv = self.entropy_bottleneck
        self.entropy_hyper_res = EntropyBottleneck(out_channel_N)
        self.entropy_bottleneck_mv = GaussianConditional(None)
        self.entropy_bottleneck_res = GaussianConditional(None)

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe):
        estmv = self.opticFlow(input_image, referframe)
        mv_fea = self.mvEncoder(estmv)

        mv_prior = self.mvpriorEncoder(mv_fea)
        quant_mvprior, mvprior_likelihoods = self.entropy_hyper_mv(mv_prior)
        recon_mv_sigma = self.mvpriorDecoder(quant_mvprior)

        quant_mv = self.entropy_bottleneck_mv.quantize(
                mv_fea, "noise" if self.training else "dequantize")
        _, mv_likelihoods = self.entropy_bottleneck_mv(mv_fea, recon_mv_sigma)
        recon_mv = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, recon_mv)
        
        res = input_image - prediction
        res_fea = self.resEncoder(res)

        batch_size = res_fea.size()[0]

        res_prior = self.respriorEncoder(res_fea)
        quant_resprior, resprior_likelihoods = self.entropy_hyper_res(res_prior)
        recon_res_sigma = self.respriorDecoder(quant_resprior)

        quant_res = self.entropy_bottleneck_res.quantize(
            res_fea, "noise" if self.training else "dequantize")
        _, res_likelihoods = self.entropy_bottleneck_res(res_fea, recon_res_sigma)

        recon_res = self.resDecoder(quant_res)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)

        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))
        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))

        im_shape = input_image.size()

        bpp_mv = torch.log(mv_likelihoods).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_mvprior = torch.log(mvprior_likelihoods).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_res = torch.log(res_likelihoods).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_resprior = torch.log(resprior_likelihoods).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_mv + bpp_mvprior + bpp_res + bpp_resprior

        return clipped_recon_image, mse_loss, warploss, interloss, bpp_res, bpp_resprior, bpp_mv, bpp
     
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_hyper_res,
            "entropy_hyper_res",
            ["_quantized_cdf","_offset","_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck_mv,
            "entropy_bottleneck_mv",
            ["_quantized_cdf","_offset","_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck_res,
            "entropy_bottleneck_res",
            ["_quantized_cdf","_offset","_cdf_length","scale_table"],
            state_dict,
        )
        print('finish loading entropy botteleneck buffer')

        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):

        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64

        def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
            return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.entropy_bottleneck_mv.update_scale_table(scale_table, force=force)
        updated = self.entropy_bottleneck_res.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, input_image, referframe):     
        estmv = self.opticFlow(input_image, referframe)
        mv_fea = self.mvEncoder(estmv)

        mv_prior = self.mvpriorEncoder(mv_fea)
        mvprior_strings = self.entropy_hyper_mv.compress(mv_prior)
        quant_mvprior = self.entropy_hyper_mv.decompress(mvprior_strings, mv_prior.size()[-2:])
        recon_mv_sigma = self.mvpriorDecoder(quant_mvprior)
        mv_indexes = self.entropy_bottleneck_mv.build_indexes(recon_mv_sigma)
        mv_strings = self.entropy_bottleneck_mv.compress(mv_fea, mv_indexes)
        quant_mv = self.entropy_bottleneck_mv.decompress(mv_strings, mv_indexes)
        recon_mv = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, recon_mv)

        res = input_image - prediction
        res_fea = self.resEncoder(res)
        batch_size = res_fea.size()[0]

        res_prior = self.respriorEncoder(res_fea)
        resprior_strings = self.entropy_hyper_res.compress(res_prior)
        quant_resprior = self.entropy_hyper_res.decompress(resprior_strings, res_prior.size()[-2:])
        
        recon_res_sigma = self.respriorDecoder(quant_resprior)
        res_indexes = self.entropy_bottleneck_res.build_indexes(recon_res_sigma)
        res_strings = self.entropy_bottleneck_res.compress(res_fea, res_indexes)

        return {"strings": [mv_strings, mvprior_strings, res_strings, resprior_strings], "shape": [mv_prior.size()[-2:], res_prior.size()[-2:]]}

    def decompress(self, referframe, strings, shape):
        mvprior_hat = self.entropy_hyper_mv.decompress(strings[1], shape[0])
        recon_mv_sigma = self.mvpriorDecoder(mvprior_hat)
        mv_indexes = self.entropy_bottleneck_mv.build_indexes(recon_mv_sigma)
        mv_hat = self.entropy_bottleneck_mv.decompress(strings[0], mv_indexes)
        recon_mv = self.mvDecoder(mv_hat)
        prediction, _ = self.motioncompensation(referframe, recon_mv)
        
        resprior_hat = self.entropy_hyper_res.decompress(strings[3], shape[1])
        recon_res_sigma = self.respriorDecoder(resprior_hat)
        res_indexes = self.entropy_bottleneck_res.build_indexes(recon_res_sigma)
        res_hat = self.entropy_bottleneck_res.decompress(strings[2], res_indexes)
        recon_res = self.resDecoder(res_hat)
        
        recon_frame = prediction + recon_res
        recon_frame = recon_frame.clamp(0., 1.)
        return {"x_hat": recon_frame}
