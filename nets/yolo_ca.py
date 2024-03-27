import torch
import torch.nn as nn
import sys
import numpy as np
# sys.path.append('..')
from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()

        #20*20*1024 -> 
        c_ = int(2 * c2 * e) # hidden channels
        # c1 == 1024, c_ == 512, c2 == 512 
        # self.m = nn.MaxPool2d(kernel_size=k, stride=k)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        
        # self.mp  = nn.ModuleList(
        #     [nn.MaxPool2d(k[i], stride=1, padding=i // 2) for i in range(len(k)) ]
        # )
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(c_ * 4, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        # print(f"The x1 shape is {x1.shape}.")
        # y1 = self.cv6(self.cv5(torch.cat([x1] + [self.mp[i] for i in self.mp], 1)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        out = self.cv7(torch.cat([y1] + [y2], 1))

        # x_1 = self.cv1(x)
        # x_2 = self.cv2(x)
        # x_3 = self.cv3(x_1)
        # x_4 = self.cv4(x_2)
        # x_all = [x_4]
        # for i in range(len(self.mp)):
        #     x_4 = self.mp[i](x_4)
        #     x_all.append(x_4)
        # x_5 = self.cv5(torch.cat([x.all[id] for id in range(len(self.mp)+1)], 1))
        # x_6 = self.cv6(x_5)
        # out = self.cv7(torch.cat([x_2, x_6], 1))
        return out
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11  = autopad(k, p) - k // 2
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std     = (bn.running_var + bn.eps).sqrt()
        bias    = bn.bias - bn.running_mean * bn.weight / std

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        
        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None
      
class CA_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_block, self).__init__()
        self.conv_hw      = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False)
        self.conv_split_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False)
        self.conv_split_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False)

        self.BN           = nn.BatchNorm2d(channel // reduction)
        self.sigmoid      = nn.Sigmoid()
        self.relu         = nn.ReLU()

    def forward(self, x):
        # x -> 1*512*40*40
        _, _, h, w = x.size()
        # H_avgpool = nn.AvgPool2d(kernel_size=3)[2]
        # W_avgpool = nn.AvgPool2d(kernel_size=3)[3]
        # 1*512*40*40 -> 1*512*1*40
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # 1*512*40*40 -> 1*512*1*40
        x_w = torch.mean(x, dim=2, keepdim=True)
        # 1*512*1*40 cat 1*512*1*40 -> 1*512*1*80
        cat_h_and_w = torch.cat((x_h, x_w), 3)
        
        conv_h_w    = self.conv_hw(cat_h_and_w)
        bn          = self.BN(conv_h_w) 
        relu        = self.relu(bn)

        x_cat_conv_split_h, x_cat_conv_split_w = relu.split([h, w], 3)

        sigmoid_h = self.sigmoid(self.conv_split_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        sigmoid_w = self.sigmoid(self.conv_split_h(x_cat_conv_split_w))

        out = x * sigmoid_h.expand_as(x) * sigmoid_w.expand_as(x)
        return out

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()

        transition_channels = {'l' : 32, 'x' :40}[phi]
        block_channels      = 32
        panet_channels      = {'l' : 32, 'x' : 64}[phi]

        e    = {'l' : 2, 'x' : 1}[phi]
        n    = {'l' : 4, 'x' : 6}[phi]
        ids  = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]
        conv = {'l' : RepConv, 'x' : Conv}[phi]

        #-----------------------------------------------#

        #-----------------------------------------------#
        self.backbone           = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        # transition_channels = 32 when using size 'l' model
        self.upsample           = nn.Upsample(scale_factor=2, mode='nearest')

        # sppcspc : 20*20*1024 -> 20*20*512
        self.sppcspc            = SPPCSPC(transition_channels * 32, block_channels * 16)

        # feat1 : 80*80*512 -> 80*80*128
        self.feat1_conv         = Conv(transition_channels * 16, block_channels * 4) 
        # feat2 : 40*40*1024 -> 40*40*256
        self.feat2_conv         = Conv(transition_channels * 32, block_channels * 8) 


        #conv after sppcspc : 20*20*512 -> 20*20*256
        self.sppcspc_conv       = Conv(transition_channels * 16, block_channels * 8)
        #p4_conv : 40*40*256 -> 40*40*128
        self.p4_conv            = Conv(transition_channels * 8, block_channels * 4)        

        #-----------------------------------------------#
        #   ELAN block
        #-----------------------------------------------#
        # 40*40*512 -> 40*40*256
        self.P4_multi_concat_block = Multi_Concat_Block(transition_channels * 16, transition_channels * 4, block_channels * 8, n=n, e=e, ids=ids)
        # 80*80*256 -> 80*80*128
        self.P3_multi_concat_block = Multi_Concat_Block(transition_channels * 8, transition_channels * 2, block_channels * 4, n=n, e=e, ids=ids)
        # 40*40*512 -> 40*40*256
        self.P2_multi_concat_block = Multi_Concat_Block(transition_channels * 16, transition_channels * 4, block_channels * 8, n=n, e=e, ids=ids)
        # 20*20*1024 -> 20*20*512
        self.P1_multi_concat_block = Multi_Concat_Block(transition_channels * 32, transition_channels * 8, block_channels * 16, n=n, e=e, ids=ids)
        
        #-----------------------------------------------#
        #   MP-2 block
        #-----------------------------------------------#
        # 80*80*128 -> (40*40*128 cat 40*40*128) cat 40*40*256 -> 40*40*512
        self.P2_transition_block = Transition_Block(transition_channels * 4, block_channels * 4)
        # 40*40*256 -> (20*20*256 cat 20*20*256) cat 20*20*512 -> 20*20*1024
        self.P1_transition_block = Transition_Block(transition_channels * 8, block_channels * 8)

        #-----------------------------------------------#
        #   RepConv block
        #-----------------------------------------------#
        # 20*20*512 -> 20*20*1024
        self.P1_RepConv_block = conv(transition_channels * 16, block_channels * 32, 3, 1)
        # 40*40*256 -> 40*40*512
        self.P2_RepConv_block = conv(transition_channels * 8, block_channels * 16, 3, 1)
        # 80*80*128 -> 80*80*256
        self.P3_RepConv_block = conv(transition_channels * 4, block_channels * 8, 3, 1)

        #-----------------------------------------------#
        #   YOLO Head
        #  * In VOC2007, num_classes = 20
        #  * 1 + 4 + num_classes = 25
        #-----------------------------------------------#
        # 20*20*1024 -> 20*20*3 * 25
        self.P1_yolo_head     = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
        # 40*40*512 -> 40*40*3 * 25
        self.P2_yolo_head     = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 80*80*256 -> 80*80*3 * 25
        self.P3_yolo_head     = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)

        # input_shape == output_shape
        self.P1_yolo_head_ca  = CA_block(len(anchors_mask[0]) * (5 + num_classes))
        self.P2_yolo_head_ca  = CA_block(len(anchors_mask[0]) * (5 + num_classes))
        self.P3_yolo_head_ca  = CA_block(len(anchors_mask[0]) * (5 + num_classes))

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
    # YOLO Head need inputs: feat1, feat2, feat3. From BackBone.
        feat1, feat2, feat3   = self.backbone.forward(x)   

        #-----------------------------------------------#
        #   For P5
        #-----------------------------------------------#
        # sppcspc : 20*20*1024 -> 20*20*512
        P5                    = self.sppcspc(feat3)
        # feat3 : 20*20*512 -> 20*20*256
        P5_Conv               = self.sppcspc_conv(P5)
        # P5_Upsample : 20*20*256 -> 40*40*256
        P5_Upsample           = self.upsample(P5_Conv)
        # P5_out : 40*40*256 -> 40*40*512
        P5_out                = torch.cat([self.feat2_conv(feat2), P5_Upsample], 1)
        # ELAN : 40*40*512 -> 40*40*256

        #-----------------------------------------------#
        #   For P4
        #-----------------------------------------------#
        P4_Multi_Concat_Block = self.P4_multi_concat_block(P5_out)
        P4_Conv               = self.p4_conv(P4_Multi_Concat_Block)
        p4_Upsample           = self.upsample(P4_Conv)
        P4_out                = torch.cat([self.feat1_conv(feat1), p4_Upsample], 1)

        #-----------------------------------------------#
        #   For P3
        #-----------------------------------------------#
        # 80*80*256 -> 80*80*128
        P3_out                = self.P3_multi_concat_block(P4_out)
        #-----------------------------------------------#
        #   For P2
        #-----------------------------------------------#
        # 80*80*128 -> (40*40*128 cat 40*40*128) cat 40*40*256 -> 40*40*512
        P2                    = torch.cat([self.P2_transition_block(P3_out), P4_Multi_Concat_Block], 1)
        # 40*40*512 -> 40*40*256
        P2_out                = self.P2_multi_concat_block(P2)
        #-----------------------------------------------#
        #   For P1
        #-----------------------------------------------#
        # 40*40*256 -> (20*20*256 cat 20*20*256) cat 20*20*512 -> 20*20*1024
        P1                    = torch.cat([self.P1_transition_block(P2_out), P5], 1)
        # 20*20*1024 -> 20*20*512
        P1_out                = self.P1_multi_concat_block(P1)

        out1                  = self.P1_RepConv_block(P1_out)
        out2                  = self.P2_RepConv_block(P2_out)
        out3                  = self.P3_RepConv_block(P3_out)

        #-----------------------------------------------#
        #   YOLO Head
        #  * In VOC2007, num_classes = 20
        #  * 1 + 4 + num_classes = 25
        #-----------------------------------------------#
        # 20*20*1024 -> 20*20*3 * 25
        out0                  = self.P1_yolo_head(out1)
        # 40*40*512 -> 40*40*3 * 25
        out1                  = self.P2_yolo_head(out2)
        # 80*80*256 -> 80*80*3 * 25
        out2                  = self.P3_yolo_head(out3)

        out0                  = self.P1_yolo_head_ca(out0)
        out1                  = self.P2_yolo_head_ca(out1)
        out2                 = self.P3_yolo_head_ca(out2)

        return [out0, out1, out2]

        



# def main():
#     # sppcspc = SPPCSPC.
#     img_tensor = torch.randn(1, 1024, 20, 20)
#     print(f"The image2tensor shape is {img_tensor.shape}.")
#     c1 = img_tensor.shape[1]
#     c2 = c1 // 2
#     sppcspc = SPPCSPC(c1, c2)
#     output = sppcspc(img_tensor)
#     print(f"The SPPCSPC shape is {output.shape}.")

# if __name__ == '__main__' :
#     main()





    