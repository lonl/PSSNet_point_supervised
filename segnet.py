import torch
from torch import nn
from torch.nn import init
# from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck
from net.backbone import build_backbone
import layer.function as fun
import math


# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5 # 1
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    # print(layer.weight.data.shape, w.div(in_planes).repeat(in_planes, out_planes, 1, 1).shape)
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  if bias:
    init.constant(layer.bias, 0)
  return layer


# Returns 2D batch normalisation layer
def bn(planes):
  layer = nn.BatchNorm2d(planes)
  # Use mean 0, standard deviation 1 init
  init.constant(layer.weight, 1)
  init.constant(layer.bias, 0)
  return layer

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out



class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])

    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1) # change the stride
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


class FeatureResNet(ResNet):
  def __init__(self):
    # super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
    super().__init__(BasicBlock, [3, 4, 6, 3], 1000)

  def forward(self, x):
    x1 = self.conv1(x)
    x = self.bn1(x1)
    x = self.relu(x)
    x2 = self.maxpool(x)
    x_middle = self.layer1(x2)
    x3 = self.layer2(x_middle)
    x4 = self.layer3(x3)
    # x5 = self.layer4(x4)
    # return x_middle, x1, x2, x3
    return x_middle, x1, x2, x3, x4
    # torch.Size([2, 64, 128, 128])
    # torch.Size([2, 64, 256, 256])
    # torch.Size([2, 64, 128, 128])
    # torch.Size([2, 128, 64, 64])
    # torch.Size([2, 256, 32, 32])



class SegResNet(nn.Module):
  def __init__(self, pretrained_net):
    super().__init__()
    # self.pretrained_net = FeatureResNet()
    # old_dict = torch.load('/mnt/a409/users/lisuicheng/Machine_Learning/SceneChangeDet/code/pretrained_model/resnet101-5d3b4d8f.pth')
    # model_dict = self.pretrained_net.state_dict()
    # old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    # model_dict.update(old_dict)
    # self.pretrained_net.load_state_dict(model_dict)
    # self.pretrained_net.load_state_dict(pretrained_dict)

    # self.backbone = build_backbone('res101_atrous', os=16)  # res101_atrous     16
    # self.backbone_layers = self.backbone.get_layers()

    self.pretrained_net = pretrained_net

    self.relu = nn.ReLU(inplace=True)
    # self.conv5 = conv(512, 256, stride=2, transposed=True)
    # self.bn5 = bn(256)
    self.conv4 = conv(256, 128, stride=2, transposed=True)
    self.bn4 = bn(128)
    self.conv5 = conv(128, 64, stride=2, transposed=True)
    self.bn5 = bn(64)
    self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
    # self.conv6 = conv(64, 64, stride=2, transposed=True)
    # self.bn6 = bn(64)
    # self.conv7 = conv(64, 32, stride=2, transposed=True)
    # self.bn7 = bn(32)
    # self.conv8 = conv(32, 1, kernel_size=7)
    # init.constant(self.conv8.weight, 0)  # Zero init



  def forward(self, x):
    x_middle, x1, x2, x3, x4= self.pretrained_net(x)
    # print(x_middle.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
    x = self.relu(self.bn4(self.conv4(x4)))
    # print(x.shape)
    x = self.relu(self.bn5(self.conv5(x + x3)))
    # print(x.shape)
    result = self.upsample4(x)
    # x_ = self.relu(self.bn6(self.conv6(x + x2)))
    # # print(x_.shape)
    # x = self.relu(self.bn7(self.conv7(x_ + x1)))
    # # print(x.shape)
    # result = self.conv8(x)

    return x_middle, x4, result
    # return middle_feature, middle_feature, up_bottom_feature



class SiameseNet(nn.Module):
  def __init__(self, pretrained_net,norm_flag='l2'):
    super(SiameseNet, self).__init__()
    self.CNN = SegResNet(pretrained_net)
    if norm_flag == 'l2':
      self.norm = fun.l2normalization(scale=1)
    if norm_flag == 'exp':
      self.norm = nn.Softmax2d()


  def forward(self, t0, t1):

    out_t0_middle, out_t0_bottom, out_t0_final = self.CNN(t0)
    out_t1_middle, out_t1_bottom, out_t1_final = self.CNN(t1)
    out_t0_middle_norm, out_t1_middle_norm = self.norm(out_t0_middle), self.norm(out_t1_middle)
    out_t0_bottom_norm, out_t1_bottom_norm = self.norm(out_t0_bottom), self.norm(out_t1_bottom)
    out_t0_final_norm, out_t1_final_norm = self.norm(out_t0_final), self.norm(out_t1_final)
    # print(out_t0_middle_norm.shape, out_t1_middle_norm.shape, out_t0_bottom_norm.shape, out_t1_bottom_norm.shape,
    #       out_t0_final_norm.shape, out_t1_final_norm.shape)
    return [out_t0_middle_norm, out_t1_middle_norm], [out_t0_bottom_norm, out_t1_bottom_norm], [out_t0_final_norm,
                                                                                        out_t1_final_norm]

