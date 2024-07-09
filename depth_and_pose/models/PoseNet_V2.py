import torch
import torch.nn as nn
from .resnet_encoder import *


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.conv_squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)

        self.convs_pose = []
        self.convs_pose.append(
            nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))
        self.convs_pose.append(nn.Conv2d(256, 256, 3, stride, 1))
        self.convs_pose.append(
            nn.Conv2d(256, 6 * num_frames_to_predict_for, 1))

        self.relu = nn.ReLU()

        self.convs_pose = nn.ModuleList(list(self.convs_pose))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.conv_squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        pose = 0.01 * out.view(-1, 6)

        return pose


class PoseNet_V2(nn.Module):

    def __init__(self, num_layers=18, pretrained=True,model_version = 'L'):
        super(PoseNet_V2, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)
        self.model_version = model_version
    def add_intrinsics_head(self, bottleneck, image_height, image_width):
        # Since the focal lengths in pixels tend to be in the order of magnitude of
        # the image width and height, we multiply the network prediction by them.

        focal_lengths = nn.Conv2d(
            in_channels=bottleneck.size(1),
            out_channels=2,
            kernel_size=(1,1),device=bottleneck.device
        )(bottleneck).softmax(dim=1).squeeze(2).squeeze(2) * torch.tensor(
            [[image_width, image_height]], dtype=torch.float32, device=bottleneck.device)
        #print('focal_lengths',focal_lengths.size())
        #* torch.tensor(
         #   [[image_width, image_height]], dtype=torch.float32)

        # The pixel offsets tend to be around the center of the image, and they
        # are typically a fraction the image width and height in pixels. We thus
        # multiply the network prediction by the width and height, and the
        # additional 0.5 them by default at the center of the image.
        offsets = nn.Conv2d(
            in_channels=bottleneck.size(1),
            out_channels=2,
            kernel_size=(1,1), device=bottleneck.device
        )(bottleneck).squeeze(2).squeeze(2) + 0.5
        offsets *= torch.tensor([[image_width, 
                                  image_height]], dtype=torch.float32, device=bottleneck.device)
       # print('offsets',offsets.size())
        foci = torch.diag_embed(focal_lengths)

        intrinsic_mat = torch.cat([foci, offsets.unsqueeze(-1)], dim=2)
        batch_size = bottleneck.size(0)
        last_row = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32,device=intrinsic_mat.device
                                ).expand(batch_size, -1, -1)
        intrinsic_mat = torch.cat([intrinsic_mat, last_row], dim=1)
        return intrinsic_mat
    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)
        features = self.encoder(x)
        #print('botteleneck',features[-1].size())
        bottleneck = torch.mean(features[-1], dim=(1,2),keepdims=True)
        bottleneck = bottleneck.permute(0,3,2,1)
        K = self.add_intrinsics_head(bottleneck, img1.size(2), img1.size(3))
       #print('model_version',self.model_version)
        if self.model_version != 'L':
            pose = self.decoder([features])  
            return pose, K
        else:
            return K


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseNet_V2().cuda()
    model.eval()

    tgt_img = torch.randn(4, 3, 256, 832).cuda()

    pose = model(tgt_img, tgt_img)

    print(pose.size())
