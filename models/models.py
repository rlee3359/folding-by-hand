import torch
from torch import nn
import torch.nn.functional as F

#------------------------------------------------------------------------------
# Network Class
#------------------------------------------------------------------------------

# Goal conditioned pick place model
class PickPlaceModel(nn.Module):
    def __init__(self, input_channels ):
        super(PickPlaceModel, self).__init__()
        self.net = UNet(input_channels, 1, bilinear=True)

    def forward(self, obs, goal, pick_map):
        x = torch.cat([obs, goal, pick_map], dim=1)
        return self.net(x)

    def get_loss(self, batch):
        place_pred = self(batch['obs'], batch['goal'], batch['pick'])
        place_pred = place_pred
        places = batch["place"]

        loss = torch.mean(F.binary_cross_entropy_with_logits(place_pred, places, reduction="none"))
        return loss

#------------------------------------------------------------------------------

# Goal conditioned pick place model
class PickPlaceKeypointsModel(nn.Module):
    def __init__(self, input_channels ):
        super(PickPlaceModel, self).__init__()
        self.net = UNet(input_channels, 2, bilinear=True)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        return self.net(x)

#------------------------------------------------------------------------------

class PickThenPlaceModel(nn.Module):
    def __init__(self, input_channels ):
        super(PickPlaceModel, self).__init__()
        self.pick_net = UNet(input_channels, 1, bilinear=True)
        self.place_net = UNet(input_channels+1, 1, bilinear=True)

    def get_pick_map(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        pick_map = self.pick_net(x)
        return pick_map

    def get_place_map(self, obs, goal, pick_map):
        x = torch.cat([obs, goal, pick_map], dim=1)
        place_map = self.place_net(x)
        return pick_map, place_map

