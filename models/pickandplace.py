import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import cv2

from utils import utils
from .unet import UNet
from .net import FCN

#------------------------------------------------------------------------------
# Network Class
#------------------------------------------------------------------------------

#pick place model
class PickAndPlaceModel(nn.Module):
    def __init__(self, image_channels, image_width):
        super(PickAndPlaceModel, self).__init__()
        input_channels = image_channels
        self.net = FCN(input_channels, 2, bilinear=True)
        self.image_width = image_width

    def forward(self, obs):
        return self.net(obs)

    #--------------------------------------------------------------------------

    def get_loss(self, batch):
        pred_heatmaps = self(batch['obs'])
        heatmaps = torch.cat([batch['pick'], batch['place']], dim=1)

        loss = torch.mean(F.binary_cross_entropy_with_logits(pred_heatmaps, heatmaps, reduction="none"))
        return loss

    #--------------------------------------------------------------------------

    def get_pick_place(self, obs):
        obs = obs.unsqueeze(0)
        # print(obs.shape)
        heatmaps = F.sigmoid(self(obs))
        pick_map = heatmaps[0,0].squeeze(0).detach().cpu().numpy()
        place_map = heatmaps[0,1].squeeze(0).detach().cpu().numpy()
        pred_pick, pred_place = self._get_pick_place_inds_from_maps(pick_map, place_map)
        return pred_pick, pred_place, pick_map, place_map

    #--------------------------------------------------------------------------

    def _get_pick_place_inds_from_maps(self, pick_map, place_map):
        # Unravel pick location
        pick_map = pick_map
        pick = np.array(np.unravel_index(np.argmax(pick_map), pick_map.shape))
        pick = pick[::-1]

        # Unravel place location
        place = np.array(np.unravel_index(np.argmax(place_map), place_map.shape))
        place = place[::-1]
        return pick, place

#------------------------------------------------------------------------------

def get_pickmaps_from_obs(obs, image_width, heatmap_sigma=2):
  mask = utils.get_cloth_mask(obs)

  # indices = get_indices_from_mask(mask>0)
  indices = get_downsampled_indices_from_mask(mask, image_width)
  possible_picks = indices

  pick_maps = []
  for index in possible_picks:
    pick_map = utils.gaussian_heatmap(index[::-1], image_width, heatmap_sigma)
    pick_maps.append(pick_map)
    # if debug:
    #   cv2.imshow("pick", pick_map)
    #   cv2.waitKey(0)

  return pick_maps, indices #, possible_dwnsc

#------------------------------------------------------------------------------

def get_downsampled_indices_from_mask(mask, image_width, downsample_scale=2):
  # Create a grid of points evenly spaced across the image, 20x20
  grid = np.mgrid[0:image_width:downsample_scale,
                  0:image_width:downsample_scale].reshape(2,-1).T
  # Reverse x and y to match the mask
  grid = grid[:, [1,0]]
  # Remove the grid points that are not in the mask
  grid = grid[mask[grid[:,0], grid[:,1]] == 1]
  return grid

