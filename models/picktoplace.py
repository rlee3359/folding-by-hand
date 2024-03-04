import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import cv2

from utils import utils
from .net import FCN

#------------------------------------------------------------------------------
# Network Class
#------------------------------------------------------------------------------

# pick place model
class PickToPlaceModel(nn.Module):
    def __init__(self, image_channels, image_width):
        super(PickToPlaceModel, self).__init__()

        input_channels = image_channels + 1 
        self.image_width = image_width
        self.net = FCN(input_channels, 1, bilinear=True)

    #--------------------------------------------------------------------------

    def forward(self, obs, pick_map):
        x = torch.cat([obs, pick_map], dim=1)
        return self.net(x)

    #--------------------------------------------------------------------------

    def get_loss(self, batch):
        place_pred = self(batch['obs'], batch['pick'])
        places = batch["place"]

        loss = torch.mean(F.binary_cross_entropy_with_logits(place_pred, places, reduction="none"))
        return loss

    #--------------------------------------------------------------------------

    def get_pick_place(self, obs):
        obs_rgb = obs.permute(1,2,0).detach().cpu().numpy() # Kinda hacky...
        pick_gauss, pick_indices = get_pickmaps_from_obs(obs_rgb, self.image_width)

        
        place_maps = []
        i = 0
        batch_size = 1000
        while i < len(pick_gauss):
            batch = pick_gauss[i:i+batch_size]
            torch.cuda.empty_cache()
            place_map = torch.sigmoid(self._get_place_maps(obs, batch)).squeeze(0).detach().cpu().numpy()
            print("place_map", place_map.shape)
            if len(batch) == 1:
                place_map = np.expand_dims(place_map, axis=0)
            place_maps.append(place_map)

            # print(len(place_maps))
            i += batch_size
        place_maps = np.concatenate(place_maps, axis=0)


        # place_maps = []
        # for pick in pick_gauss:
        #     print(i)
        #     pick = np.expand_dims(pick, axis=0)
        #     place_map = torch.sigmoid(self._get_place_maps(obs, pick)).squeeze(0).detach().cpu().numpy()
        #     place_maps.append(place_map)
        #     i += 1
        # place_maps = np.stack(place_maps)
        # print("place_maps", place_maps.shape)
        # for i in range(len(place_maps)):
        #     print(np.max(place_maps[i]))
        #     print(np.min(fast_place_maps[i]))
        #     cv2.imshow("place_map", place_maps[i][0])
        #     cv2.imshow("fast_place_map", fast_place_maps[i][0])
        #     cv2.waitKey(0)

        pick_map, best_pick_num = self._get_pick_map(place_maps, pick_indices)
        pred_pick, pred_place = self._get_pick_place_inds_from_maps(pick_map, place_maps, best_pick_num)
        best_place_map = place_maps[best_pick_num].squeeze(0)#.detach().cpu().numpy()
        return pred_pick, pred_place, pick_map, best_place_map

    #--------------------------------------------------------------------------

    def _get_pick_place_inds_from_maps(self, pick_map, place_maps, best_pick_num):
        # Unravel pick location
        pick = np.array(np.unravel_index(np.argmax(pick_map), pick_map.shape))
        pick = pick[::-1]

        # Unravel place location
        # place_map = place_maps[best_pick_num].squeeze(0).detach().cpu().numpy()
        place_map = place_maps[best_pick_num].squeeze(0)
        place = np.array(np.unravel_index(np.argmax(place_map), place_map.shape))
        place = place[::-1]
        return pick, place

    #--------------------------------------------------------------------------

    def _get_place_maps(self, obs, pick_gauss):
        obs = obs.repeat(len(pick_gauss),1,1,1)#.permute(0,3,1,2)
        picks = torch.FloatTensor(np.array(pick_gauss)).cuda().unsqueeze(1)

        place_maps = self(obs, picks)
        return place_maps
    
    #--------------------------------------------------------------------------

    def _get_pick_map(self, place_maps, pick_indices):
      pickmap = np.zeros((self.image_width,self.image_width))
      maxes = []
      for ind, place_maps in zip(pick_indices, place_maps):
        place_maps = place_maps#.detach().cpu().numpy()
        q_max = np.max(place_maps)
        maxes.append(q_max)
        pickmap[ind[0],ind[1]] = q_max
        # # draw circle at pick location
        # circle_im = np.zeros((self.image_width,self.image_width))
        # cv2.circle(circle_im, tuple(ind[::-1]), 5, (1.0,1.0,1.0), -1)
        # circle_im = circle_im*q_max
        # pickmap = pickmap + circle_im
        # if debug:
        #   cv2.imshow("circle", circle_im)
        #   cv2.imshow("pickmap", pickmap)
        #   cv2.waitKey(0)

      best_pick_num = np.argmax(maxes)
      return pickmap, best_pick_num


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

