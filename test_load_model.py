from utils.utils import *
from utils.model_utils import *
import utils.visualize as viz

from models.picktoplace import PickToPlaceModel
from models.pickandplace import PickAndPlaceModel

import torch
import collections

import random
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import jsonlines



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", type=str, required=True, help="weights filename")

  return parser.parse_args()


def norm_depth(depth, depth_min=2726, depth_max=3023):
  depth = (depth-depth_min)/(depth_max-depth_min)
  depth = 1 - depth
  return depth


if __name__ == "__main__":
  args = parse_args()

  # Load policy
  model = PickToPlaceModel(1, 64).cuda()
  model.load_state_dict(torch.load(args.weights))

  # Load data
  data = torch.load("./data/square_cloth_test_buf.pt")
  depth_goal  = data[0][-1]["obs"]["depth"]
  color_goal  = data[0][-1]["obs"]["rgb"]/255.0
  depth_goal  = torch.FloatTensor(depth_goal).unsqueeze(0).cuda()

  for i in range(len(data[0])):
    depth_input = data[0][i]["obs"]["depth"]
    color_input = data[0][i]["obs"]["rgb"]/255.0

    depth_input = torch.FloatTensor(depth_input).unsqueeze(0).cuda()

    # Get action
    pick, place, pick_map, place_map = model.get_pick_place(depth_input, depth_goal)
    img = viz.get_viz_img(pick, place, pick, place, color_input, color_goal, pick_map, place_map)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    

