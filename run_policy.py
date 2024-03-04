from utils.utils import *
from utils.model_utils import *
import utils.visualize as viz

from models.picktoplace import PickToPlaceModel
from models.pickandplace import PickAndPlaceModel

import torch
import collections

import copy
import random
import torch.nn as nn
import argparse
from tqdm import tqdm
import wandb
import jsonlines
import json
import os

from dependencies.panda_folding.panda_fold_env import PandaFoldEnv as Env


def random_policy(color):
  # Pick random point in mask
  mask = color[:,:,0] > 0
  inds = np.transpose(np.nonzero(mask))
  pick = inds[np.random.randint(0, len(inds))][::-1]

  # Place random point in image 64*64
  place = np.random.randint(0, 64, 2)
  return pick, place


def human_policy():
  input("Perform action and press enter to continue")
  return np.array([0,0]), np.array([0,0])


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", type=str, required=False, help="weights filename")
  parser.add_argument("--use_depth", type=bool, default=True, help="use depth for obs")
  parser.add_argument("--method", type=str, default="pick_to_place", help="architecture")
  parser.add_argument("--num_episodes", type=int, default=5, help="number of episodes to run")
  parser.add_argument("--num_steps", type=int, default=100, help="number of steps per episode")
  parser.add_argument("--task_name", type=str, default="crumpled_0", help="task name")
  return parser.parse_args()


def norm_depth(depth, mask, depth_min=2800, depth_max=3400):
  depth = (depth-depth_min)/(depth_max-depth_min)
  depth = 1 - depth
  depth[mask == False]=0
  print("after norm")
  print(np.min(depth))
  print(np.max(depth))
  return depth


if __name__ == "__main__":
  args = parse_args()

  print("Task Selected: " + args.task_name)

  # Load policy
  channels = 1 if args.use_depth else 3
  if args.method == "pick_to_place":
    print("Pick Conditioned Placing Model")
    model = PickToPlaceModel(channels, 64).cuda()
  elif args.method == "pick_and_place":
    print("Pick and Place Model")
    model = PickAndPlaceModel(channels, 64).cuda()

  if args.method == "pick_to_place" or args.method == "pick_and_place":
    model.load_state_dict(torch.load(args.weights))

  env = Env()

  path = "./eval/{}_{}".format(args.method, args.task_name)
  # Run episodes
  for ep_num in range(args.num_episodes):
    input("Reset cloth to task and press enter to continue")
    # Create directory
    episode_path = path+"/episode_{}".format(ep_num)
    if not os.path.exists(episode_path): 
      os.makedirs(episode_path)
      print("Created directories.")

    # Reset environment
    env.reset()

    # Run episode
    for ts in range(args.num_steps+1):
      # Get observation
      img, depth, ir, _ = env.get_image()
      mask = ir > 20
      color = copy.deepcopy(img)
      color[mask == False] = (0, 0, 0)
      color = cv2.resize(color, (64,64))/255.0
      depth = norm_depth(depth, mask)
      depth = cv2.resize(depth, (64,64))

      # Observations to tensors
      if args.use_depth:
        obs = torch.FloatTensor(depth).unsqueeze(0).cuda()
      else:
        obs = torch.FloatTensor(color).permute(2,0,1).cuda()

      # Get action
      if args.method == "pick_to_place" or args.method == "pick_and_place":
        pick, place, pick_map, place_map = model.get_pick_place(obs)

        # Get visualizations
        viz_img = viz.get_viz_img(pick, pick, pick, place, color, color, pick_map, place_map)
        cv2.imshow("viz", viz_img)
        cv2.waitKey(1)

      elif args.method == "random":
        pick, place = random_policy(color)
      elif args.method == "human":
        pick, place = human_policy()

      # Format action for env
      pick = pick * 300/64
      place = place * 300/64
      pick = pick[::-1]
      place = place[::-1]
      # pick = [20,20]
      # place = [150,150]

      # cv2.circle(img, (int(pick[1]), int(pick[0])), 5, (0,200,0), -1)
      # cv2.circle(img, (int(place[1]), int(place[0])), 5, (0,200,0), -1)
      # cv2.imshow("img", img)
      # cv2.waitKey(0)

      # Take action
      if ts < args.num_steps:
        if args.method != "human":
          env.perform_fold(pick, place)

      # Save data
      cv2.imwrite(episode_path + "/{}_rgb_obs.png".format(ts), color*255)
      cv2.imwrite(episode_path + "/{}_rgb_full.png".format(ts), img)
      cv2.imwrite(episode_path + "/{}_ir.png".format(ts), ir)
      if args.method == "pick_to_place" or args.method == "pick_and_place":
        cv2.imwrite(episode_path + "/{}_viz.png".format(ts), viz_img*255)
      np.save(episode_path + "/{}_depth.npy".format(ts), depth)
      action_data = {"pick": list(pick), "place": list(place)}
      with open("{}/{}_action.json".format(episode_path, ts), 'w', encoding='utf-8') as f:
        json.dump(action_data, f, ensure_ascii=False, indent=4)



