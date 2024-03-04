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
import jsonlines

#------------------------------------------------------------------------------

def sample_batch(dataset, batch_size, args, true_sample=False, fold_only=False):
  raw_batch = random.sample(dataset, batch_size)
  # batch = collections.defaultdict(list)
  batch = []

  for transition in raw_batch:
    if fold_only:
      # If place not in obs mask, skip
      mask = get_cloth_mask(transition['obs']['rgb'])
      if mask[transition['place'][::-1]] == 0: continue

    batch_elem = {}
    # Store the original images for viz later
    batch_elem['obs_rgb'] = transition['obs']['rgb']/255.0
    batch_elem['nobs_rgb'] = transition['nobs']['rgb']/255.0


    # Handle depth or rgb observations, convert to torch tensors
    if args.use_depth:
      obs = transition['obs']['depth']
      obs = torch.FloatTensor(obs).cuda()
      obs = obs.unsqueeze(2)
    else:
      obs = transition['obs']['rgb']/255.0
      obs = torch.FloatTensor(obs).cuda()
    obs  = obs.permute(2, 0, 1)

    # Half the time, the sample is a fake negative pick example to balance the positives
    negative_pick_sample = bool(np.random.randint(0, 2)) if args.architecture == "pick_to_place" and not true_sample else False
    if negative_pick_sample:
      pick = get_random_pick(transition['obs']['rgb'])
      place_map = np.zeros((args.image_width,args.image_width))
    else:
      pick = transition['pick']
      place_map = gaussian_heatmap(transition['place'], args.image_width, args.heatmap_sigma)
    pick_map = gaussian_heatmap(pick, args.image_width, args.heatmap_sigma)

    pick_map  = torch.FloatTensor(pick_map).cuda() .unsqueeze(0)
    place_map = torch.FloatTensor(place_map).cuda().unsqueeze(0)

    # Add to batch
    batch_elem['obs'] = obs
    batch_elem['pick'] = pick_map
    batch_elem['place'] = place_map
    # Original pick and place index
    batch_elem['place_index'] = transition['place']
    batch_elem['pick_index'] = transition['pick']
    batch.append(batch_elem)

  return batch

#------------------------------------------------------------------------------

def get_random_pick(img):
  mask = get_cloth_mask(img)
  indices = get_indices_from_mask(mask)
  pick = choose_random_index(indices)[::-1]
  return pick

#------------------------------------------------------------------------------

# Get batch loss for training
def get_error(batch, args):
  if args.viz: imgs = []
  errors = []
  for batch_elem in batch:
    pred_pick, pred_place, pick_map, place_map = model.get_pick_place(batch_elem['obs'])
    if args.viz:
      viz_img = viz.get_viz_img(batch_elem['pick_index'], batch_elem['place_index'], pred_pick, pred_place, batch_elem['obs_rgb'], batch_elem['nobs_rgb'], pick_map, place_map)
      imgs.append(viz_img)
    pred = np.array([pred_pick, pred_place])/args.image_width
    true = np.array([batch_elem['pick_index'], batch_elem['place_index']])/args.image_width

    error = np.mean((true - pred)**2)
    errors.append(error)
  mean_error = np.mean(errors)

  # Images for visualization
  img = None
  if args.viz:
    img = viz.viz_images(imgs, "viz")
  return mean_error, img

#------------------------------------------------------------------------------

def batch_to_tensors(batch):
  # Convert list of dicts to dict of lists
  batch = {k: [d[k] for d in batch] for k in batch[0]}
  
  # Convert to torch tensors
  for key in batch:
    # Don't convert the data for visualization
    if key == "obs_rgb" or key == "nobs_rgb" or key == "pick_index" or key == "place_index":
      continue
    batch[key] = torch.stack(batch[key])
  return batch

#------------------------------------------------------------------------------

def train(dataset, args):
  # Sample batch and get loss
  batch = sample_batch(dataset, args.batch_size, args)
  batch_tensors = batch_to_tensors(batch)

  loss = model.get_loss(batch_tensors)

  # Perform training step
  opt.zero_grad()
  loss.backward()
  opt.step()

  # Log
  if args.wandb:
    wandb.log({"train/loss": loss.item()})

#------------------------------------------------------------------------------

def run_validation(dataset, args):
  global lowest_error
  # Sample validation batch and get loss
  # Loss is BCE for heatmaps,
  # Error is mean squared error for actual pick and place
  batch = sample_batch(dataset, args.batch_size, args)
  batch_tensors = batch_to_tensors(batch)
  loss = model.get_loss(batch_tensors)

  batch = sample_batch(dataset, args.batch_size, args, true_sample=True, fold_only=False)
  error, img = get_error(batch, args)
  # Visualize
  img = img[:,:,::-1]

  # Log
  if args.wandb:
    img = wandb.Image(img, caption="Test Set Visualization")
    wandb.log({"val/val_loss": loss.item(), "val/val_error": error, "val/viz": img})

  # Use error on validation batch to decide when to save model
  if error < lowest_error:
    lowest_error = error

    # Save model
    if args.save_model:
      filename = "{}{}".format(args.architecture, train_step)
      torch.save(model.state_dict(), "./weights/{}_{}_{}.pt".format(args.name, wandb.run.id, filename))
      with jsonlines.open("./weights/{}_{}_{}.json".format(args.name, wandb.run.id, filename), mode='w') as writer:
        writer.write(vars(args))
        writer.write({"lowest_error": lowest_error})
        writer.write({"train_step": train_step})
    # Log the lowest error metrics/viz
    if args.wandb:
      wandb.log({"best_model/viz": img})
      wandb.log({"best_model/loss": loss, "best_model/error": error})

#------------------------------------------------------------------------------

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, required=True, help="Name of task")
  parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset")
  parser.add_argument("--val_dataset", type=str, required=True, help="Path to val dataset")
  parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
  parser.add_argument("--val_interval", type=int, default=500, help="Run validation every n steps")
  parser.add_argument("--training_steps", type=int, default=int(500000), help="Number of training steps")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
  parser.add_argument("--use_depth", type=bool, default=True, help="Use depth for observations")
  parser.add_argument("--image_width", type=int, default=64, help="Image width")
  parser.add_argument("--architecture", type=str, default="pick_to_place", help="Architecture to use for model. Options: pick_to_place, pick_then_place, pick_and_place")
  parser.add_argument("--one_step_model", type=bool, default=False, help="Train inverse model instead of multi-step goal model")
  parser.add_argument("--save_model", type=bool, default=True, help="Save model")
  parser.add_argument("--heatmap_sigma", type=int, default=3, help="Sigma for gaussian heatmaps")
  parser.add_argument("--wandb", type=bool, default=True, help="Use wandb for logging")
  parser.add_argument("--viz", type=bool, default=True, help="Visualize training")
  return parser.parse_args()

#------------------------------------------------------------------------------

if __name__ == "__main__":
  args = parse_args()

  train_dataset = torch.load(args.train_dataset)
  print("Training architecture: {}".format(args.architecture))
  print("for {} steps".format(args.training_steps))
  print("Training dataset size: {}".format(len(train_dataset)))
  val_dataset = torch.load(args.val_dataset)

  if args.wandb:
    import wandb
    wandb.init(project="folding_by_hand_{}".format(args.name), config=vars(args))
    wandb.config.update(args)
    wandb.run.name = "{}_{}_{}".format(args.architecture, wandb.run.id, len(train_dataset))
  # Depth stats - save for loading at test time
  # depth_normalizer = DepthNorm(train_dataset)
  # with jsonlines.open("./data/depth_min_max.json", mode='w') as writer:
  #   print(depth_normalizer.min_depth, depth_normalizer.max_depth)
  #   writer.write({"min_depth": int(depth_normalizer.min_depth), "max_depth": int(depth_normalizer.max_depth)})

  # Input is obs, pick_map 
  image_channels = 1 if args.use_depth else 3
  if args.architecture == "pick_to_place":
    model = PickToPlaceModel(image_channels, args.image_width).cuda()
  elif args.architecture == "pick_and_place":
    model = PickAndPlaceModel(image_channels, args.image_width).cuda()

  opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  lossfn = nn.BCELoss()

  lowest_error = np.inf # Starting point for lowest error
  for train_step in tqdm(range(args.training_steps)):
    train(train_dataset, args)
    if train_step % args.val_interval == 0 and train_step > 0:
      with torch.no_grad():
        run_validation(val_dataset, args)
