import torch
import sys
from models import *
import argparse


"""
python3 make_weights.py --model_def config/yolov3-custom.cfg --pretrained_weights checkpoints/yolov3_sport_416.pth
"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
	parser.add_argument("--pretrained_weights", type=str, default="checkpoints/yolov3_sport_416.pth", help="if specified starts from checkpoint model")
	opt = parser.parse_args()
	print(opt)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initiate model
	model = Darknet(opt.model_def).to(device)

	# torch.load("checkpoints/yolov3_ckpt_24.pth"
	model.load_state_dict(torch.load(opt.pretrained_weights, map_location=torch.device('cpu')))
	Darknet.save_darknet_weights(model, 'yolov3_sport_416.weights', cutoff=-1)
