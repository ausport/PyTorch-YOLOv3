import os
import glob
import subprocess
import csv
import sys

"""
Train on Custom Dataset

Custom model

Run the commands below to create a custom model definition, replacing <num-classes> with the number of classes in your dataset.

$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
Classes

Add class names to data/custom/classes.names. This file should have one row per class name.

Image Folder

Move the images of your dataset to data/custom/images/.

Annotation Folder

Move your annotations to data/custom/labels/. The dataloader expects that the annotation file corresponding to the image data/custom/images/train.jpg has the path data/custom/labels/train.txt. Each row in the annotation file should define one bounding box, using the syntax label_idx x_center y_center width height. The coordinates should be scaled [0, 1], and the label_idx should be zero-indexed and correspond to the row number of the class name in data/custom/classes.names.

Define Train and Validation Sets

In data/custom/train.txt and data/custom/valid.txt, add paths to images that will be used as train and validation data respectively.

Train

To train on the custom dataset run:

$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
Add --pretrained_weights weights/darknet53.conv.74 to train using a backend pretrained on ImageNet.


"""
video_files = "/home/stuart/Dropbox/_Microwork/30sec_detection"
annotation_files = "/home/stuart/Dropbox/_Microwork/Annotations"


partition = {'train': ['1_hockey_1', '3_hockey_3', '4_hockey_4', '5_hockey_5', '6_hockey_6', '7_hockey_7', '8_hockey_8',
                       '11_hockey_11', '12_hockey_12', '13_hockey_13', '14_netball_1', '15_netball_2', '16_netball_3', '17_netball_4',
                       '18_netball_5', '19_netball_6', '20_netball_7', '21_netball_8', '22_netball_9', '23_netball_10',
                       '24_netball_11', '26_netball_13', '27_netball_14', '28_netball_15', '31_netball_18',
                       '33_afl_2', '34_afl_3', '36_afl_4', '37_afl_5', '39_hockey_14', '41_tennis_2', '42_tennis_3', '43_cricket_1',
                       '47_bball_2', '48_rugby_2', '49_bball_3', '50_bball_4', '51_bball_5', '52_afl_6', '53_afl_7', '54_tennis_4',
                       '55_bball_6', '56_bball_7', '58_soccer_2', '59_soccer_3', '60_rugby_3', '61_rugby_4', '63_rugby_6',
                       '64_rugby_7', '65_rugby_8', '66_rugby_9', '67_rugby_10', '68_rugby_11', '70_rugby_13',
                       '71_hockey_15', '73_hockey_17', '74_hockey_18', '75_hockey_19', '76_hockey_20', '76_hockey_21',
                       '78_bball_8', '79_bball_9', '80_bball_10', '81_beach_1', '82_beach_2', '83_beach_3', '84_beach_4',
                       '87_cricket_3', '90_hockey_24', '91_hockey_25', '92_hockey_26', '94_hockey_28', '95_hockey_29', '96_hockey_30', '97_hockey_31',
                       '98_hockey_32', '99_hockey_33'],

             'test': ['0_hockey_0', '2_hockey_2', '10_hockey_10', '25_netball_12', '29_netball_16', '30_netball_17',
                      '32_afl_1', '38_bball_1', '40_tennis_1', '46_rugby_1', '57_soccer_1', '62_rugby_5', '69_rugby_12',
                      '72_hockey_16', '77_hockey_22', '88_cricket_4', '89_hockey_23', '93_hockey_27', '100_hockey_34'],

             'hold': ['9_hockey_9', '44_hockey_1', '45_hockey_1', '86_beach_6']}  # Hold out portrait samples for now.


for f in partition['train']:
	_export = "./images"
	_video = f"{video_files}/{f}.mp4"
	_thruth = f"{annotation_files}/{f}.csv"
	print(_video)
	print(_thruth)
	assert os.path.exists(_video), _video
	assert os.path.exists(_thruth), _thruth

	# Read all of the ground truth data for this file.
	# Open predictions and convert to the JSON format we've adopted.
	with open(_thruth, 'rt') as g:
		reader = csv.reader(g, delimiter=',', quotechar='"')
		sortedlist = sorted(reader, key=lambda _row: _row[4], reverse=False)
		_current_frame = 1
		_frame_labels = []
		_last_valid_frame_number = 0
		_is_new_frame_number = True
		json_frames = {"frames": [], "class": "video", "filename": _thruth}

		frame = {'timestamp': 0, 'num': 0, "class": "frame", "annotations": []}

		for row in sortedlist:
			if not row[0] == '#':
				_t = int(row[4].rsplit('.', 1)[0])
				print(_t)
				if _t > _last_valid_frame_number:
					# New frame
					_last_valid_frame_number = _t
					_is_new_frame_number = True

					if len(frame["annotations"]) > 0:
						json_frames["frames"].append(frame)

					# Create frame stub
					frame = {'timestamp': float(_t / 25.), 'num': _t, "class": "frame", "annotations": []}

				_top = int(row[9])
				_left = int(row[10])
				_width = int(row[11])
				_height = int(row[12])
				_bbox = [_left, _top, _width, _height]

				new_annotation = {
					"dco": False,		# Not required for hypotheses..
					"height": _height,
					"width": _width,
					"id": row[2],
					"y": _top - _height / 2,
					"x": _left - _width / 2
				}

				frame["annotations"].append(new_annotation)

	subprocess.call('ffmpeg -i {0} -vf fps=25 {1}/{2}_%04d.png'.format(_video, _export, f), shell=True)

	for img in glob.glob('{0}/{1}*.png'.format(_export, f)):
		print(os.path.basename(img))
		_image_path = os.path.basename(img)
		_image_name = os.path.splitext(_image_path)[0]

		with open("train.txt", "a") as myfile:
			myfile.write("data/custom/images/{0}\n".format(os.path.basename(img)))

		with open("./labels/{0}.txt".format(_image_name), "a") as myfile:
			# TODO get matching frame rows and append to text file.
			myfile.write("Hello")


	break
