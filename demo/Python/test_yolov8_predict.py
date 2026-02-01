import colorama
import argparse
from ultralytics import YOLO
import os
import torch
import cv2

import numpy as np
np.bool = np.bool_ # Fix Error: AttributeError: module 'numpy' has no attribute 'bool'. OR: downgrade numpy: pip unistall numpy; pip install numpy==1.23.1

# Blog:
# 	https://blog.csdn.net/fengbingchun/article/details/139377787
#	https://blog.csdn.net/fengbingchun/article/details/140691177
#	https://blog.csdn.net/fengbingchun/article/details/141931184
#	https://blog.csdn.net/fengbingchun/article/details/157615429

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8/YOLO11 predict")
	parser.add_argument("--model", required=True, type=str, help="model file")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment", "classify", "obb"], help="specify what kind of task")
	parser.add_argument("--dir_images", type=str, default="", help="directory of test images")
	parser.add_argument("--verbose", action="store_true", help="whether to output detailed information")
	parser.add_argument("--dir_result", type=str, default="", help="directory where the image or video results are saved")

	args = parser.parse_args()
	return args

def get_images(dir):
	# supported image formats
	img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
	images = []

	for file in os.listdir(dir):
		if os.path.isfile(os.path.join(dir, file)):
			# print(file)
			_, extension = os.path.splitext(file)
			for format in img_formats:
				if format == extension.lower():
					images.append(file)
					break

	return images

def predict(task, model, verbose, dir_images, dir_result):
	model = YOLO(model) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	if task == "detect" or task =="segment" or task == "obb":
		os.makedirs(dir_result, exist_ok=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	images = get_images(dir_images)

	for image in images:
		results = model.predict(dir_images+"/"+image, verbose=verbose, device=device)

		if task == "detect" or task =="segment" or task == "obb":
			for result in results:
				print("result:", result)
				result.save(dir_result+"/"+image)
		else:
			print(f"class names:{results[0].names}: top5: {results[0].probs.top5}; conf:{results[0].probs.top5conf}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()
	if args.dir_images == "":
		raise ValueError(colorama.Fore.RED + f"dir_images cannot be empty:{args.dir_images}")

	print("Running on GPU") if torch.cuda.is_available() else print("Running on CPU")

	predict(args.task, args.model, args.verbose, args.dir_images, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
