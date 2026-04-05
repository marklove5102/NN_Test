import colorama
import argparse
import torch
import torchvision
import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/159862438

def parse_args():
	parser = argparse.ArgumentParser(description="model convert: pytorch tor openvino")
	parser.add_argument("--task", required=True, type=str, choices=["convert", "predict"], help="specify what kind of task")
	parser.add_argument("--openvino_model_name", type=str, help="openvino model file, for example: result/densenet121.xml")
	parser.add_argument("--device_name", type=str, choices=["CPU", "GPU", "AUTO"], default="CPU", help="device name")
	parser.add_argument("--image_name", type=str, default="", help="test image")

	args = parser.parse_args()
	return args

def convert(openvino_model_name):
	model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
	model.eval()

	ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 224, 224))
	ov_model.reshape({ov_model.input(0): [1, 3, 224, 224]}) # fixed input shape, static rather than dynamic
	ov.save_model(ov_model, openvino_model_name, compress_to_fp16=False)

def _letterbox(img, imgsz):
	shape = img.shape[:2] # current shape: [height, width, channel]
	new_shape = [imgsz, imgsz]

	# scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

	# compute padding
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
	dw /= 2 # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad: # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # add border

	return img, left, top, r

def _preprocess(img, input_shape):
	_, _, h, _ = input_shape

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img, x_offset, y_offset, r = _letterbox(img, imgsz=h)

	img = img.astype(np.float32) / 255.0
	mean = np.array([0.485, 0.456, 0.406])
	std  = np.array([0.229, 0.224, 0.225])
	img = (img - mean) / std

	img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
	img = np.expand_dims(img, axis=0) # NCHW

	return img, x_offset, y_offset, r

def _softmax(x):
	x = x - np.max(x)
	exp_x = np.exp(x)
	return exp_x / np.sum(exp_x)

def predict(model_name, device_name, image_name):
	if model_name is None or not model_name or not Path(model_name).is_file():
		raise ValueError(colorama.Fore.RED + f"{model_name} is not a model file")
	if image_name is None or not image_name or not Path(image_name).is_file():
		raise ValueError(colorama.Fore.RED + f"{image_name} is not a image file")

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"image not found: {image_name}")

	core = ov.Core()
	model = core.read_model(model=model_name)
	compiled_model = core.compile_model(model=model, device_name=device_name)

	input_layer = compiled_model.input(0)
	input_shape = input_layer.shape
	output_layer = compiled_model.output(0)
	output_shape = output_layer.shape
	print(f"input shape: {input_shape}; output shape: {output_shape}")

	input_tensor, x_offset, y_offset, r = _preprocess(img, input_shape)

	result = compiled_model([input_tensor])[output_layer]

	probs = _softmax(result[0])
	class_id = int(np.argmax(probs))
	score = float(probs[class_id])
	print(f"class: {class_id}, score: {score:.6f}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "convert":
		convert(args.openvino_model_name)
	elif args.task == "predict":
		predict(args.openvino_model_name, args.device_name, args.image_name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
