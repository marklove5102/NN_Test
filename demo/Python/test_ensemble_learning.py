import colorama
import ast
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/158318385

def str2tuple(value):
	if not isinstance(value, tuple):
		value = ast.literal_eval(value) # str to tuple
	return value

class EnsembleModel(nn.Module):
	def __init__(self, regression_model, model_names, device, means, stds, strategy=0, weights=None):
		super().__init__()

		self.device = device
		self.model_names = tuple(s.strip() for s in model_names.strip("()").split(","))
		self.strategy = strategy

		self.weights = weights
		if self.weights is not None:
			self.weights = np.array(str2tuple(self.weights), dtype=np.float32)
			if len(self.weights) != len(self.model_names):
				raise ValueError(colorama.Fore.RED + f"weights length mismatch: {len(self.weights)}")

		means_ = str2tuple(means)
		stds_ = str2tuple(stds)
		if len(means_) % 3 != 0 or len(means_) != 3 * len(self.model_names):
			raise ValueError(colorama.Fore.RED + f"mean length mismatch: {len(means_)}")
		if len(stds_) % 3 != 0 or len(stds_) != 3 * len(self.model_names):
			raise ValueError(colorama.Fore.RED + f"std length mismatch: {len(stds_)}")

		self.means = [means_[i:i+3] for i in range(0, len(means_), 3)]
		self.stds = [stds_[i:i+3] for i in range(0, len(stds_), 3)]

		self.models = []
		for model_name in self.model_names:
			model = regression_model
			model.load_state_dict(torch.load(model_name))
			model.to(self.device)
			model.eval()
			self.models.append(model)

	def forward(self, x):
		preds = []
		with torch.no_grad():
			for idx, model in enumerate(self.models):
				self.preprocess = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=self.means[idx], std=self.stds[idx]) # RGB
				])

				input_tensor = self.preprocess(x) # (c,h,w)
				input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model, (1,c,h,w)
				input_batch = input_batch.to(self.device)

				output = model(input_batch)
				preds.append(output[0,0].item())

		if self.strategy == 0: # simple average
			final_pred = sum(preds) / len(preds)
		elif self.strategy == 1: # weighted average
			if self.weights is None:
				raise ValueError(colorama.Fore.RED + f"weights cannot be None")

			self.weights = self.weights / sum(self.weights)
			final_pred = sum(w * p for w, p in zip(self.weights, preds))
		elif self.strategy == 2: # median
			final_pred = np.median(preds)
		elif self.strategy == 3: # trimmed mean
			preds = np.asarray(preds)
			preds = np.sort(preds)
			preds = preds[1 : len(preds) - 1]
			final_pred = preds.mean()
		else:
			raise ValueError(colorama.Fore.RED + f"unsupported ensemble learning strategies: {self.strategy}")

		return final_pred

class RegressMetrics:
	@staticmethod
	def convert(y_true, y_pred):
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)

		if len(y_true) != len(y_pred):
			raise ValueError(colorama.Fore.RED + f"inconsistent length: {len(y_true)}, {len(y_pred)}")
		return y_true, y_pred

	@staticmethod
	def mae(y_true, y_pred): # Mean Absolute Error
		y_true, y_pred = RegressMetrics.convert(y_true, y_pred)
		return float(np.mean(np.abs(y_true - y_pred)))

	@staticmethod
	def mse(y_true, y_pred): # Mean Squared Error
		y_true, y_pred = RegressMetrics.convert(y_true, y_pred)
		return float(np.mean((y_true - y_pred) ** 2))

	@staticmethod
	def rmse(y_true, y_pred): # Root Mean Squared Error
		return float(np.sqrt(RegressMetrics.mse(y_true, y_pred)))

	@staticmethod
	def r2(y_true, y_pred): # R-squared/Coefficient of Determination
		y_true, y_pred = RegressMetrics.convert(y_true, y_pred)

		ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
		ss_residual = np.sum((y_true - y_pred) ** 2)

		if ss_total == 0:
			if ss_residual == 0:
				return 1.0
			else:
				return 0.0

		return float(1.0 - (ss_residual / ss_total))

	@staticmethod
	def metrics(y_true, y_pred):
		result = {}

		result["MAE"] = float(f"{RegressMetrics.mae(y_true, y_pred):.4f}")
		result["MSE"] = float(f"{RegressMetrics.mse(y_true, y_pred):.4f}")
		result["RMSE"] = float(f"{RegressMetrics.rmse(y_true, y_pred):.4f}")
		result["R2"] = float(f"{RegressMetrics.r2(y_true, y_pred):.4f}")
		return result

