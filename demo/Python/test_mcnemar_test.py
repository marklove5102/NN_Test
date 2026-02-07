import argparse
import colorama
import csv
from scipy.stats import chi2, binomtest
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/157844198

def parse_args():
	parser = argparse.ArgumentParser(description="mcnemar test")
	parser.add_argument("--src_file", required=True, type=str, help="src file name")
	parser.add_argument("--src_file2", required=True, type=str, help="src file name")
	parser.add_argument("--threshold", type=float, default=0.5, help="error margin")

	args = parser.parse_args()
	return args

def mcnemar_test(src_file, src_file2, threshold):
	if src_file is None or not src_file or not Path(src_file).is_file():
		raise ValueError(colorama.Fore.RED + f"{src_file} is not a file")
	if src_file2 is None or not src_file2 or not Path(src_file2).is_file():
		raise ValueError(colorama.Fore.RED + f"{src_file2} is not a file")

	def parse_csv(file):
		with open(file, "r", encoding="utf-8") as f:
			reader = csv.reader(f)
			all_rows = list(reader)
			data = all_rows[1:-1] # remove the first and last rows
		return data

	data1 = parse_csv(src_file)
	data2 = parse_csv(src_file2)
	if len(data1) != len(data2):
		raise ValueError(colorama.Fore.RED + f"length mismath: {src_file}:{len(data1)}, {src_file2}:{len(data2)}")
	print(f"number of data rows: {len(data1)}")

	is_same = all(row1[0] == row2[0] for row1, row2 in zip(data1, data2))
	if not is_same:
		raise ValueError(colorama.Fore.RED + f"image name mismatch: {src_file}, {src_file2}")

	n11 = 0; n10 = 0; n01 = 0; n00 = 0
	for i in range(len(data1)):
		value1 = abs(float(data1[i][1]) - float(data1[i][2]))
		value2 = abs(float(data2[i][1]) - float(data2[i][2]))

		if value1 <= threshold and value2 <= threshold:
			n11 += 1
		elif value1 > threshold and value2 > threshold:
			n00 += 1
		elif value1 <= threshold and value2 > threshold:
			n10 += 1
		elif value1 > threshold and value2 <= threshold:
			n01 += 1
		else:
			raise ValueError(colorama.Fore.RED + f"unsupported conditions: value: {value1}, {value2}")
	print(f"n11: {n11}; n10: {n10}; n01: {n01}; n00: {n00}")

	if n10 + n01 == 0:
		print(colorama.Fore.YELLOW + "unable to test differences")
		return

	def calculate_pvalue(n10, n01, method): # method: 0:Yates; 1:original; 2:exact binomial test
		if method == 0:
			stat = (abs(n01 - n10) - 1) ** 2 / (n10 + n01)
			return chi2.sf(stat, df=1)
		elif method == 1:
			stat = (n01 - n10) ** 2 / (n10 + n01)
			return chi2.sf(stat, df=1)
		else:
			return binomtest(k=min(n10, n01), n=n10+n01, p=0.5, alternative="two-sided").pvalue

	if n10 + n01 >= 25:
		pvalue = calculate_pvalue(n10, n01, 0)
	elif 10 <= n10 + n01 < 25:
		pvalue = calculate_pvalue(n10, n01, 1)
	else:
		pvalue = calculate_pvalue(n10, n01, 2)

	if pvalue < 0.05:
		print(colorama.Fore.GREEN + f"pvalue: {pvalue:.4f}, the two models show a significant difference")
	else:
		print(colorama.Fore.YELLOW + f"pvalue: {pvalue:.4f}, the two models no not show a significant difference")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	mcnemar_test(args.src_file, args.src_file2, args.threshold)

	print(colorama.Fore.GREEN + "====== execution completed ======")
