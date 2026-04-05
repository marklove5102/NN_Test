#include <openvino/openvino.hpp>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Blog: https://blog.csdn.net/fengbingchun/article/details/159862438

namespace {

constexpr char model_path[]{ "../../../data/densenet121.xml" };
constexpr char image_name[]{ "../../../data/images/hen.webp" };
constexpr char device_name[]{ "CPU" }; // CPU, GPU
constexpr int input_width{ 224 }, input_height{ 224 };
constexpr float imagenet_mean[3] = { 0.485f, 0.456f, 0.406f };
constexpr float imagenet_std[3] = { 0.229f, 0.224f, 0.225f };

} // namespace

int test_openvino_classify()
{
	ov::Core core{};

	std::cout << "available devices: ";
	for (const auto& dev: core.get_available_devices())
		std::cout << dev << " ";
	std::cout << std::endl;

	try {
		auto model = core.read_model(model_path);
		auto compiled_model = core.compile_model(model, device_name);

		auto img = cv::imread(image_name);
		if (img.empty()) {
			std::cerr << "Error: unable to read image: " << image_name << std::endl;
			return -1;
		}

		auto blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false); // no use letterbox
		float* data = reinterpret_cast<float*>(blob.data);

		int channel_size = input_width * input_height;
		for (int c = 0; c < 3; ++c) {
			float* ptr = data + c * channel_size;

			for (int i = 0; i < channel_size; ++i)
				ptr[i] = (ptr[i] - imagenet_mean[c]) / imagenet_std[c];
		}

		ov::Tensor input_tensor(ov::element::f32, { 1, 3, input_height, input_width }, data);
		auto infer_request = compiled_model.create_infer_request();
		infer_request.set_input_tensor(input_tensor);
		infer_request.infer();

		auto output = infer_request.get_output_tensor();
		const float* out_data = output.data<const float>();
		int num_classes = output.get_shape()[1];
		int class_id = 0;
		float max_val = out_data[0];

		for (int i = 1; i < num_classes; ++i) {
			if (out_data[i] > max_val) {
				max_val = out_data[i];
				class_id = i;
			}
		}

		// softmax
		std::vector<float> probs(num_classes);
		float max_logit = *std::max_element(out_data, out_data + num_classes);

		float sum = 0.f;
		for (int i = 0; i < num_classes; ++i) {
			probs[i] = std::exp(out_data[i] - max_logit);
			sum += probs[i];
		}

		for (int i = 0; i < num_classes; ++i) {
			probs[i] /= sum;
		}

		std::cout << "classes num: " << num_classes << ", current image class id: " << class_id << ", score: " << probs[class_id] << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}

	return 0;
}

