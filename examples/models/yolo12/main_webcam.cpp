#include "inference.h"
#include <thread>
#include <future>

#include <gflags/gflags.h>

void draw_detection(
    cv::Mat& frame,
    const Detection detection,
    const cv::Scalar color);

DetectionConfig DEFAULT_YOLO_CONFIG = {
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"},
    0.45,
    0.50};

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(input_path, "input.mp4", "Path to the mp4 input video");

DEFINE_string(output_path, "output.mp4", "Path to the mp4 output video");

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Use Mmap model to enable loading of big YOLO models in OpenVINO
  Module yolo_module(FLAGS_model_path, Module::LoadMode::Mmap);

  auto error = yolo_module.load();

  ET_CHECK_MSG(
      error == Error::Ok,
      "Loading of the model failed with status 0x%" PRIx32,
      (uint32_t)error);
  error = yolo_module.load_forward();
  ET_CHECK_MSG(
      error == Error::Ok,
      "Loading of the forward method failed with status 0x%" PRIx32,
      (uint32_t)error);

  const auto model_input_shape =
      yolo_module.method_meta("forward")->input_tensor_meta(0)->sizes();
  std::cout << "Model input shape: [";
  for (auto& dim : model_input_shape) {
    std::cout << dim << ", ";
  }
  std::cout << "]" << std::endl;
  const cv::Size img_dims = {model_input_shape[3], model_input_shape[2]};

  cv::namedWindow("Video Detection", cv::WND_PROP_FULLSCREEN);
  cv::setWindowProperty("Video Detection", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  //cv::VideoCapture cap(FLAGS_input_path.c_str());
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }
  const auto frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  const auto frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  const auto video_lenght = cap.get(cv::CAP_PROP_FRAME_COUNT);
  std::cout << "Input video shape: [3, " << frame_width << ", " << frame_height
            << ", ]" << std::endl;

  cv::VideoWriter video(
      FLAGS_output_path.c_str(),
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      30,
      cv::Size(frame_width, frame_height));

  std::cout << "Start the detection..." << std::endl;
  et_timestamp_t time_spent_executing = 0;
  unsigned long long iters = 0;
  // Show progress every 10%
  unsigned long long progress_bar_tick = std::round(video_lenght / 10);

  struct frame_ctx {
    cv::Mat frame;
    cv::Mat scaled_input;
    cv::Mat blob;
    int pad_x;
    int pad_y;
    float scale;
  };
  std::queue<frame_ctx*> ready_q;
  std::queue<std::pair<frame_ctx*, std::future<cv::Mat>>> scale_q;
  std::queue<std::pair<frame_ctx*, std::future<std::shared_ptr<executorch::aten::Tensor>>>> input_q;
  std::queue<std::pair<frame_ctx*, std::future<executorch::aten::Tensor>>> execute_q;
  std::queue<std::pair<frame_ctx*, std::future<std::vector<Detection>>>> output_q;
  const et_timestamp_t before_execute = et_pal_current_ticks();
  size_t frame_queue_size = 2;
  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty() && ready_q.empty() && scale_q.empty() && input_q.empty() && execute_q.empty() && output_q.empty())
      break;

    if (!frame.empty() && ready_q.size() < 1) {
      frame_ctx *new_frame_ctx = new frame_ctx;
      new_frame_ctx->frame = frame;
      ready_q.push(new_frame_ctx);
    }

    while (!ready_q.empty() && scale_q.size() < frame_queue_size) {
      frame_ctx *scale_f = ready_q.front();
      scale_q.push(std::make_pair(scale_f, std::async(std::launch::async, scale_with_padding, std::ref(scale_f->frame), &(scale_f->pad_x), &(scale_f->pad_y), &(scale_f->scale), img_dims)));
      ready_q.pop();
    }
    while (!scale_q.empty() && input_q.size() < frame_queue_size) {
      auto status = scale_q.front().second.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::ready) {
          scale_q.front().first->scaled_input = scale_q.front().second.get();
          input_q.push(std::make_pair(scale_q.front().first, std::async(std::launch::async, prepare_input, std::ref(scale_q.front().first->scaled_input), std::ref(scale_q.front().first->blob), img_dims)));
          scale_q.pop();
      } else {
          break;
      }
    }
    while (!input_q.empty() && execute_q.size() < frame_queue_size) {
      auto status = input_q.front().second.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::ready) {
          std::shared_ptr<executorch::aten::Tensor> prepared_input = input_q.front().second.get();
          execute_q.push(std::make_pair(input_q.front().first, std::async(std::launch::async, execute_frame, std::ref(yolo_module), prepared_input)));
          input_q.pop();
      } else {
          break;
      }
    }
    while (!execute_q.empty() && output_q.size() < frame_queue_size) {
      auto status = execute_q.front().second.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::ready) {
          executorch::aten::Tensor raw_output = execute_q.front().second.get();
          output_q.push(std::make_pair(execute_q.front().first, std::async(std::launch::async, process_output, std::ref(raw_output), DEFAULT_YOLO_CONFIG, execute_q.front().first->pad_x, execute_q.front().first->pad_y, execute_q.front().first->scale)));
          execute_q.pop();
      } else {
          break;
      }
    }
    while (!output_q.empty()) {
      auto status = output_q.front().second.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::ready) {
          std::vector<Detection> output = output_q.front().second.get();
          for (auto& detection : output) {
            draw_detection(output_q.front().first->frame, detection, cv::Scalar(0, 0, 255));
          }
          iters++;

          //if (!(iters % progress_bar_tick)) {
          //  const int precent_ready = (100 * iters) / video_lenght;
          //  std::cout << iters << " out of " << video_lenght
          //            << " frames are are processed (" << precent_ready << "\%)"
          //            << std::endl;
          //}
          //video.write(output_q.front().first->frame);
	  cv::imshow("Video Detection", output_q.front().first->frame);
          output_q.pop();
      } else {
          break;
      }
    }
    if (cv::waitKey(10) == 'q') {
        break;
    }
  }
  const et_timestamp_t after_execute = et_pal_current_ticks();
  time_spent_executing = after_execute - before_execute;

  const auto tick_ratio = et_pal_ticks_to_ns_multiplier();
  constexpr auto NANOSECONDS_PER_MILLISECOND = 1000000;

  double elapsed_ms = static_cast<double>(time_spent_executing) *
      tick_ratio.numerator / tick_ratio.denominator /
      NANOSECONDS_PER_MILLISECOND;
  std::cout << "Model executed successfully " << iters << " times in "
            << elapsed_ms << " ms." << std::endl;
  std::cout << "Average detection time: " << elapsed_ms / iters << " ms."
            << std::endl;
  cap.release();
  video.release();
}

void draw_detection(
    cv::Mat& frame,
    const Detection detection,
    const cv::Scalar color) {
  cv::Rect box = detection.box;

  // Detection box
  cv::rectangle(frame, box, color, 2);

  // Detection box text
  std::string classString = detection.className + ' ' +
      std::to_string(detection.confidence).substr(0, 4);
  cv::Size textSize =
      cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
  cv::Rect textBox(
      box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

  cv::rectangle(frame, textBox, color, cv::FILLED);
  cv::putText(
      frame,
      classString,
      cv::Point(box.x + 5, box.y - 10),
      cv::FONT_HERSHEY_DUPLEX,
      1,
      cv::Scalar(0, 0, 0),
      2,
      0);
}
