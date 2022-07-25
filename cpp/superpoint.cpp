#include <torch/script.h> // One-stop header.
#include<opencv2/opencv.hpp>
#include <iostream>
#include <memory>

cv::Mat vizFloat2colorMap(cv::Mat map,double min = 0, double max = 0) {

    if(min == 0 && max == 0)
        cv::minMaxIdx(map, &min, &max);
    std::cout<<"min max: "<<min<<" "<<max<<"\n";
    
    cv::Mat adjMap;
    cv::Mat falseColorsMap;
    
    // expand your range to 0..255. Similar to histEq();
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min); 
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_PARULA);
    return falseColorsMap;
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
    }

    torch::Device device = torch::kCUDA;


    torch::jit::script::Module module;
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }

    std::cout << "ok\n";

    std::vector<torch::jit::IValue> net_input;
    at::Tensor net_output; 

    cv::Mat input = cv::imread("/home/cecco/ws/rtls/tk.legacy/build/GPIO_run3/keyframes/1650532384030129920_1.png", cv::IMREAD_GRAYSCALE);

    cv::Mat float_input;
    input.convertTo(float_input, CV_32FC1, 1.0f/255.0f); 
    //std::cout<<float_input.cols<<" "<<float_input.rows<<"\n";

    net_input.resize(1);

    at::Tensor t = torch::zeros({1, 1, 800, 848});
    //memcpy(t.data_ptr<float>(), float_input.data, sizeof(float)*848*800);
    FILE *f = fopen("../../input.bin", "rb");
    fread(t.data_ptr<float>(),sizeof(float)*848*800,1,f); // read 10 bytes to our buffer


    net_input[0] = t.to(device);
    net_output = module.forward(net_input).toTensor().cpu();

    cv::Mat cvOutput(float_input.size(), CV_32FC1, net_output.data_ptr<float>());

    cv::Mat viz_in = vizFloat2colorMap(float_input);
    cv::Mat viz = vizFloat2colorMap(cvOutput);
    cv::imshow("input", viz_in);
    cv::imshow("scores", viz);
    cv::waitKey(0);

}