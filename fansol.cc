/*******************************************************************************
 * TinyML 风扇异常检测项目
 * 阶段二: 设备端实时推理
 * * 工作流程:
 * 1. 以约 119Hz 的频率从加速度计读取 X,Y,Z 轴数据。
 * 2. 将数据存入一个40个样本的缓冲区。
 * 3. 缓冲区满后，对这40个样本进行与Python中完全相同的预处理：
 * a. 标准化 (Scaling)
 * b. PCA 降维
 * 4. 将处理后的一维数据送入 TensorFlow Lite 模型进行推理。
 * 5. 计算模型的输入与输出之间的重建误差 (MAE)。
 * 6. 如果误差超过预设阈值，则判定为异常。
 *******************************************************************************/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 包含您从 Colab 生成的模型文件。
// 您的Python脚本使用'xxd -i autoencoder.tflite > model.h'
// 这会自动生成一个名为 'autoencoder_tflite' 的C数组。
#include "model.h"

// --- 全局常量和变量 ---

// 1. 神经网络配置
const int WINDOW_SIZE = 40;        // 必须与您训练时使用的窗口大小一致
const int TENSOR_ARENA_SIZE = 100 * 1024; // 为模型分配4KB内存，对于这个模型应该足够
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// TFLite Micro 的核心组件
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// 2. 数据采集和处理的缓冲区
float data_window[WINDOW_SIZE]; // 存储一个窗口降维后的数据
int data_index = 0;             // 当前窗口已采集的数据点数量

// =================================================================================
//  ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 关键参数区域 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//  请将您从 Google Colab 脚本最后一步打印出的参数，完整地复制并替换到这里！
// =================================================================================

// 3. 数据预处理参数 (从 Colab 脚本中获取)
const float scaler_mean[3] = { -0.008062962009488434f, -0.020294950535103527f, 1.0164002795042475f };
const float scaler_scale[3] = { 0.0077848401950793835f, 0.037119756206916155f, 0.004799918469097551f };
const float pca_components[3] = { 0.2717441475008236f, 0.766907456863307f, 0.5813846152991198f };

// 4. 异常检测阈值 (这是一个需要通过实验微调的值)
//const float ANOMALY_THRESHOLD = 1.212650f;; // <-- 这是一个初始猜测值，您需要根据实际情况调整

// 阈值1：用于区分“正常”和“卡住”。(正常 < 1.2, 卡住 > 4.0)
const float THRESHOLD_NORMAL_VS_STUCK = 3.0f;
//
// 阈值2：用于区分“卡住”和“倾斜”。(卡住 < 25.0, 倾斜 > 49.0)
const float THRESHOLD_STUCK_VS_TILTED = 35.0f;

// =================================================================================
//  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 关键参数区域 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
// =================================================================================


void setup() {
  Serial.begin(115200);
  while (!Serial); // 等待串口连接

  // 初始化板载IMU传感器
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  // --- 初始化 TensorFlow Lite Micro ---
  
  // 1. 加载模型
  const tflite::Model* model = tflite::GetModel(autoencoder_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 2. 创建操作解析器 (OpResolver)
  // ==================== 最终修改：将 <3> 改回 <4> ======================
  static tflite::MicroMutableOpResolver<6> op_resolver; 
  // =====================================================================
  
  op_resolver.AddFullyConnected();
  op_resolver.AddReshape();
  op_resolver.AddDequantize();

  // ==================== 最终修改：添加 SHAPE 操作 ======================
  op_resolver.AddShape(); // 解决 "Failed to get registration from op code SHAPE" 错误
  op_resolver.AddStridedSlice(); // 添加 StridedSlice 操作
  op_resolver.AddPack();
  // =====================================================================

  // 3. 实例化模型解释器
  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  // 4. 为模型的输入输出张量分配内存
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // 5. 获取模型输入和输出张量的指针
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  
  Serial.println("Arduino Nano 33 BLE is ready.");
  Serial.println("Starting anomaly detection...");
}

void loop() {
  // 检查是否有新的加速度数据
  if (IMU.accelerationAvailable()) {
    float ax, ay, az;
    IMU.readAcceleration(ax, ay, az); // 读取3轴原始数据

    // --- 步骤A: 在设备上实时进行数据预处理 ---
    // 这个过程必须与您在Python中训练时使用的预处理完全一致

    // 1. 标准化 (Scaling)
    float scaled_data[3];
    scaled_data[0] = (ax - scaler_mean[0]) / scaler_scale[0];
    scaled_data[1] = (ay - scaler_mean[1]) / scaler_scale[1];
    scaled_data[2] = (az - scaler_mean[2]) / scaler_scale[2];

    // 2. PCA降维 (投影到第一主成分)
    float pca_value = scaled_data[0] * pca_components[0] + 
                      scaled_data[1] * pca_components[1] + 
                      scaled_data[2] * pca_components[2];

    // --- 步骤B: 收集数据到窗口 ---
    data_window[data_index++] = pca_value;

    // --- 步骤C: 当窗口集满40个数据点时，进行一次模型推理 ---
    if (data_index >= WINDOW_SIZE) {
      
      // 1. 将窗口中的数据加载到模型的输入张量中
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        model_input->data.f[i] = data_window[i];
      }
      
      // 2. 运行模型推理
      if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed!");
        return;
      }
      
      // 3. 计算重建误差 (与 Colab 脚本一致，使用平均绝对误差 MAE)
      float reconstruction_error = 0;
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        float diff = model_output->data.f[i] - data_window[i];
        reconstruction_error += abs(diff);
      }
      reconstruction_error /= WINDOW_SIZE;
      
      // 4. 根据重建误差进行分类判断
      Serial.print("Reconstruction Error: ");
      Serial.print(reconstruction_error, 6); // 打印6位小数

      // 判断条件1：如果误差低于第一个阈值，则为“正常”
      if (reconstruction_error < THRESHOLD_NORMAL_VS_STUCK) {
        Serial.println("  -> Status: Normal");
      }
      // 判断条件2：如果误差低于第二个阈值 (但高于第一个)，则为“卡住”
      else if (reconstruction_error < THRESHOLD_STUCK_VS_TILTED) {
        Serial.println("  -> Status: Stuck");
      }
      // 判断条件3：如果误差高于所有阈值，则为“倾斜”
      else {
        Serial.println("  -> Status: Tilted");
      }

      // 5. 重置窗口索引，准备下一次采集
      data_index = 0;
    }
  }
  // 稍微延迟以匹配大约104Hz的采样率 (1000ms / 104Hz ≈ 9.6ms)
  delay(9);
}