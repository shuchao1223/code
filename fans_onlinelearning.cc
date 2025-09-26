#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

// --- 全局常量和变量 ---
const int WINDOW_SIZE = 40;
const int TENSOR_ARENA_SIZE = 12 * 1024; 
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

float data_window[WINDOW_SIZE];
int window_index = 0;             

const float scaler_mean[3] = { -0.008062962009488434f, -0.020294950535103527f, 1.0164002795042475f };
const float scaler_scale[3] = { 0.0077848401950793835f, 0.037119756206916155f, 0.004799918469097551f };
const float pca_components[3] = { 0.2717441475008236f, 0.766907456863307f, 0.5813846152991198f };

float threshold_normal_vs_stuck = 3.0f; 
float threshold_stuck_vs_tilted = 35.0f;

// 用于存储学习到的输出层偏置补偿向量
float output_bias_vector[WINDOW_SIZE]; 
// 标志位，表示偏置是否已经学习完毕
bool bias_is_calibrated = false; 

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  const tflite::Model* model = tflite::GetModel(autoencoder_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<6> op_resolver; 
  op_resolver.AddFullyConnected();
  op_resolver.AddReshape();
  op_resolver.AddDequantize();
  op_resolver.AddShape();
  op_resolver.AddStridedSlice();
  op_resolver.AddPack();

  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  
  Serial.println("Arduino Nano 33 BLE is ready.");
  Serial.println("Send 'b' via Serial Monitor to start Bias Calibration.");
  Serial.println("Starting anomaly detection...");
}

// ======================================================================
// 【新增】用于打印偏置向量的辅助函数
// ======================================================================
void printBiasVector(const char* title) {
  Serial.println(title);
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    Serial.print(output_bias_vector[i], 6);
    Serial.print(" ");
    if ((i + 1) % 5 == 0) {
      Serial.println();
    }
  }
  if (WINDOW_SIZE % 5 != 0) {
    Serial.println();
  }
  Serial.println("----------------------------------------");
}

// ======================================================================
// “自适应偏置”校准函数
// ======================================================================
void runBiasCalibration() {
  Serial.println("\n--- Entering Bias Adaptation Mode ---");
  Serial.println("Please ensure the fan is running in a STABLE 'NEW NORMAL' state.");
  
  const int CALIBRATION_SAMPLES = 50;
  int samples_collected = 0;
  
  // 1. 清空上一次的偏置和累加器
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    output_bias_vector[i] = 0.0f;
  }
  bias_is_calibrated = false;

  Serial.println("Collecting data...");
  while (samples_collected < CALIBRATION_SAMPLES) {
    if (IMU.accelerationAvailable()) {
      float ax, ay, az;
      IMU.readAcceleration(ax, ay, az);
      
      float scaled_data[3] = { (ax - scaler_mean[0]) / scaler_scale[0], (ay - scaler_mean[1]) / scaler_scale[1], (az - scaler_mean[2]) / scaler_scale[2] };
      float pca_value = scaled_data[0] * pca_components[0] + scaled_data[1] * pca_components[1] + scaled_data[2] * pca_components[2];
      
      data_window[window_index++] = pca_value;
      if (window_index < WINDOW_SIZE) continue;
      window_index = 0;

      for (int i = 0; i < WINDOW_SIZE; ++i) model_input->data.f[i] = data_window[i];
      if (interpreter->Invoke() != kTfLiteOk) continue;
      
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        float error_in_this_dimension = data_window[i] - model_output->data.f[i];
        output_bias_vector[i] += error_in_this_dimension;
      }
      
      samples_collected++;
      Serial.print("Calibration Sample ");
      Serial.println(samples_collected);
    }
  }

  // 4. 计算平均误差向量
  Serial.println("Calculating average error vector...");
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    output_bias_vector[i] /= samples_collected;
  }

  bias_is_calibrated = true;

  // 【修改】在校准完成后，打印学习到的新偏置值
  printBiasVector("--- Bias Vector AFTER Calibration ---");

  Serial.println("--- Bias Adaptation Complete! ---");
  Serial.println("Resuming Standard Monitoring with corrected output...");
  delay(1000);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'b') {
      // 【修改】在开始校准前，打印当前的偏置值
      printBiasVector("--- Bias Vector BEFORE Calibration ---");
      runBiasCalibration();
    }
  }

  if (IMU.accelerationAvailable()) {
    float ax, ay, az;
    IMU.readAcceleration(ax, ay, az);

    float scaled_data[3] = { (ax - scaler_mean[0]) / scaler_scale[0], (ay - scaler_mean[1]) / scaler_scale[1], (az - scaler_mean[2]) / scaler_scale[2] };
    float pca_value = scaled_data[0] * pca_components[0] + 
                      scaled_data[1] * pca_components[1] + 
                      scaled_data[2] * pca_components[2];
    
    data_window[window_index++] = pca_value;

    if (window_index >= WINDOW_SIZE) {
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        model_input->data.f[i] = data_window[i];
      }
      
      if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed!");
        return;
      }
      
      float reconstruction_error = 0;
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        float raw_output = model_output->data.f[i];
        
        float corrected_output = raw_output;
        if (bias_is_calibrated) {
          corrected_output += output_bias_vector[i];
        }
        
        float diff = corrected_output - data_window[i];
        reconstruction_error += abs(diff); // 与您代码保持一致，使用 MAE
      }
      reconstruction_error /= WINDOW_SIZE;
      
      Serial.print("Reconstruction Error: ");
      Serial.print(reconstruction_error, 6);

      if (reconstruction_error < threshold_normal_vs_stuck) {
        Serial.println("  -> Status: Normal");
      }
      else if (reconstruction_error < threshold_stuck_vs_tilted) {
        Serial.println("  -> Status: Stuck");
      }
      else {
        Serial.println("  -> Status: Tilted");
      }

      window_index = 0;
    }
  }
  delay(9);
}