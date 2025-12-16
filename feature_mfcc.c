/*2025.12.16*/
#include "feature_mfcc.h"

/* 【修复 1】引入必要的标准库，解决 size_t 报错 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h> 
#include <stdint.h>

/* =========================================================================
 * Part 1: 参数配置 
 * ========================================================================= */
#define kNoiseReductionSmoothingBits 0 
#define kLogScaleShift 6

/* =========================================================================
 * Part 2: 引入 Google 前端库
 * ========================================================================= */
#include "frontend.h"
#include "frontend_util.h"

/* =========================================================================
 * Part 3: 内部状态与缓冲区
 * ========================================================================= */
static struct FrontendConfig frontend_config;
static struct FrontendState frontend_state;

#define FEATURE_COUNT 40
#define FEATURE_SLICES 49
static int8_t g_feature_buffer[FEATURE_SLICES][FEATURE_COUNT];
static volatile bool g_is_new_data_ready = false;

/* =========================================================================
 * Part 4: 函数实现
 * ========================================================================= */

void fe_init(void) {
  // 1. 清空缓冲区
  memset(g_feature_buffer, -128, sizeof(g_feature_buffer));
  g_is_new_data_ready = false;

  // 2. 初始化配置结构体
  memset(&frontend_config, 0, sizeof(frontend_config));

  // 3. 配置参数
  frontend_config.window.size_ms = 30;
  frontend_config.window.step_size_ms = 20;
  frontend_config.filterbank.num_channels = FEATURE_COUNT;
  
  // 【修复 2】设置正确的频率范围，解决 Assert 报错
  frontend_config.filterbank.lower_band_limit = 20.0;   
  frontend_config.filterbank.upper_band_limit = 8000.0; 

  frontend_config.noise_reduction.smoothing_bits = kNoiseReductionSmoothingBits;
  frontend_config.log_scale.enable_log = 1;
  frontend_config.log_scale.scale_shift = kLogScaleShift;

  // 4. 初始化状态
  FrontendPopulateState(&frontend_config, &frontend_state, 16000);
}

void fe_process_samples(const int16_t* audio_samples, int num_samples) {
  size_t num_samples_read; 
  struct FrontendOutput frontend_output = FrontendProcessSamples(
      &frontend_state, audio_samples, num_samples, &num_samples_read);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    memmove(&g_feature_buffer[0], &g_feature_buffer[1], 
            (FEATURE_SLICES - 1) * FEATURE_COUNT * sizeof(int8_t));

    const int16_t* raw_features = frontend_output.values + (i * FEATURE_COUNT);

    for (int j = 0; j < FEATURE_COUNT; ++j) {
        int32_t val = (int32_t)raw_features[j];

        // ================================================
        // 【核心修正】: 放大信号！
        // ================================================
        // 原理：STM32 麦克风采集的特征值(Log Energy)通常在 0~600 左右
        // 我们希望 (Max / K) - 128 ≈ 127 (最大亮值)
        // 所以 (600 / K) ≈ 255  ->  K ≈ 2.3
        // 我们取整数 2，效果最好。
        
        int32_t q_val = (val / 2) - 128; 

        if (q_val < -128) q_val = -128;
        if (q_val > 127) q_val = 127;

        g_feature_buffer[FEATURE_SLICES - 1][j] = (int8_t)q_val;
    }
    g_is_new_data_ready = true;
  }
}

bool fe_is_new_data_ready(void) {
    if (g_is_new_data_ready) {
        g_is_new_data_ready = false; 
        return true;
    }
    return false;
}

void fe_copy_features(int8_t* output) {
    memcpy(output, g_feature_buffer, sizeof(g_feature_buffer));
}
