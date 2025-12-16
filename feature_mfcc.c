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


#include "main.h"
#include "stlogo.h"
#include "gpio.h"
#include "dma.h"
#include "usart.h"
#include "stm32h747i_discovery_audio.h"
#include "app_x-cube-ai.h"
#include "micro_speech.h"
#include "micro_speech_data.h"
#include "feature_mfcc.h" 
#include "ai_platform.h"
#include <stdio.h>
#include <string.h>
#include "ai_device_adaptor.h"

/* Private define ------------------------------------------------------------*/
#define AUDIO_FREQUENCY            16000U
#define AUDIO_IN_PDM_BUFFER_SIZE   (uint32_t)(128*AUDIO_FREQUENCY/16000*2)
#define RECORD_BUFFER_SIZE         4096

/* Private variables ---------------------------------------------------------*/
#if defined ( __CC_ARM ) 
  ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]) __attribute__((section(".RAM_D3")));
#elif defined ( __ICCARM__ ) 
  #pragma location=0x38000000
  ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]);
#elif defined ( __GNUC__ ) 
  ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]) __attribute__((section(".RAM_D3")));
#endif

ALIGN_32BYTES (uint16_t RecPlayback[2*RECORD_BUFFER_SIZE]);
uint32_t playbackPtr = 0;

BSP_AUDIO_Init_t AudioInInit;

/* —— 模型输出反量化参数 —— */
#define QOUT_SCALE  (0.00390625f)   // 1/256
#define QOUT_ZP     (-128)

#define IN_SIZE   (49*40)
#define OUT_SIZE  (4)

/* AI 缓冲区 */
AI_ALIGNED(4) static int8_t in_q[IN_SIZE];    
AI_ALIGNED(4) static int8_t out_q[OUT_SIZE];  
static float probs[OUT_SIZE];

static ai_handle net = AI_HANDLE_NULL;
AI_ALIGNED(4) static ai_u8 activations[AI_MICRO_SPEECH_DATA_ACTIVATIONS_SIZE];

/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void MPU_Config(void);
static void CPU_CACHE_Enable(void);
void Error_Handler(void);

/* 输出反量化工具 */
static inline float dequant_s8(int8_t q) {
    return QOUT_SCALE * ((int)q - QOUT_ZP); 
}

/* 初始化AI模型 */
static int ai_init(void) {
    ai_error err;
    const ai_network_params params = {
        AI_MICRO_SPEECH_DATA_WEIGHTS(ai_micro_speech_data_weights_get()),
        AI_MICRO_SPEECH_DATA_ACTIVATIONS(activations)
    };
    err = ai_micro_speech_create(&net, AI_MICRO_SPEECH_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) return -1;
    if (!ai_micro_speech_init(net, &params)) return -2;
    return 0;
}

int fputc(int ch, FILE *f) {
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

/* =========================================================================
                                     Main Function
   ========================================================================= */
int main(void) {
    /* 1. 硬件初始化 */
    HAL_Init();
    SystemClock_Config();
    MPU_Config();       
    CPU_CACHE_Enable(); 

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART1_UART_Init();
    
    printf("\r\n=== STM32H7 Audio AI: Google Streaming + Right Channel ===\r\n");

    /* 2. 软件模块初始化 */
    fe_init();          // 初始化特征提取
    if(ai_init() != 0){
        printf("AI Init Failed!\r\n");
        Error_Handler();
    }
    MX_X_CUBE_AI_Init(); 

    /* 3. 音频采集初始化 (双声道 16kHz) */
    AudioInInit.Device = AUDIO_IN_DEVICE_DIGITAL_MIC;
    AudioInInit.ChannelsNbr = 2; 
    AudioInInit.SampleRate = 16000;
    AudioInInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInInit.Volume = 100;

    BSP_AUDIO_IN_Init(1, &AudioInInit);

    /* 4. 启动录音 */
    HAL_Delay(500); 
    BSP_AUDIO_IN_RecordPDM(1, (uint8_t*)&recordPDMBuf, AUDIO_IN_PDM_BUFFER_SIZE * sizeof(uint16_t));
    /* 5. 准备推理 Buffer 指针 */
    ai_buffer *io_in  = ai_micro_speech_inputs_get(net, NULL);
    ai_buffer *io_out = ai_micro_speech_outputs_get(net, NULL);

    io_in[0].format  = AI_BUFFER_FORMAT_S8;
    io_in[0].data    = AI_HANDLE_PTR(in_q);
    io_out[0].format = AI_BUFFER_FORMAT_S8;
    io_out[0].data   = AI_HANDLE_PTR(out_q);

    static const char* kLabels[OUT_SIZE] = {"silence", "unknown", "yes", "no"};

    /* =================== 主循环 =================== */
    while (1) {
        if (fe_is_new_data_ready()) {
            
            // 获取数据
            fe_copy_features(in_q);
						// ===========================================
						// 【调试核心】：看看声音够不够大
						// ===========================================
						int8_t max_val = -128;
						int8_t min_val = 127;
						for(int i=0; i<IN_SIZE; i++) {
								if (in_q[i] > max_val) max_val = in_q[i];
								if (in_q[i] < min_val) min_val = in_q[i];
						}
						
						// 打印调试信息
						// 正常说话时，Max 应该 > 0，甚至 > 50
						// 如果 Max 一直是 -128 或 -120，说明【增益还是太小】，需要去 feature_mfcc.c 把 >> 改得更小
						printf("DEBUG: Feat Max=%d, Min=%d\r\n", max_val, min_val);

            // 设置指针
            io_in[0].data = AI_HANDLE_PTR(in_q);
            io_out[0].data = AI_HANDLE_PTR(out_q);

            // 推理
            if (ai_micro_speech_run(net, io_in, io_out) != 1) {
                printf("AI Run Error\r\n");
                continue;
            }

            // 反量化
            float sum = 0.f;
            for (int i = 0; i < OUT_SIZE; ++i) {
                probs[i] = dequant_s8(out_q[i]);
                if (probs[i] < 0.f) probs[i] = 0.f;
                sum += probs[i];
            }
            if (sum > 1e-6f) {
                for (int i = 0; i < OUT_SIZE; ++i) probs[i] /= sum;
            }

            // 结果去抖动处理
            static int stable_cnt[OUT_SIZE] = {0};
            int best = 0;
            for (int i = 1; i < OUT_SIZE; ++i) {
                if (probs[i] > probs[best]) best = i;
            }

            for (int i = 0; i < OUT_SIZE; ++i) {
                stable_cnt[i] = (i == best) ? (stable_cnt[i] + 1) : 0;
            }

            // 打印结果 (去除 LED 操作)
            if (best > 1 && stable_cnt[best] >= 2 && probs[best] > 0.6f) {
                printf(">>> DETECTED: %s (%.2f)\r\n", kLabels[best], probs[best]);
                stable_cnt[best] = 0; // 重置计数
            } 
            
            // 调试用：如果需要看实时概率，取消下面注释
            printf("Sil:%.2f Unk:%.2f Yes:%.2f No:%.2f\r\n", probs[0], probs[1], probs[2], probs[3]);
        }
    }
}

/* =========================================================================
                          DMA Callbacks (右声道提取)
   ========================================================================= */

static int16_t pcm_mono_temp[128]; 

void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    if(Instance == 1U)
    {
        SCB_InvalidateDCache_by_Addr((uint32_t *)&recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE/2], AUDIO_IN_PDM_BUFFER_SIZE*2);
        BSP_AUDIO_IN_PDMToPCM(Instance, (uint16_t*)&recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE/2], &RecPlayback[playbackPtr]);
        
        uint32_t n_stereo_samples = AUDIO_IN_PDM_BUFFER_SIZE / 4; 
        uint32_t n_mono_samples = n_stereo_samples / 2;

        /* 提取右声道 */
        for(uint32_t i = 0; i < n_mono_samples; i++) {
            if (i < 128) {
                // 【修改点】：直接提取数据，不要乘 4，不要乘 16！
                // 保持原始波形，避免削顶失真
                pcm_mono_temp[i] = (int16_t)RecPlayback[playbackPtr + (2 * i) + 1];
            }
        }
        
        fe_process_samples(pcm_mono_temp, n_mono_samples);

        playbackPtr += n_stereo_samples;
        if(playbackPtr >= RECORD_BUFFER_SIZE * 2) playbackPtr = 0;
    }
}

void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    if(Instance == 1U)
    {
        SCB_InvalidateDCache_by_Addr((uint32_t *)&recordPDMBuf[0], AUDIO_IN_PDM_BUFFER_SIZE*2);
        BSP_AUDIO_IN_PDMToPCM(Instance, (uint16_t*)&recordPDMBuf[0], &RecPlayback[playbackPtr]);
        
        uint32_t n_stereo_samples = AUDIO_IN_PDM_BUFFER_SIZE / 4;
        uint32_t n_mono_samples = n_stereo_samples / 2;

        /* 提取右声道 */
        for(uint32_t i = 0; i < n_mono_samples; i++) {
            if (i < 128) {
                // 【修改点】：直接提取数据
                pcm_mono_temp[i] = (int16_t)RecPlayback[playbackPtr + (2 * i) + 1];
            }
        }
        
        fe_process_samples(pcm_mono_temp, n_mono_samples);
    
        playbackPtr += n_stereo_samples;
        if(playbackPtr >= RECORD_BUFFER_SIZE * 2) playbackPtr = 0;
    }
}

void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance) { Error_Handler(); }

/* =========================================================================
                          Helper Functions
   ========================================================================= */

/* 简单的死循环处理错误 */
void Error_Handler(void)
{
  while(1)
  {
      // 错误发生，程序卡死在这里
  }
}

void __aeabi_assert(const char *expr, const char *file, int line) {
    printf("Assert failed: %s, file %s, line %d\n", expr, file, line);
    while (1) { 
        // 断言失败，卡死
    }
}

static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;
  HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_OFF;
  RCC_OscInitStruct.CSIState = RCC_CSI_OFF;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 5;
  RCC_OscInitStruct.PLL.PLLN = 160;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
  ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
  if(ret != HAL_OK) Error_Handler();
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_D1PCLK1 | RCC_CLOCKTYPE_PCLK1 | \
                                 RCC_CLOCKTYPE_PCLK2  | RCC_CLOCKTYPE_D3PCLK1);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;
  ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4);
  if(ret != HAL_OK) Error_Handler();
  __HAL_RCC_CSI_ENABLE() ;
  __HAL_RCC_SYSCFG_CLK_ENABLE() ;
  HAL_EnableCompensationCell();
}

static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct;
  HAL_MPU_Disable();
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.BaseAddress = 0x00;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.BaseAddress = SDRAM_DEVICE_ADDR;
  MPU_InitStruct.Size = MPU_REGION_SIZE_32MB;
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER1;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.SubRegionDisable = 0x00;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.BaseAddress = 0x90000000;
  MPU_InitStruct.Size = MPU_REGION_SIZE_128MB;
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER2;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.SubRegionDisable = 0x0;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

static void CPU_CACHE_Enable(void)
{
  SCB_EnableICache();
  SCB_EnableDCache();
}


