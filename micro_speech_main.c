/**TinyML Micro_speech test
  ******************************************************************************
  * @file    BSP/CM7/Src/main.c
  * @author  MCD Application Team
  * @brief   This example code shows how to use the STM32H747I-DISCO BSP Drivers
  *          This is the main program for Cortex-M7.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
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
#include "ai_device_adaptor.h"

/** @addtogroup STM32H7xx_HAL_Examples
  * @{
  */

/** @addtogroup BSP
  * @{
  */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
//uint8_t DemoIndex = 0;
//__IO uint8_t NbLoop = 1;

#ifndef USE_FULL_ASSERT
uint32_t    ErrorCounter = 0;
#endif

/* Wave Player Pause/Resume Status. Defined as external in waveplayer.c file */
__IO uint32_t PauseResumeStatus = IDLE_STATUS;

/* Counter for Sel Joystick pressed*/
__IO uint32_t PressCount = 0;
__IO uint32_t ButtonState=0;
uint8_t toggle_led = 0;
__IO uint32_t CameraTest=0;
/* Volume of the audio playback */
/* Initial volume level (from 0% (Mute) to 100% (Max)) */
__IO uint8_t volume = 60;
__IO uint8_t VolumeChange = 0;
__IO uint32_t SRAMTest = 0;
__IO uint32_t SdramTest=0;
/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void Display_DemoDescription(void);
static void MPU_Config(void);
static void CPU_CACHE_Enable(void);

//BSP_DemoTypedef  BSP_examples[]=
//{
//  {AudioRecord_demo, "AUDIO RECORD", 0},
//};

extern AUDIO_ErrorTypeDef AUDIO_Start(uint32_t audio_start_address, uint32_t audio_file_size);
#define AUDIO_FREQUENCY            16000U
#define AUDIO_IN_PDM_BUFFER_SIZE   (uint32_t)(128*AUDIO_FREQUENCY/16000*2)
#define AUDIO_NB_BLOCKS    ((uint32_t)4)
#define AUDIO_BLOCK_SIZE   ((uint32_t)0xFFFE)
/* Size of the recorder buffer */
//#define RECORD_DURATION_SEC       2U
//#define RECORD_BUFFER_SIZE        (AUDIO_FREQUENCY * RECORD_DURATION_SEC)
#define RECORD_BUFFER_SIZE        4096
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Define record Buf at D3SRAM @0x38000000 since the BDMA for SAI4 use only this memory */
#if defined ( __CC_ARM )  /* !< ARM Compiler */
  ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]) __attribute__((section(".RAM_D3")));
#elif defined ( __ICCARM__ )  /* !< ICCARM Compiler */
  #pragma location=0x38000000
ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]);
#elif defined ( __GNUC__ )  /* !< GNU Compiler */
  ALIGN_32BYTES (uint16_t recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE]) __attribute__((section(".RAM_D3")));
#endif
static uint32_t AudioFreq[9] = {8000 ,11025, 16000, 22050, 32000, 44100, 48000, 96000, 192000};
ALIGN_32BYTES (uint16_t  RecPlayback[2*RECORD_BUFFER_SIZE]);
ALIGN_32BYTES (uint16_t  PlaybackBuffer[2*RECORD_BUFFER_SIZE]);
uint32_t VolumeLevel = 80;
uint32_t  InState = 0;
uint32_t *AudioFreq_ptr;
uint16_t playbackBuf[RECORD_BUFFER_SIZE*2];
//volatile uint8_t recordDone = 0;

BSP_AUDIO_Init_t  AudioInInit;
/* Pointer to record_data */
uint32_t playbackPtr=0;
uint32_t AudioBufferOffset;
/* Private function prototypes -----------------------------------------------*/
typedef enum {
  BUFFER_OFFSET_NONE = 0,
  BUFFER_OFFSET_HALF,
  BUFFER_OFFSET_FULL,
}BUFFER_StateTypeDef;

#define QIN_SCALE   (0.1014093166f)
#define QIN_ZP      (-128)
#define QOUT_SCALE  (0.00390625f)   // 1/256
#define QOUT_ZP     (-128)

#define IN_SIZE   (49*40)    // 特征大小
#define OUT_SIZE  (4)        // 输出类别数

static int8_t in_q[IN_SIZE];    // 输入数据
static int8_t out_q[OUT_SIZE];  // 模型输出数据
static float probs[OUT_SIZE];   // 存放概率值

// 量化和反量化工具
static inline int8_t quantize_s8(float x) {
    float qf = x / QIN_SCALE - (-QIN_ZP);
    int q = (int)lrintf(qf);
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static inline float dequant_s8(int8_t q) {
    return QOUT_SCALE * ((int)q - QOUT_ZP);  // 反量化公式
}

static ai_handle net = AI_HANDLE_NULL;
AI_ALIGNED(4) static ai_u8 activations[AI_MICRO_SPEECH_DATA_ACTIVATIONS_SIZE];

// 初始化AI模型
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

// Python 生成的特征数据（此处您可以手动填充特征数据）
static const int8_t feat1960[IN_SIZE] = 
	{
	-64, -70, -65, -70, -65, -70, -69, -71, -66, -70, -66, -70, 
  -66, -70, -65, -70, -65, -70, -66, -72, -69, -71, -69, -74, 
  -72, -73, -72, -73, -74, -77, -75, -78, -75, -74, -73, -76, 
  -72, -73, -70, -73, -73, -74, -69, -76, -75, -79, -66, -71, 
  -68, -74, -69, -75, -66, -73, -70, -73, -72, -84, -75, -76, 
  -72, -79, -75, -84, -78, -77, -72, -78, -78, -78, -76, -79, 
  -76, -79, -73, -75, -73, -76, -71, -76, -73, -86, -73, -77, 
  -74, -78, -69, -77, -79, -83, -72, -74, -71, -75, -69, -82, 
  -74, -77, -70, -84, -78, -81, -76, -81, -76, -85, -77, -83, 
  -77, -80, -76, -80, -76, -79, -77, -77, -73, -79, -75, -78, 
  -73, -81, -73, -80, -80, -96, -81, -90, -85, -96, -80, -83, 
  -77, -85, -75, -85, -76, -94, -78, -85, -82, -92, -85, -93, 
  -83, -97, -82, -88, -77, -80, -78, -83, -77, -83, -76, -83, 
  -76, -81, -77, -85, -81, -86, -85, -96, -79, -90, -86, -128, 
  -100, -99, -90, -98, -82, -99, -85, -128, -87, -128, -92, -110, 
  -93, -104, -83, -88, -81, -94, -81, -86, -77, -86, -77, -86, 
  -78, -85, -77, -81, -77, -84, -75, -79, -92, -100, -82, -84, 
  -77, -84, -102, -128, -89, -110, -128, -128, -128, -128, -128, -128, 
  -114, -128, -128, -128, -92, -114, -87, -99, -89, -98, -88, -103, 
  -87, -94, -80, -86, -81, -83, -78, -85, -77, -85, -77, -82, 
  -82, -90, -78, -95, -81, -87, -81, -128, -79, -85, -77, -78, 
  -72, -78, -75, -88, -73, -82, -80, -83, -80, -88, -83, -95, 
  -82, -91, -81, -81, -78, -90, -80, -86, -81, -83, -76, -79, 
  -72, -81, -76, -85, -74, -75, -73, -74, -69, -78, -74, -79, 
  -66, -71, -67, -72, -68, -74, -69, -72, -68, -74, -67, -73, 
  -68, -72, -71, -74, -75, -76, -70, -77, -75, -88, -79, -80, 
  -74, -83, -77, -88, -80, -88, -81, -93, -67, -72, -67, -76, 
  -70, -72, -67, -71, -69, -72, -66, -75, -68, -73, -66, -75, 
  -67, -72, -70, -75, -66, -71, -66, -73, -67, -78, -70, -74, 
  -73, -77, -72, -77, -69, -78, -72, -78, -74, -78, -73, -84, 
  -70, -78, -71, -79, -69, -74, -68, -75, -69, -74, -65, -73, 
  -69, -75, -67, -73, -70, -73, -67, -73, -72, -77, -70, -79, 
  -72, -80, -72, -77, -76, -78, -77, -79, -71, -78, -71, -77, 
  -72, -76, -76, -85, -66, -72, -71, -74, -69, -79, -77, -80, 
  -70, -79, -74, -75, -68, -75, -73, -76, -72, -81, -74, -80, 
  -72, -80, -75, -78, -72, -82, -74, -86, -81, -94, -89, -78, 
  -70, -79, -71, -76, -73, -79, -77, -81, -63, -69, -63, -69, 
  -63, -69, -63, -69, -63, -70, -67, -72, -65, -70, -65, -74, 
  -68, -74, -69, -74, -68, -74, -70, -77, -70, -77, -76, -81, 
  -73, -82, -88, -80, -73, -82, -73, -80, -73, -79, -73, -80, 
  -66, -76, -65, -71, -66, -70, -66, -71, -67, -72, -67, -74, 
  -70, -74, -67, -73, -70, -76, -69, -75, -72, -77, -69, -76, 
  -74, -76, -69, -74, -71, -84, -77, -73, -71, -77, -72, -81, 
  -73, -77, -75, -82, -68, -79, -68, -76, -68, -74, -71, -78, 
  -74, -79, -72, -81, -74, -82, -73, -78, -71, -77, -72, -82, 
  -72, -82, -71, -77, -70, -76, -77, -81, -72, -81, -82, -96, 
  -79, -82, -74, -84, -79, -100, -84, -95, -72, -82, -71, -81, 
  -69, -77, -82, -128, -93, -128, -78, -114, -84, -128, -78, -83, 
  -83, -101, -77, -85, -77, -128, -74, -78, -74, -83, -83, -107, 
  -75, -86, -79, -128, -91, -110, -80, -87, -79, -128, -85, -97, 
  -74, -82, -73, -82, -69, -78, -81, -128, -95, -128, -80, -128, 
  -95, -128, -82, -98, -81, -114, -85, -97, -87, -128, -79, -83, 
  -75, -81, -82, -128, -97, -90, -81, -128, -95, -96, -76, -84, 
  -79, -128, -97, -128, -77, -83, -76, -82, -71, -87, -85, -128, 
  -128, -128, -92, -128, -128, -128, -94, -103, -81, -128, -81, -107, 
  -95, -128, -83, -90, -82, -92, -114, -128, -104, -93, -85, -128, 
  -103, -92, -79, -94, -90, -128, -87, -98, -81, -85, -79, -86, 
  -77, -128, -89, -128, -128, -128, -128, -128, -128, -128, -128, -110, 
  -80, -100, -90, -128, -98, -128, -85, -98, -80, -87, -100, -128, 
  -97, -88, -81, -128, -92, -88, -87, -128, -86, -128, -90, -110, 
  -82, -85, -79, -104, -84, -128, -81, -107, -99, -128, -82, -128, 
  -114, -128, -91, -105, -84, -128, -90, -110, -87, -128, -84, -94, 
  -79, -82, -72, -81, -75, -81, -74, -92, -78, -82, -87, -128, 
  -86, -114, -86, -110, -89, -85, -82, -128, -95, -128, -79, -83, 
  -79, -128, -79, -95, -83, -94, -80, -128, -114, -128, -85, -114, 
  -94, -128, -77, -83, -77, -86, -75, -83, -92, -128, -85, -86, 
  -79, -83, -79, -82, -72, -102, -79, -91, -90, -87, -84, -128, 
  -128, -128, -79, -94, -85, -128, -84, -98, -84, -90, -79, -128, 
  -86, -105, -86, -128, -87, -95, -76, -87, -80, -95, -76, -97, 
  -85, -128, -88, -97, -82, -95, -90, -87, -76, -107, -82, -90, 
  -90, -89, -85, -128, -110, -128, -79, -96, -96, -128, -91, -128, 
  -90, -128, -128, -128, -103, -128, -128, -128, -89, -94, -77, -88, 
  -84, -93, -78, -105, -96, -128, -87, -128, -114, -128, -95, -93, 
  -76, -96, -86, -102, -87, -92, -84, -128, -101, -114, -81, -105, 
  -86, -128, -81, -90, -77, -100, -128, -128, -128, -128, -128, -128, 
  -107, -104, -80, -105, -91, -128, -82, -128, -110, -128, -95, -128, 
  -128, -128, -94, -91, -76, -92, -85, -102, -86, -96, -83, -128, 
  -92, -128, -82, -128, -89, -93, -88, -105, -92, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -82, -128, -110, -128, -96, -128, 
  -128, -128, -93, -128, -128, -128, -128, -98, -80, -105, -94, -100, 
  -85, -96, -84, -128, -93, -107, -78, -128, -94, -100, -82, -89, 
  -110, -128, -128, -128, -128, -128, -128, -128, -128, -128, -83, -128, 
  -128, -128, -88, -128, -128, -128, -114, -128, -128, -128, -105, -103, 
  -82, -99, -96, -128, -83, -128, -83, -114, -88, -128, -78, -128, 
  -87, -86, -81, -93, -128, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -110, -128, -128, -128, -114, -128, -128, -128, -128, -128, 
  -128, -128, -128, -102, -83, -128, -92, -107, -84, -114, -84, -128, 
  -88, -128, -79, -110, -90, -92, -81, -83, -94, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -114, -128, -128, -128, -100, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -92, -128, -90, -128, 
  -86, -128, -128, -128, -128, -128, -85, -98, -94, -128, -82, -95, 
  -80, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -97, -128, -128, -128, -102, -128, -128, -128, -128, -128, 
  -93, -128, -92, -128, -128, -128, -114, -128, -128, -128, -102, -128, 
  -92, -128, -85, -96, -83, -114, -128, -128, -128, -128, -128, -128, 
  -114, -128, -88, -128, -128, -128, -84, -101, -95, -100, -81, -128, 
  -128, -128, -128, -128, -114, -128, -90, -128, -91, -100, -96, -128, 
  -128, -128, -128, -128, -93, -128, -82, -81, -78, -83, -81, -114, 
  -93, -110, -98, -97, -91, -102, -83, -95, -97, -97, -79, -84, 
  -91, -85, -77, -91, -98, -128, -91, -128, -128, -107, -90, -128, 
  -128, -128, -128, -128, -128, -128, -128, -91, -81, -100, -86, -90, 
  -78, -82, -77, -95, -78, -91, -91, -93, -92, -88, -80, -92, 
  -89, -96, -81, -89, -80, -85, -79, -90, -82, -86, -84, -128, 
  -128, -110, -92, -128, -128, -128, -128, -128, -128, -128, -89, -128, 
  -87, -91, -82, -93, -90, -92, -78, -77, -77, -87, -81, -90, 
  -84, -83, -77, -90, -85, -114, -85, -93, -84, -91, -93, -128, 
  -86, -86, -83, -128, -128, -96, -87, -114, -82, -128, -128, -128, 
  -128, -128, -103, -107, -94, -128, -85, -105, -87, -128, -86, -87, 
  -78, -89, -85, -93, -78, -85, -92, -128, -95, -128, -86, -98, 
  -93, -128, -128, -114, -81, -86, -82, -128, -128, -110, -94, -95, 
  -78, -110, -79, -82, -91, -85, -86, -82, -76, -79, -80, -89, 
  -81, -86, -77, -82, -79, -91, -78, -86, -82, -91, -83, -128, 
  -93, -128, -90, -96, -91, -128, -128, -91, -76, -83, -85, -128, 
  -128, -128, -100, -107, -80, -128, -79, -87, -86, -89, -82, -85, 
  -78, -86, -83, -107, -86, -94, -82, -81, -78, -83, -81, -91, 
  -81, -83, -80, -105, -90, -114, -85, -90, -80, -128, -100, -82, 
  -77, -88, -85, -128, -128, -128, -95, -105, -81, -128, -83, -104, 
  -83, -94, -84, -107, -100, -128, -128, -128, -105, -128, -114, -114, 
  -85, -102, -100, -128, -91, -128, -114, -128, -114, -128, -128, -128, 
  -107, -110, -110, -128, -95, -100, -128, -128, -128, -128, -128, -128, 
  -82, -128, -87, -114, -80, -86, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -97, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -94, -96, -128, -128, 
  -128, -104, -107, -128, -86, -128, -91, -128, -81, -94, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -114, -128, -128, -128, 
  -128, -128, -103, -128, -128, -128, -128, -128, -105, -96, -128, -128, 
  -92, -105, -128, -128, -128, -128, -128, -128, -90, -128, -93, -128, 
  -83, -110, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -114, -103, -128, -128, -96, -128, -128, -128, -128, -128, -105, -128, 
  -128, -128, -107, -128, -91, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -107, 
  -100, -128, -96, -114, -91, -104, -88, -128, -107, -128, -128, -128, 
  -101, -98, -110, -128, -101, -128, -107, -128, -114, -128, -85, -114, 
  -128, -128, -128, -128, -128, -128, -104, -128, -128, -128, -114, -114, 
  -94, -128, -91, -94, -104, -128, -128, -94, -87, -90, -76, -96, 
  -98, -128, -128, -128, -87, -97, -91, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -114, -128, -128, -128, -97, -95, -88, -128, 
  -128, -128, -128, -128, -128, -128, -105, -96, -91, -128, -95, -97, 
  -128, -128, -128, -128, -128, -128, -128, -128, -95, -100, -96, -128, 
  -128, -128, -128, -128, -128, -128, -93, -128, -107, -128, -94, -114, 
  -86, -85, -91, -128, -102, -128, -128, -128, -105, -128, -89, -128, 
  -96, -128, -92, -128, -128, -128, -128, -128, -128, -128, -128, -114, 
  -96, -128, -90, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -103, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -114, -128, -90, -104, -128, -128, -105, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -95, -114, -107, -128, -114, -128, -128, -128, 
  -128, -128, -88, -128, -128, -128, -128, -107, -85, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -110, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -89, -107, -95, -128, 
  -103, -104, -95, -128, -114, -114, -82, -128, -105, -128, -128, -128, 
  -90, -128, -128, -128, -128, -128, -128, -128, -128, -128, -99, -128, 
  -128, -128, -114, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -95, -128, -105, -128, -99, -103, -90, -128, -97, -107, -78, -97, 
  -87, -105, -95, -128, -110, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -114, -128, -97, -128, -104, -104, -93, -128, 
  -99, -114, -79, -100, -88, -100, -92, -105, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -97, -128, -128, -128, -128, -128, 
  -128, -128, -128, -128, -128, -128, -128, -128, -114, -128, -101, -128, 
  -99, -128, -103, -128, -100, -128, -81, -105, -128, -128, -100, -107, 
  -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -98, -114, 
  -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -107, 
  -102, -128, -104, -128,};

int fputc(int ch, FILE *f) {
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

AI_ALIGNED(4) static ai_i8 in_q[IN_SIZE];
AI_ALIGNED(4) static ai_i8 out_q[OUT_SIZE];

int main(void) {
    uint32_t channel_nbr = 2;

    AudioInInit.Device = AUDIO_IN_DEVICE_DIGITAL_MIC;
    AudioInInit.ChannelsNbr = channel_nbr;
    AudioInInit.SampleRate = 16000;  // 16kHz采样率
    AudioInInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInInit.Volume = 80;

    /* 初始化系统时钟等配置 */
    HAL_Init();
    SystemClock_Config();
    fe_init();                  // 前处理 (窗/FFT/Mel)
    ai_init();                  // 初始化 AI 模型
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART1_UART_Init();
    MX_X_CUBE_AI_Init();
//    BSP_AUDIO_IN_Init(1, &AudioInInit);

    // 模型输入输出 buffer
    ai_buffer *io_in  = ai_micro_speech_inputs_get(net, NULL);
    ai_buffer *io_out = ai_micro_speech_outputs_get(net, NULL);

    io_in[0].format  = AI_BUFFER_FORMAT_S8;
    io_out[0].format = AI_BUFFER_FORMAT_S8;
    io_in[0].data  = AI_HANDLE_PTR(in_q);
    io_out[0].data = AI_HANDLE_PTR(out_q);

    // 标签
    static const char* kLabels[OUT_SIZE] = {"silence", "unknown", "yes", "no"};

    while (1) {
        // 假设这里在计算特征时，fe_compute_49x40返回true，表示我们有足够的数据
        for (int i = 0; i < IN_SIZE; ++i) {
        in_q[i] = feat1960[i];
    }

    // 每次推理前，重新设置数据指针（确保指针指向量化后的数据）
    io_in[0].data = AI_HANDLE_PTR(in_q);
    io_out[0].data = AI_HANDLE_PTR(out_q);

    // 执行推理
    if (ai_micro_speech_run(net, io_in, io_out) != 1) {
        ai_error e = ai_micro_speech_get_error(net);
        printf("ai_run err type=%d code=%d\r\n", e.type, e.code);
        continue;
    }

    // 反量化输出结果并计算概率
    float sum = 0.f;
		for (int i = 0; i < OUT_SIZE; ++i) {
				probs[i] = dequant_s8(out_q[i]);  // 反量化
				if (probs[i] < 0.f) probs[i] = 0.f;  // 防止负数
				sum += probs[i];  // 计算总和

				// 打印每个 prob 的值，帮助调试
				printf("probs[%d] = %.4f\n", i, probs[i]);
		}

		if (sum > 1e-8f) {
				for (int i = 0; i < OUT_SIZE; ++i) probs[i] /= sum;  // 归一化
		}

		// 直接打印每次推理的概率
		printf("Detected: Silence=%.2f, Unknown=%.2f, Yes=%.2f, No=%.2f\n", probs[0], probs[1], probs[2], probs[3]);

    HAL_Delay(100);  // 加点延时避免过快的循环，防止 CPU 占用过高
  }
}

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow :
  *            System Clock source            = PLL (HSE)
  *            SYSCLK(Hz)                     = 400000000 (Cortex-M7 CPU Clock)
  *            HCLK(Hz)                       = 200000000 (Cortex-M4 CPU, Bus matrix Clocks)
  *            AHB Prescaler                  = 2
  *            D1 APB3 Prescaler              = 2 (APB3 Clock  100MHz)
  *            D2 APB1 Prescaler              = 2 (APB1 Clock  100MHz)
  *            D2 APB2 Prescaler              = 2 (APB2 Clock  100MHz)
  *            D3 APB4 Prescaler              = 2 (APB4 Clock  100MHz)
  *            HSE Frequency(Hz)              = 25000000
  *            PLL_M                          = 5
  *            PLL_N                          = 160
  *            PLL_P                          = 2
  *            PLL_Q                          = 4
  *            PLL_R                          = 2
  *            VDD(V)                         = 3.3
  *            Flash Latency(WS)              = 4
  * @param  None
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;

  /*!< Supply configuration update enable */
  HAL_PWREx_ConfigSupply(PWR_DIRECT_SMPS_SUPPLY);

  /* The voltage scaling allows optimizing the power consumption when the device is
     clocked below the maximum system frequency, to update the voltage scaling value
     regarding system frequency refer to product datasheet.  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /* Enable HSE Oscillator and activate PLL with HSE as source */
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
  if(ret != HAL_OK)
  {
    Error_Handler();
  }

/* Select PLL as system clock source and configure  bus clocks dividers */
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
  if(ret != HAL_OK)
  {
    Error_Handler();
  }

 /*
  Note : The activation of the I/O Compensation Cell is recommended with communication  interfaces
          (GPIO, SPI, FMC, QSPI ...)  when  operating at  high frequencies(please refer to product datasheet)
          The I/O Compensation Cell activation  procedure requires :
        - The activation of the CSI clock
        - The activation of the SYSCFG clock
        - Enabling the I/O Compensation Cell : setting bit[0] of register SYSCFG_CCCSR
 */

  /*activate CSI clock mondatory for I/O Compensation Cell*/
  __HAL_RCC_CSI_ENABLE() ;

  /* Enable SYSCFG clock mondatory for I/O Compensation Cell */
  __HAL_RCC_SYSCFG_CLK_ENABLE() ;

  /* Enables the I/O Compensation Cell */
  HAL_EnableCompensationCell();
}

/**
  * @brief  Display main demo messages
  * @param  None
  * @retval None
  */


/**
  * @brief  Check for user input
  * @param  None
  * @retval Input state (1 : active / 0 : Inactive)
  */
uint8_t CheckForUserInput(void)
{
  return ButtonState;
}

/**
  * @brief  Button Callback
  * @param  Button Specifies the pin connected EXTI line
  * @retval None
  */
void BSP_PB_Callback(Button_TypeDef Button)
{
  if(Button == BUTTON_WAKEUP)
  {

    ButtonState = 1;
  }

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
void Error_Handler(void)
{
  /* Turn LED REDon */
  BSP_LED_On(LED_RED);
  while(1)
  {
  }
}

#ifdef USE_FULL_ASSERT

/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

  /* Infinite loop */
  while (1)
  {
  }
}
#endif /* USE_FULL_ASSERT */

/**
  * @brief  Configure the MPU attributes as Write Through for SDRAM.
  * @note   The Base Address is SDRAM_DEVICE_ADDR.
  *         The Region Size is 32MB.
  * @param  None
  * @retval None
  */
static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct;

  /* Disable the MPU */
  HAL_MPU_Disable();

  /* Configure the MPU as Strongly ordered for not defined regions */
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

  /* Configure the MPU attributes as WT for SDRAM */
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

  /* Configure the MPU QSPI flash */
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

  /* Enable the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/**
  * @brief  CPU L1-Cache enable.
  * @param  None
  * @retval None
  */
static void CPU_CACHE_Enable(void)
{
  /* Enable I-Cache */
  SCB_EnableICache();

  /* Enable D-Cache */
  SCB_EnableDCache();
}

void  BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    if(Instance == 1U)
  {
    /* Invalidate Data Cache to get the updated content of the SRAM*/
    SCB_InvalidateDCache_by_Addr((uint32_t *)&recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE/2], AUDIO_IN_PDM_BUFFER_SIZE*2);

    BSP_AUDIO_IN_PDMToPCM(Instance, (uint16_t*)&recordPDMBuf[AUDIO_IN_PDM_BUFFER_SIZE/2], &RecPlayback[playbackPtr]);
		
    /* Clean Data Cache to update the content of the SRAM */
    SCB_CleanDCache_by_Addr((uint32_t*)&RecPlayback[playbackPtr], AUDIO_IN_PDM_BUFFER_SIZE/4);
		
		fe_push_pcm_isr((int16_t*)&RecPlayback[playbackPtr], AUDIO_IN_PDM_BUFFER_SIZE/4);
		

    playbackPtr += AUDIO_IN_PDM_BUFFER_SIZE/4/2;
    if(playbackPtr >= RECORD_BUFFER_SIZE)
		 playbackPtr = 0;
  }
  else
  {
    AudioBufferOffset = BUFFER_OFFSET_FULL;
  }

}

/**
  * @brief  Manages the DMA Half Transfer complete interrupt.
  * @param  None
  * @retval None
  */
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    if(Instance == 1U)
  {
    /* Invalidate Data Cache to get the updated content of the SRAM*/
    SCB_InvalidateDCache_by_Addr((uint32_t *)&recordPDMBuf[0], AUDIO_IN_PDM_BUFFER_SIZE*2);

    BSP_AUDIO_IN_PDMToPCM(Instance, (uint16_t*)&recordPDMBuf[0], &RecPlayback[playbackPtr]);
		
    /* Clean Data Cache to update the content of the SRAM */
    SCB_CleanDCache_by_Addr((uint32_t*)&RecPlayback[playbackPtr], AUDIO_IN_PDM_BUFFER_SIZE/4);
		
		fe_push_pcm_isr((int16_t*)&RecPlayback[playbackPtr], AUDIO_IN_PDM_BUFFER_SIZE/4);
	
    playbackPtr += AUDIO_IN_PDM_BUFFER_SIZE/4/2;
    if(playbackPtr >= RECORD_BUFFER_SIZE)
    {
			 playbackPtr = 0;
    }
  }
  else
  {
    AudioBufferOffset = BUFFER_OFFSET_HALF;
  }

}


/**
  * @brief  Audio IN Error callback function
  * @param  None
  * @retval None
  */
void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance)
{
  /* Stop the program with an infinite loop */
  Error_Handler();
}
