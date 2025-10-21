/**
  * @brief  初始化DWT寄存器用于精确计时，由TRAE IDE协助生成
  * @retval None
	* @date 2025.10
  */
void InitDWT(void)
{
    /* 使能DWT外设 */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    /* 重置计数器 */
    DWT->CYCCNT = 0;
    /* 使能计数器 */
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/**
  * @brief  获取当前DWT计数器值
  * @retval 当前CPU周期计数
  */
uint32_t GetDWT_CycleCount(void)
{
    return DWT->CYCCNT;
}

/**
  * @brief  测试DTCM SRAM和SDRAM的读取延迟
  * @retval None
  * @note: 测试DTCM SRAM和SDRAM的读取延迟，分别在DTCM和SDRAM中分配内存，测试读取延迟
    -在DTCM中分配内存 (DTCM通常在0x20000000地址开始) 
    -在SDRAM中分配内存 (根据代码中的SDRAM测试，SDRAM地址为EXT_SDRAM_ADDR)
  * @date 2025.10
  */
volatile uint32_t dtcm_buffer[BUFFER_SIZE] __attribute__((section(".ARM.__at_0x20000000")));
volatile uint32_t sram_buffer[BUFFER_SIZE] __attribute__((section(".ARM.__at_0x24030000")));
volatile uint32_t *sdram_buffer = (volatile uint32_t *)EXT_SDRAM_ADDR;
void TestMemoryLatency(void)
{
    /* 在DTCM中分配内存 (DTCM通常在0x20000000地址开始) */    
    /* 在SDRAM中分配内存 (根据代码中的SDRAM测试，SDRAM地址为EXT_SDRAM_ADDR) */
    
    uint32_t start_cycle, end_cycle;
    uint64_t total_dtcm_cycles = 0;
    uint64_t total_sram_cycles = 0;
    uint64_t total_sdram_cycles = 0;
    uint32_t dummy = 0;  // 用于避免编译器优化掉读取操作
    
    printf("\n=== Memory Latency Test Start ===\n");
    
    /* 初始化缓冲区数据 */
    printf("Initializing buffers...\n");
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        dtcm_buffer[i] = i;
        sram_buffer[i] = i;
        sdram_buffer[i] = i;
    }
    
    /* 预热缓存 */
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        dummy += dtcm_buffer[i];
        dummy += sram_buffer[i];
        dummy += sdram_buffer[i];
    }
    
    /* 测试DTCM读取延迟 */
    printf("Testing DTCM read latency...\n");
    for (int iter = 0; iter < TEST_ITERATIONS; iter++)
    {
        int index = iter % BUFFER_SIZE;
        start_cycle = GetDWT_CycleCount();
        dummy = dtcm_buffer[index];  // 读取DTCM中的数据
        end_cycle = GetDWT_CycleCount();
        total_dtcm_cycles += (end_cycle - start_cycle);
    }
    /* 测试SRAM读取延迟 */
    printf("Testing SRAM read latency...");
        for (int iter = 0; iter < TEST_ITERATIONS; iter++)
    {
        int index = iter % BUFFER_SIZE;
        start_cycle = GetDWT_CycleCount();
        dummy = sram_buffer[index];  // 读取SRAM中的数据
        end_cycle = GetDWT_CycleCount();
        total_sram_cycles += (end_cycle - start_cycle);
    }


    /* 测试SDRAM读取延迟 */
    printf("Testing SDRAM read latency...\n");
    for (int iter = 0; iter < TEST_ITERATIONS; iter++)
    {
        int index = iter % BUFFER_SIZE;
        start_cycle = GetDWT_CycleCount();
        dummy = sdram_buffer[index];  // 读取SDRAM中的数据
        end_cycle = GetDWT_CycleCount();
        total_sdram_cycles += (end_cycle - start_cycle);
    }
    
    /* 计算平均延迟 */
    double avg_dtcm_cycles = (double)total_dtcm_cycles / TEST_ITERATIONS;
    double avg_sram_cycles = (double)total_sram_cycles / TEST_ITERATIONS;
    double avg_sdram_cycles = (double)total_sdram_cycles / TEST_ITERATIONS;
    
    /* 获取系统时钟频率 */
    double sys_clk_mhz = (double)SystemCoreClock / 1000000.0;
    double ns_per_cycle = 1000.0 / sys_clk_mhz;
    
    /* 输出结果 */
    printf("\n=== Memory Latency Test Results ===\n");
    printf("System Clock: %.2f MHz\n", sys_clk_mhz);
    printf("Time per CPU cycle: %.2f ns\n", ns_per_cycle);
    printf("\nDTCM SRAM:");
    printf("\n  Average read latency: %.2f cycles (%.2f ns)\n", avg_dtcm_cycles, avg_dtcm_cycles * ns_per_cycle);
    printf("SRAM:");
    printf("\n  Average read latency: %.2f cycles (%.2f ns)\n", avg_sram_cycles, avg_sram_cycles * ns_per_cycle);
    printf("SDRAM:");
    printf("\n  Average read latency: %.2f cycles (%.2f ns)\n", avg_sdram_cycles, avg_sdram_cycles * ns_per_cycle);
    printf("\nSDRAM latency is %.2f times higher than DTCM\n", avg_sdram_cycles / avg_dtcm_cycles);
    printf("=====================================\n\n");
    
    /* 使用dummy变量防止编译器优化 */
    if (dummy == 0) {}
}
