/**
 * main.c User Code Snippets
 *
 * Copy each section into the corresponding USER CODE block
 * in the CubeMX-generated main.c
 *
 * ═══════════════════════════════════════════════════════
 * STM32CubeIDE (.ioc) Configuration:
 * ═══════════════════════════════════════════════════════
 *
 * [Pinout & Configuration]
 *
 *   SPI1:
 *     Mode:             Full-Duplex Master
 *     NSS:              Disable (Software)
 *     Prescaler:        64  (→ 1.125 MHz SCLK)
 *     CPOL:             Low
 *     CPHA:             2 Edge  (= SPI Mode 1)
 *     Data Size:        8 Bits
 *     First Bit:        MSB First
 *
 *   USART2:
 *     Mode:             Asynchronous
 *     Baud Rate:        115200
 *     Word Length:      8 Bits
 *     Parity:           None
 *     Stop Bits:        1
 *
 *   GPIO:
 *     PA4  → GPIO_Output   (label: ADS_CS,    default HIGH)
 *     PA8  → GPIO_Output   (label: ADS_RESET, default HIGH)
 *     PA9  → GPIO_EXTI9    (label: ADS_DRDY)
 *       External Interrupt: Falling edge trigger
 *       GPIO Pull:          No pull-up/pull-down
 *       NVIC: EXTI9_5 → Enable ✓
 *
 *   [Clock Configuration]
 *     HSE: 8 MHz crystal
 *     PLL: x9 → SYSCLK = 72 MHz
 *     APB1: /2 → 36 MHz (USART2)
 *     APB2: /1 → 72 MHz (SPI1)
 *
 * ═══════════════════════════════════════════════════════
 */


/* ───────────────────────────────────────────────────────
 * USER CODE BEGIN Includes
 * ─────────────────────────────────────────────────────── */
#include "ads1299.h"
#include <string.h>
#include <stdio.h>
/* USER CODE END Includes */


/* ───────────────────────────────────────────────────────
 * USER CODE BEGIN PV  (Private Variables)
 * ─────────────────────────────────────────────────────── */
volatile uint8_t drdy_flag = 0;

uint8_t ads_raw[ADS_RAW_BYTES];     /* 27 bytes from ADS1299 */

/* UART packet: [0xA0][seq][ch1..ch3 = 9B][0xC0] = 12 bytes (3ch for BT demo) */
/* For 8ch: change to PKT_SIZE=27, memcpy 24 bytes, end marker at [26] */
#define PKT_SIZE  12
uint8_t uart_pkt[PKT_SIZE];
uint8_t pkt_seq = 0;

char dbg[128];                      /* debug print buffer */
/* USER CODE END PV */


/* ───────────────────────────────────────────────────────
 * USER CODE BEGIN 2  (after all MX_xxx_Init() calls)
 *
 * Test sequence:
 *   Step 1: Read ID → verify SPI works
 *   Step 2: Init → start 250 Hz continuous sampling
 *   Step 3: (optional) call ADS1299_StartTestSignal()
 *            to output internal square wave without electrodes
 * ─────────────────────────────────────────────────────── */
/* Wait for HC-05 Bluetooth to pair before sending data */
HAL_Delay(10000);

ADS_CS_HIGH();
HAL_Delay(100);

/* ── Step 1: Verify SPI communication ─────────────────── */
uint8_t chip_id = ADS1299_ReadID(&hspi1);
sprintf(dbg, "\r\n== ADS1299 ==\r\nID: 0x%02X ", chip_id);
HAL_UART_Transmit(&huart2, (uint8_t*)dbg, strlen(dbg), 100);

if (chip_id == ADS1299_ID_8CH) {
    HAL_UART_Transmit(&huart2, (uint8_t*)"[OK 8-ch]\r\n", 11, 100);
} else if (chip_id == 0x00 || chip_id == 0xFF) {
    /* 0x00 = MISO stuck low / ADS not powered
     * 0xFF = MISO stuck high / CS not toggling       */
    sprintf(dbg, "[ERROR] SPI fault (check wiring)\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)dbg, strlen(dbg), 100);
    while (1) { HAL_Delay(1000); }
} else {
    sprintf(dbg, "[WARN] unexpected ID\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)dbg, strlen(dbg), 100);
}

/* ── Step 2: Full initialization ──────────────────────── */
ADS1299_Init(&hspi1);

/* ── Step 3 (uncomment to use test signal, no electrodes needed) ── */
// ADS1299_StartTestSignal();

sprintf(dbg, "Sampling at 250 Hz, 8 ch, gain=24\r\n");
HAL_UART_Transmit(&huart2, (uint8_t*)dbg, strlen(dbg), 100);
sprintf(dbg, "UART packet: [A0][seq][24B data][C0], %d bytes\r\n\r\n", PKT_SIZE);
HAL_UART_Transmit(&huart2, (uint8_t*)dbg, strlen(dbg), 100);
/* USER CODE END 2 */


/* ───────────────────────────────────────────────────────
 * USER CODE BEGIN 3  (inside while(1) loop)
 * ─────────────────────────────────────────────────────── */
if (drdy_flag) {
    drdy_flag = 0;

    /* Read 27 bytes: [status 3B][ch1 3B]...[ch8 3B] */
    ADS1299_ReadData(ads_raw);

    /* Build UART packet */
    uart_pkt[0]  = 0xA0;              /* start marker */
    uart_pkt[1]  = pkt_seq++;         /* sequence (wraps at 255) */
    memcpy(&uart_pkt[2], &ads_raw[3], 9);   /* 3ch x 3B (C3,C4,Cz), skip status */
    uart_pkt[11] = 0xC0;              /* end marker */

    /* Send (blocking ~1.0 ms at 115200 baud) */
    HAL_UART_Transmit(&huart2, uart_pkt, PKT_SIZE, 10);
}
/* USER CODE END 3 */


/* ───────────────────────────────────────────────────────
 * USER CODE BEGIN 4  (after main, EXTI callback)
 * ─────────────────────────────────────────────────────── */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_9) {     /* PA9 = DRDY falling edge */
        drdy_flag = 1;
    }
}
/* USER CODE END 4 */
