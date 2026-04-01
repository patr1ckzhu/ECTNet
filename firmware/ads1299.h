/**
 * ADS1299 SPI Driver for STM32F103
 *
 * Pin mapping (from schematic):
 *   SPI1: PA5(SCK), PA6(MISO/DOUT), PA7(MOSI/DIN)
 *   CS:   PA4 (software NSS)
 *   DRDY: PA9 (EXTI falling edge)
 *   RESET: PA8 (GPIO output)
 *   START: floating (internal pull-down, use SPI START command)
 *   CLKSEL: HIGH (internal 2.048 MHz oscillator)
 */

#ifndef ADS1299_H
#define ADS1299_H

#include "main.h"

/* ── SPI Commands ─────────────────────────────────────── */
#define ADS_CMD_WAKEUP    0x02
#define ADS_CMD_STANDBY   0x04
#define ADS_CMD_RESET     0x06
#define ADS_CMD_START     0x08
#define ADS_CMD_STOP      0x0A
#define ADS_CMD_RDATAC    0x10    /* continuous read mode */
#define ADS_CMD_SDATAC    0x11    /* stop continuous read */
#define ADS_CMD_RDATA     0x12    /* read single sample  */

/* ── Register Addresses ──────────────────────────────── */
#define ADS_REG_ID        0x00
#define ADS_REG_CONFIG1   0x01
#define ADS_REG_CONFIG2   0x02
#define ADS_REG_CONFIG3   0x03
#define ADS_REG_LOFF      0x04
#define ADS_REG_CH1SET    0x05
#define ADS_REG_CH2SET    0x06
#define ADS_REG_CH3SET    0x07
#define ADS_REG_CH4SET    0x08
#define ADS_REG_CH5SET    0x09
#define ADS_REG_CH6SET    0x0A
#define ADS_REG_CH7SET    0x0B
#define ADS_REG_CH8SET    0x0C
#define ADS_REG_BIAS_SENSP 0x0D
#define ADS_REG_BIAS_SENSN 0x0E
#define ADS_REG_LOFF_SENSP 0x0F
#define ADS_REG_LOFF_SENSN 0x10
#define ADS_REG_LOFF_FLIP  0x11
#define ADS_REG_LOFF_STATP 0x12
#define ADS_REG_LOFF_STATN 0x13
#define ADS_REG_GPIO      0x14
#define ADS_REG_MISC1     0x15
#define ADS_REG_MISC2     0x16
#define ADS_REG_CONFIG4   0x17

/* ── Preset Register Values ──────────────────────────── */
#define ADS_CONFIG1_250SPS    0x96  /* HR mode, no daisy, 250 SPS */
#define ADS_CONFIG2_NORMAL    0xC0  /* test signal off */
#define ADS_CONFIG2_TEST      0xD0  /* internal test signal on */
#define ADS_CONFIG3_REFBUF    0xEC  /* internal ref + BIAS buffer on */
#define ADS_CHNSET_GAIN24     0x60  /* gain=24, normal electrode input */
#define ADS_CHNSET_TEST       0x65  /* gain=24, internal test signal */
#define ADS_CHNSET_SHORTED    0x61  /* gain=24, input shorted (noise test) */
#define ADS_MISC1_SRB1        0x20  /* SRB1 as common reference */

/* ── Expected Chip ID ────────────────────────────────── */
#define ADS1299_ID_8CH        0x3E

/* ── Constants ───────────────────────────────────────── */
#define ADS_NUM_CH            8
#define ADS_RAW_BYTES         27    /* 3B status + 8 x 3B channel */

/* ── Pin Control ─────────────────────────────────────── */
#define ADS_CS_LOW()     HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET)
#define ADS_CS_HIGH()    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET)
#define ADS_RESET_LOW()  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET)
#define ADS_RESET_HIGH() HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET)

/* ── API ─────────────────────────────────────────────── */
uint8_t  ADS1299_ReadID(SPI_HandleTypeDef *hspi);
void     ADS1299_Init(SPI_HandleTypeDef *hspi);
void     ADS1299_SendCommand(uint8_t cmd);
void     ADS1299_WriteReg(uint8_t addr, uint8_t val);
uint8_t  ADS1299_ReadReg(uint8_t addr);
void     ADS1299_ReadData(uint8_t *buf);       /* reads 27 bytes raw */
void     ADS1299_StartTestSignal(void);        /* switch to test mode */
void     ADS1299_StartNormal(void);            /* switch to electrode mode */

#endif /* ADS1299_H */
