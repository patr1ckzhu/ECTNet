/**
 * ADS1299 SPI Driver Implementation
 *
 * SPI Mode 1: CPOL=0 (idle low), CPHA=1 (data on rising edge)
 * SPI clock: 1.125 MHz (72 MHz / 64)
 * Internal oscillator: 2.048 MHz (CLKSEL = HIGH)
 */

#include "ads1299.h"

static SPI_HandleTypeDef *_hspi;

/* ── Low-level SPI ───────────────────────────────────── */

static uint8_t SPI_Transfer(uint8_t tx)
{
    uint8_t rx = 0;
    HAL_SPI_TransmitReceive(_hspi, &tx, &rx, 1, 10);
    return rx;
}

void ADS1299_SendCommand(uint8_t cmd)
{
    ADS_CS_LOW();
    SPI_Transfer(cmd);
    ADS_CS_HIGH();
    /* most commands need >= 4 tCLK (~2 us at 2.048 MHz) */
    HAL_Delay(1);
}

void ADS1299_WriteReg(uint8_t addr, uint8_t val)
{
    ADS_CS_LOW();
    SPI_Transfer(0x40 | (addr & 0x1F));   /* WREG | addr   */
    SPI_Transfer(0x00);                   /* n-1 = 0       */
    SPI_Transfer(val);                    /* register data  */
    ADS_CS_HIGH();
    HAL_Delay(1);
}

uint8_t ADS1299_ReadReg(uint8_t addr)
{
    uint8_t val;
    ADS_CS_LOW();
    SPI_Transfer(0x20 | (addr & 0x1F));   /* RREG | addr   */
    SPI_Transfer(0x00);                   /* n-1 = 0       */
    val = SPI_Transfer(0x00);             /* read data     */
    ADS_CS_HIGH();
    return val;
}

void ADS1299_ReadData(uint8_t *buf)
{
    /* In RDATAC mode: after DRDY falls, just clock out 27 bytes */
    ADS_CS_LOW();
    for (int i = 0; i < ADS_RAW_BYTES; i++) {
        buf[i] = SPI_Transfer(0x00);
    }
    ADS_CS_HIGH();
}

/* ── Initialization ──────────────────────────────────── */

uint8_t ADS1299_ReadID(SPI_HandleTypeDef *hspi)
{
    _hspi = hspi;

    /* Must exit RDATAC before reading registers */
    ADS1299_SendCommand(ADS_CMD_SDATAC);
    return ADS1299_ReadReg(ADS_REG_ID);
}

void ADS1299_Init(SPI_HandleTypeDef *hspi)
{
    _hspi = hspi;

    /* ── 1. Hardware reset ────────────────────────────── */
    ADS_CS_HIGH();
    ADS_RESET_HIGH();
    HAL_Delay(200);             /* wait for VCAP1 to charge  */

    ADS_RESET_LOW();
    HAL_Delay(1);               /* pulse low >= 2 tCLK (~1us) */
    ADS_RESET_HIGH();
    HAL_Delay(1);               /* wait 18 tCLK (~9us)       */

    /* ── 2. Stop continuous read for register config ──── */
    ADS1299_SendCommand(ADS_CMD_SDATAC);

    /* ── 3. Write configuration registers ─────────────── */
    ADS1299_WriteReg(ADS_REG_CONFIG1, ADS_CONFIG1_250SPS);
    ADS1299_WriteReg(ADS_REG_CONFIG2, ADS_CONFIG2_NORMAL);
    ADS1299_WriteReg(ADS_REG_CONFIG3, ADS_CONFIG3_REFBUF);

    /* Wait for internal reference to settle */
    HAL_Delay(150);

    /* ── 4. Configure all 8 channels: gain=24, normal ─── */
    for (uint8_t ch = 0; ch < ADS_NUM_CH; ch++) {
        ADS1299_WriteReg(ADS_REG_CH1SET + ch, ADS_CHNSET_GAIN24);
    }

    /* ── 5. SRB1 as common reference for all channels ─── */
    ADS1299_WriteReg(ADS_REG_MISC1, ADS_MISC1_SRB1);

    /* ── 6. Route all channels to BIAS amplifier ──────── */
    ADS1299_WriteReg(ADS_REG_BIAS_SENSP, 0xFF);
    ADS1299_WriteReg(ADS_REG_BIAS_SENSN, 0xFF);

    /* ── 7. Start conversions (START pin is floating) ─── */
    ADS1299_SendCommand(ADS_CMD_START);

    /* ── 8. Enable continuous data output ─────────────── */
    ADS1299_SendCommand(ADS_CMD_RDATAC);
}

/* ── Mode switching ──────────────────────────────────── */

void ADS1299_StartTestSignal(void)
{
    ADS1299_SendCommand(ADS_CMD_SDATAC);
    ADS1299_SendCommand(ADS_CMD_STOP);

    /* Internal test signal: ~1 Hz square wave */
    ADS1299_WriteReg(ADS_REG_CONFIG2, ADS_CONFIG2_TEST);
    for (uint8_t ch = 0; ch < ADS_NUM_CH; ch++) {
        ADS1299_WriteReg(ADS_REG_CH1SET + ch, ADS_CHNSET_TEST);
    }

    ADS1299_SendCommand(ADS_CMD_START);
    ADS1299_SendCommand(ADS_CMD_RDATAC);
}

void ADS1299_StartNormal(void)
{
    ADS1299_SendCommand(ADS_CMD_SDATAC);
    ADS1299_SendCommand(ADS_CMD_STOP);

    ADS1299_WriteReg(ADS_REG_CONFIG2, ADS_CONFIG2_NORMAL);
    for (uint8_t ch = 0; ch < ADS_NUM_CH; ch++) {
        ADS1299_WriteReg(ADS_REG_CH1SET + ch, ADS_CHNSET_GAIN24);
    }

    ADS1299_SendCommand(ADS_CMD_START);
    ADS1299_SendCommand(ADS_CMD_RDATAC);
}
