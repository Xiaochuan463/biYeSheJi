from machine import I2C, Pin
i2c = I2C(0, scl=Pin(9), sda=Pin(8), freq=400000)

# MODE1 reset
i2c.writeto_mem(0x40, 0x00, b'\x00')

# 50Hz
prescale = int(25000000 / 4096 / 50 - 1)
i2c.writeto_mem(0x40, 0x00, b'\x10')
i2c.writeto_mem(0x40, 0xFE, bytes([prescale]))
i2c.writeto_mem(0x40, 0x00, b'\x00')

# CH0 50% 占空比
i2c.writeto_mem(0x40, 0x06, bytes([0, 0, 0x00, 0x08]))  # OFF=2048
