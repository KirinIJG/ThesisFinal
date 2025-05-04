import serial
ser = serial.Serial('/dev/ttyUSB0', 38400, timeout=1)
ser.write(b'ATZ\r')
print(ser.read(100).decode(errors="ignore"))
ser.write(b'ATI\r')
print(ser.read(100).decode(errors="ignore"))
ser.write(b'ATSP0\r')
print(ser.read(100).decode(errors="ignore"))
ser.write(b'0100\r')
print(ser.read(100).decode(errors="ignore"))

ser.close()

