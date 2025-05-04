import obd

connection = obd.OBD("/dev/ttyUSB0", baudrate=38400)  # add force=True if needed

if connection.is_connected():
    print("Connected to ELM327 OBD-II adapter")
    response = connection.query(obd.commands.SPEED)
    print(response)
else:
    print("Failed to connect")

