import serial
import csv
import datetime

usb_serialport = r"/dev/cu.usbserial-1410"
baudrate = 9600

def get_system_datetime():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return date_time

def convert_to_decimal_degrees(coordinate):
    # Split the coordinate into two parts: degrees and minutes
    degrees, minutes = coordinate.split(".")
    degrees = int(degrees[:-2])
    minutes = int(degrees[-2:] + minutes) / 60.0

    # Return the coordinate in decimal degrees format
    return degrees + minutes

# Attempt to open the serial port at defined baud rate
try:
    ser = serial.Serial(usb_serialport, baudrate)
except serial.SerialException as e:
    print("Serial port " + usb_serialport + "is not available:", e)

# Open a CSV file for writing
date_time = get_system_datetime()
with open(date_time + '.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time', 'Latitude', 'Direction', 'Longitude', 'Direction', 'Sats Locked', 'Altitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8')
        
        # Check if the line contains the desired data
        if line.startswith('$GPGGA'):
            # Split the line into fields
            fields = line.split(',')
            
            # Extract the time, latitude, and longitude fields
            time = fields[1]
            latitude = fields[2]
            latitude_dir = fields[3]
            longitude = fields[4]
            longitude_dir = fields[5]
            satellites = fields[7]
            altitude = fields[9]

            if latitude != '' and longitude != '':
                # Convert the latitude and longitude values to decimal degrees format
                latitude = convert_to_decimal_degrees(latitude)
                longitude = convert_to_decimal_degrees(longitude)

                # Report to the user that the non-zero latitude and longitude values have been found
                print("Found non-zero latitude and longitude values:")
                print("Latitude:", latitude)
                print("Longitude:", longitude)

                # Wait for the user to press a key
                input("Press any key to start logging...")

                # Write the data to the CSV file
                writer.writerow({'UTC-Time': time, 'Latitude': latitude, 'Direction': latitude_dir , 'Longitude': longitude, 'Direction': longitude_dir, 'Sats Locked': satellites, 'Altitude': altitude})