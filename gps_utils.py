import os
import csv
import serial
import time
from adafruit_gps import GPS
from geopy.distance import distance as geopy_distance

from pipeline_utils import trigger_photos
import config

def gps_setup():
    # Get list of serial ports
    # command = '/bin/bash -c "ls /dev/ttyUSB* /dev/ttyACM*"'
    # ports = os.popen(command).read().splitlines()

    # # Print ports and prompt user to select one
    # for i, port in enumerate(ports):
    #     print(f"{i+1}. {port}")

    # while True:
    #     try:
    #         port_num = int(input("Select a port number: ")) - 1
    #         if port_num < 0 or port_num >= len(ports):
    #             print("Invalid selection. Please select a number from the list.")
    #         else:
    #             selected_port = ports[port_num]
    #             break
    #     except ValueError:
    #         print("Invalid input. Please enter a number.")

    # # Prompt user to select baud rate
    # while True:
    #     try:
    #         baud_rate = int(input("Select a baud rate (9600 or 38400): "))
    #         if baud_rate not in [9600, 38400]:
    #             print("Invalid baud rate. Please select either 9600 or 38400.")
    #         else:
    #             break
    #     except ValueError:
    #         print("Invalid input. Please enter a number.")

    while True:
        try:
            capture_distance = int(input("Set a capture interval distance (meters): "))
            if capture_distance <= 0:
                print("Invalid capture interval. Please enter a nonzero integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Connect to GPS
    uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)
    gps = GPS(uart, debug=False)

    # Optimize GPS message pipeline - don't send unnecessary data
    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    time.sleep(2)
    # Update frequency to highest rate: 10Hz
    gps.send_command(b"PMTK220,100")

    # Create a CSV file in the new folder and write the header
    with open(os.path.join('captures/' + config.mission_datetime, 'gps_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trigger Number", "UTC Time", "Latitude", "Longitude", "Altitude", "Speed (kts)", "Num Sats"])

    # Wait for a valid GPS fix
    while not gps.has_fix:
        gps.update()
        print("No Fix")
        time.sleep(5)

    # Wait for the user to press a key to start recording
    input("Press Enter to start recording...")
    return gps

def gps_distance_capture(gps, trigger_counter, capture_distance, device_infos, queues_dict):
    while not config.gps_quit_flag:
        gps.update()
        if not gps.has_fix:
            print("Lost GPS fix...")
            time.sleep(1)
            continue

        current_coordinates = (gps.latitude, gps.longitude)

        if trigger_counter == 0:
            print(f"Latitude: {gps.latitude}, Longitude: {gps.longitude}, Altitude: {gps.altitude_m}")
            for device_info in device_infos:
                trigger_photos(device_info, queues_dict)
            last_coordinates = current_coordinates
            # Write the initial coordinates to the CSV file
            with open(os.path.join('captures/' + config.mission_datetime, 'gps_data.csv'), 'a', newline='') as file:
                utc_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", gps.timestamp_utc)
                writer = csv.writer(file)
                writer.writerow([(trigger_counter), utc_time_str, gps.latitude, gps.longitude, gps.altitude_m, gps.speed_knots, gps.satellites])
            trigger_counter += 1
        else:
            # Calculate the distance between the last photo and the current position
            displacement = geopy_distance(last_coordinates, current_coordinates).meters
            if displacement > capture_distance:
                print(f"Latitude: {gps.latitude}, Longitude: {gps.longitude}, Altitude: {gps.altitude_m}")
                for device_info in device_infos:
                    trigger_photos(device_info, queues_dict)
                last_coordinates = current_coordinates
                trigger_counter += 1
                # Write the new coordinates to the CSV file
                with open(os.path.join('captures/' + config.mission_datetime, 'gps_data.csv'), 'a', newline='') as file:
                    utc_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", gps.timestamp_utc)
                    writer = csv.writer(file)
                    writer.writerow([trigger_counter, utc_time_str, gps.latitude, gps.longitude, gps.altitude_m, gps.speed_knots, gps.satellites])
                
        time.sleep(0.1)