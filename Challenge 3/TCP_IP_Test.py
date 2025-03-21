import socket
import time
import csv

# Function to send coordinates to the UR5
def send_coordinates_to_ur5(x, y, z, ip="192.168.0.5", port= 3002):
    # Create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the robot's controller
    a = sock.connect((ip, port))
    print(a)
    # Prepare the coordinate message to send to the UR5
    # Example format: "X, Y, Z"
    coordinates_message = f"{x},{y},{z}"

    try:
        # Send the coordinates to the robot
        sock.send(coordinates_message.encode('utf-8'))
        print(f"Sent coordinates to move to: X={x}, Y={y}, Z={z}")
    except Exception as e:
        print(f"Error sending coordinates: {e}")
    finally:
        # Close the connection
        sock.close()

# Example coordinates (replace these with the coordinates from your system)
coordinates = [
    [1, 2, 3],  # Example coordinates in meters
    [4, 5, 6],
    [7, 8, 9]
]

# Send the coordinates to the UR5 robot
for coord in coordinates:
    send_coordinates_to_ur5(coord[0], coord[1], coord[2])
    time.sleep(1)  # Wait between sending coordinates (if needed)
