import socket
import sys

def main():
    HOST_IP_ADDRESS = "192.168.0.2"  # Your PC's IP
    PORT = 30002  # Ensure UR5 is connecting to this port

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server_socket.bind((HOST_IP_ADDRESS, PORT))
    except socket.error as e:
        print(f"Error binding the socket: {e}")
        sys.exit(1)

    server_socket.listen()
    print(f"Listening on {HOST_IP_ADDRESS}:{PORT}")

    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")

    print("Press 's' to send XYZ coordinates to UR5 or 'q' to quit...")

    while True:
        user_input = input("Press 's' to send XYZ coordinates or 'q' to quit: ").strip().lower()

        if user_input == 'q':  # Quit the program if 'q' is pressed
            break
        elif user_input == 's':  # Send XYZ coordinates when 's' is pressed
            # Send the XYZ coordinates as separate integers (e.g., 1, 2, 3)
            x, y, z = 200, 400, 350
            print(f"Sending XYZ coordinates: ({x}, {y}, {z})")

            # Send the coordinates in the format: (val_1, val_2, val_3)\n
            message = f"({x/1000}, {y/1000}, {z/1000})\n"
            client_socket.send(message.encode())

        else:
            print("Invalid input. Press 's' to send XYZ coordinates or 'q' to quit.")

    # Close the connection when done
    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    main()
