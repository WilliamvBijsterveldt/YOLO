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

    print("Enter XYZ coordinates in the format (x,y,z) or type 'q' to quit...")

    while True:
        user_input = input("Enter XYZ coordinates (x,y,z) or 'q' to quit: ").strip()

        if user_input.lower() == 'q':  # Quit the program if 'q' is entered
            break

        # Parse user input
        try:
            x, y, z = map(int, user_input.strip("()").split(","))
            print(f"Sending XYZ coordinates: ({x}, {y}, {z})")

            # Convert to meters and format the message
            message = f"({x/1000}, {y/1000}, {z/1000})\n"
            client_socket.send(message.encode())

        except ValueError:
            print("Invalid format! Please enter coordinates in the format: (100,200,150)")

    # Close the connection when done
    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    main()
