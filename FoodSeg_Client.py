import argparse
import socket
import cv2
import FoodSeg_Protocol
import os


def main():
    parser = argparse.ArgumentParser(description="Script ejemplo que envía imagen a servidor FoodSeg")
    parser.add_argument("-i", "--path_to_image", required = True, help = "Path/a/imagen que se envía a analizar")
    parser.add_argument("-ip", "--ip_address", default = "127.0.0.1", help = "IP en la que abrir el servidor.")
    parser.add_argument("-p", "--port", default = "33334", help = "Puerto en el que abrir el servidor.")
    args = parser.parse_args()
    

    HOST, PORT = args.ip_address, int(args.port)
    image_path = args.path_to_image

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connceted to {HOST}:{PORT}")


    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rows, cols, _ = image.shape
    FoodSeg_Protocol.send_image_metadata(client_socket, image, rows, cols, encoding = "BGR")
    client_socket.sendall(image.tobytes())

    rows, cols, encoding, data_length = FoodSeg_Protocol.receive_image_metadata(client_socket)
    image_received = FoodSeg_Protocol.receive_image_data(client_socket, data_length, rows, cols, encoding)

    client_socket.close()

    show_image = image_received.copy()

    show_image[show_image != 0] = 255
    output_path = os.path.abspath('output_image.png')
    cv2.imwrite(output_path, show_image)


    cv2.imshow('Image',show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
