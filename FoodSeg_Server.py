import argparse
import socket
import FoodSeg_Protocol
from FoodSeg import FoodSeg
import cv2

def main():
    parser = argparse.ArgumentParser(description="Servidor que analiza imágenes usando FoodSeg utilizando un protocolo propio")
    parser.add_argument("- ip", "--ip_address", default = "127.0.0.1", help = "IP en la que abrir el servidor.")
    parser.add_argument("- p", "--port", default = "33334", help = "Puerto en el que abrir el servidor.")
    args = parser.parse_args()


    HOST, PORT = args.ip_address, int(args.port)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}")

    while True:
        server_client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        rows, cols, encoding, data_length = FoodSeg_Protocol.receive_image_metadata(server_client_socket)
        image_received = FoodSeg_Protocol.receive_image_data(server_client_socket, data_length, rows, cols, encoding)

        cv2.imshow('Image',image_received)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        extracted_mask = FoodSeg(image_received)

        FoodSeg_Protocol.send_image_metadata(server_client_socket, extracted_mask, rows, cols, "Grayscale")
        server_client_socket.sendall(extracted_mask.tobytes())
        
        server_client_socket.close()



if __name__ == "__main__":
    main()