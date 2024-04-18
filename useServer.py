from multiprocessing.connection import Client, Listener
import pickle
import multiprocessing as mp
import cv2
import time
import socket
import numpy as np
from RealTimeELGandVGE import process
import threading
import dlib
import torch
import sys
sys.path.append(r'./VGE-pytorch')
from vge import VGE

def queue_img_put(q,stop_event):
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        q.put(cap.read()[1])
        if q.qsize() > 1:
            q.get()
        else:
            time.sleep(0.01)

def queue_img_get(q, window_name, host, port,stop_event):
    try:
        client = Client((host, port))
    except Exception as e:
        print(f"连接失败因为: {e}")
        stop_event.set()
    shape0 = np.array(q.get().shape[:2])
    shape = np.array(q.get().shape[:2]) // 3
    shape = tuple(shape[::-1]) 
    shape0 = tuple(shape0[::-1])
    while not stop_event.is_set():
        frame = cv2.resize(q.get(), shape)
        try:
            client.send(pickle.dumps(frame))
        except Exception as e:
            print(f"连接失败因为: {e}")
            stop_event.set()
            break

        try:
            frame = cv2.resize(pickle.loads(client.recv()), shape0)
        except Exception as e:
            print(f"连接失败因为: {e}")
            stop_event.set()
            break
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_client(host, port):
    import keyboard
    mp.set_start_method(method='spawn')
    queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()
    processes = [
        mp.Process(target=queue_img_put, args=(queue,stop_event)),
        mp.Process(target=queue_img_get, args=(queue, "Local Camera", host, port,stop_event)),
    ]
    [process.start() for process in processes]
    while not stop_event.is_set():
        if keyboard.is_pressed('esc'):
            stop_event.set()
            print('Exit')
        try:
            time.sleep(0.01)
        except KeyboardInterrupt:
            stop_event.set()
            print('Exit')

    [process.terminate() for process in processes]
    [process.join() for process in processes]
    
def run_server(host, port):

    d = "./src/models/mmod_human_face_detector.dat"
    p = "./src/models/shape_predictor_5_face_landmarks.dat"
    detector = dlib.cnn_face_detection_model_v1(d)
    predictor = dlib.shape_predictor(p)
    elg_model = torch.load('./models/v0.2/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth', map_location=torch.device('cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elg_model = elg_model.to(device)    
    elg_model.eval()

    our_model = VGE()
    our_model = torch.nn.DataParallel(our_model)

    if torch.cuda.is_available():
        our_model.load_state_dict(torch.load('./model256_20.pth'))
        our_model.to(device)
        print('cuda avaliable')
    else:
        our_model.load_state_dict(torch.load('./model256_20.pth', map_location=torch.device('cpu')))
    our_model.eval()


    server_sock = Listener((host, port))
    print('Server Listening')
    conn = server_sock.accept()
    print('Server Accept')
    while True:
        data = pickle.loads(conn.recv())
        data = process(data, elg_model, our_model, detector, predictor)
        conn.send(pickle.dumps(data))

def get_ip_address(remote_server="8.8.8.8"):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((remote_server, 80))
    return s.getsockname()[0]


def run_program(server_host, server_port):
    if get_ip_address() == server_host:
        run_server(server_host, server_port)
    else:
        run_client(server_host, server_port)


if __name__ == '__main__':
    server_host = "replace content in " " with your server ip"
    server_port = 10086
    run_program(server_host, server_port)