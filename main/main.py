
import multiprocessing
import cv2
import numpy as np
import time
from threading import Thread
from config import VIDEO_URLS
from camera import camera_process, get_next_test_folder
from map_yolo_bev_LR_6 import run_mapping_process
from tracker import initialize_shared_data
import obd


# UI Constants
EXIT_BTN_POS = (700, 10)
EXIT_BTN_SIZE = (40, 30)
CAROUSEL_BTN_SIZE = (120, 40)
TOP_LEFT_BTN_POS = (10, 10)
TOP_LEFT_BTN_SIZE = (40, 30)
SAVE_BTN_POS = (700, 360)
SAVE_BTN_SIZE = (80, 30)

CAMERA_VIEW = 0
BEV_VIEW = 1
ORIENTATION_VIEW = 2

current_orientation_camera = [0]
orientation_mode_active = [False]
bev_flip_enabled = [False]

def show_loading_screen(duration=2.0):
    loading = np.zeros((400, 800, 3), dtype=np.uint8)
    cv2.putText(loading, "Initializing system...", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    cv2.putText(loading, "Please wait", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    start_time = time.time()
    while time.time() - start_time < duration:
        cv2.imshow("Unified View", loading)
        if cv2.waitKey(100) == 27:  
            break

def handle_mouse(event, x, y, flags, param):
    shared_data, current_view_mode = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if EXIT_BTN_POS[0] <= x <= EXIT_BTN_POS[0] + EXIT_BTN_SIZE[0] and EXIT_BTN_POS[1] <= y <= EXIT_BTN_POS[1] + EXIT_BTN_SIZE[1]:
            print("[UI] Exit clicked")
            shared_data["exit_flag"].value = True

        if TOP_LEFT_BTN_POS[0] <= x <= TOP_LEFT_BTN_POS[0] + TOP_LEFT_BTN_SIZE[0] and TOP_LEFT_BTN_POS[1] <= y <= TOP_LEFT_BTN_POS[1] + TOP_LEFT_BTN_SIZE[1]:
            if current_view_mode[0] == CAMERA_VIEW and not orientation_mode_active[0]:
                print("[UI] Entered Orientation Mode")
                current_view_mode[0] = ORIENTATION_VIEW
                current_orientation_camera[0] = 0
                orientation_mode_active[0] = True
            elif current_view_mode[0] == BEV_VIEW:
                bev_flip_enabled[0] = not bev_flip_enabled[0]
                print(f"[UI] Flip BEV toggled: {bev_flip_enabled[0]}")

        if current_view_mode[0] in (CAMERA_VIEW, BEV_VIEW):
            bx, by = (640, 320) if current_view_mode[0] == CAMERA_VIEW else (40, 320)
            if bx <= x <= bx + CAROUSEL_BTN_SIZE[0] and by <= y <= by + CAROUSEL_BTN_SIZE[1]:
                current_view_mode[0] = BEV_VIEW if current_view_mode[0] == CAMERA_VIEW else CAMERA_VIEW
                print(f"[UI] Switched to {'BEV' if current_view_mode[0] == BEV_VIEW else 'CAMERA'} view")

        if current_view_mode[0] == ORIENTATION_VIEW:
            if SAVE_BTN_POS[0] <= x <= SAVE_BTN_POS[0] + SAVE_BTN_SIZE[0] and SAVE_BTN_POS[1] <= y <= SAVE_BTN_POS[1] + SAVE_BTN_SIZE[1]:
                print(f"[ORIENTATION] Camera {current_orientation_camera[0]+1} saved!")
                current_orientation_camera[0] += 1
                if current_orientation_camera[0] >= 4:
                    print("[ORIENTATION] All cameras calibrated. Returning to CAMERA VIEW.")
                    current_view_mode[0] = CAMERA_VIEW
                    orientation_mode_active[0] = False

def draw_orientation_markings(frame, camera_index):
    h, w = frame.shape[:2]

    if camera_index == 0:  # FRONT camera
        pt_left = (0, int(h*0.70))    
        pt_middle = (int(w*0.5), int(h*0.80))  
        pt_right = (w, int(h*0.70))    

        cv2.line(frame, pt_left, pt_middle, (0, 255, 0), 2)
        cv2.line(frame, pt_middle, pt_right, (0, 255, 0), 2)

        for pt in [pt_left, pt_middle, pt_right]:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    elif camera_index == 1:  # BACK camera
        pt_left = (0, int(h*0.60))    
        pt_middle = (int(w*0.5), int(h*0.65))
        pt_right = (w, int(h*0.60))   

        cv2.line(frame, pt_left, pt_middle, (0, 255, 0), 2)
        cv2.line(frame, pt_middle, pt_right, (0, 255, 0), 2)

        for pt in [pt_left, pt_middle, pt_right]:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    elif camera_index == 2:  # LEFT camera
        pt_left = (0, int(h*0.60))   
        pt_middle = (int(w*0.55), int(h*0.9))  
        pt_right = (w, int(h*0.80))   
        
        cv2.line(frame, pt_left, pt_middle, (0, 255, 0), 2)
        cv2.line(frame, pt_middle, pt_right, (0, 255, 0), 2)

        for pt in [pt_left, pt_middle, pt_right]:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    elif camera_index == 3:  # RIGHT camera
        pt_left = (0, int(h*0.80))  
        pt_middle = (int(w*0.45), int(h*0.9))  
        pt_right = (w, int(h*0.60))  

        cv2.line(frame, pt_left, pt_middle, (0, 255, 0), 2)
        cv2.line(frame, pt_middle, pt_right, (0, 255, 0), 2)

        for pt in [pt_left, pt_middle, pt_right]:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    return frame

def unified_display_loop(shared_data):
    current_view_mode = [CAMERA_VIEW]
    cv2.namedWindow("Unified View")
    cv2.setMouseCallback("Unified View", handle_mouse, param=(shared_data, current_view_mode))

    fade_ratio = 0.15
    last_frame = np.zeros((400, 800, 3), dtype=np.uint8)

    while not shared_data["exit_flag"].value:
        if current_view_mode[0] == CAMERA_VIEW:
            frame = shared_data.get("camera_display_frame", None)
        elif current_view_mode[0] == BEV_VIEW:
            frame = shared_data.get("bev_display_frame", None)
            if frame is not None and bev_flip_enabled[0]:
                frame = cv2.flip(frame, 1)
        else:
            cam_idx = current_orientation_camera[0]
            frame = shared_data["camera_frames"].get(cam_idx, None)
            if frame is not None:
                frame = cv2.resize(frame, (800,400))
                frame = draw_orientation_markings(frame, cam_idx)

        if frame is None:
            frame = np.zeros((400, 800, 3), dtype=np.uint8)

        blended = cv2.addWeighted(last_frame, fade_ratio, frame, 1 - fade_ratio, 0)
        last_frame = blended.copy()

        cv2.rectangle(blended, EXIT_BTN_POS, (EXIT_BTN_POS[0] + EXIT_BTN_SIZE[0], EXIT_BTN_POS[1] + EXIT_BTN_SIZE[1]), (0, 0, 255), -1)
        cv2.putText(blended, "X", (EXIT_BTN_POS[0] + 12, EXIT_BTN_POS[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if current_view_mode[0] == CAMERA_VIEW and not orientation_mode_active[0]:
            cv2.rectangle(blended, TOP_LEFT_BTN_POS, (TOP_LEFT_BTN_POS[0] + TOP_LEFT_BTN_SIZE[0], TOP_LEFT_BTN_POS[1] + TOP_LEFT_BTN_SIZE[1]), (150, 150, 150), -1)
            cv2.putText(blended, "OM", (TOP_LEFT_BTN_POS[0]+5, TOP_LEFT_BTN_POS[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        if current_view_mode[0] == BEV_VIEW:
            cv2.rectangle(blended, TOP_LEFT_BTN_POS, (TOP_LEFT_BTN_POS[0] + TOP_LEFT_BTN_SIZE[0], TOP_LEFT_BTN_POS[1] + TOP_LEFT_BTN_SIZE[1]), (0,150,150), -1)
            flip_text = "FLIP BEV" if not bev_flip_enabled[0] else "UNFLIP"
            cv2.putText(blended, flip_text, (TOP_LEFT_BTN_POS[0]+5, TOP_LEFT_BTN_POS[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        if current_view_mode[0] in (CAMERA_VIEW, BEV_VIEW):
            bx, by = (640, 320) if current_view_mode[0] == CAMERA_VIEW else (40, 320)
            label = "? BEV VIEW" if current_view_mode[0] == CAMERA_VIEW else "? CAMERA VIEW"
            cv2.rectangle(blended, (bx, by), (bx + CAROUSEL_BTN_SIZE[0], by + CAROUSEL_BTN_SIZE[1]), (50, 50, 200), -1)
            cv2.putText(blended, label, (bx + 5, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if current_view_mode[0] == ORIENTATION_VIEW:
            cv2.rectangle(blended, SAVE_BTN_POS, (SAVE_BTN_POS[0]+SAVE_BTN_SIZE[0], SAVE_BTN_POS[1]+SAVE_BTN_SIZE[1]), (0,200,0), -1)
            cv2.putText(blended, "SAVE", (SAVE_BTN_POS[0]+5, SAVE_BTN_POS[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Unified View", blended)
        if cv2.waitKey(1) & 0xFF == 27:
            shared_data["exit_flag"].value = True
            break

    cv2.destroyAllWindows()

def camera_display_process(shared_data, num_cams):
    cam_w, cam_h = 320, 240
    while not shared_data["exit_flag"].value:
        frames = []
        for idx in range(num_cams):
            frame = shared_data["camera_frames"].get(idx, None)
            if frame is not None:
                frame = cv2.resize(frame, (cam_w, cam_h))
            else:
                frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            frames.append(frame)

        if num_cams == 1:
            display_frame = frames[0]
        elif num_cams == 2:
            display_frame = np.hstack(frames)
        elif num_cams <= 4:
            top_row = np.hstack(frames[:2])
            bottom_row = np.hstack(frames[2:4]) if len(frames) > 2 else np.zeros_like(top_row)
            display_frame = np.vstack((top_row, bottom_row))
        else:
            raise ValueError("Max 4 cameras supported")

        shared_data["camera_display_frame"] = cv2.resize(display_frame, (800, 400))
        time.sleep(0.03)

def start_video_sources(video_urls, queues, shared_data):
    for i, url in enumerate(video_urls):
        if url.startswith("http"):
            thread = Thread(target=read_mjpeg_stream, args=(url, queues[i], i, shared_data), daemon=True)
        else:
            thread = Thread(target=read_video_file, args=(url, queues[i], i, shared_data), daemon=True)
        thread.start()

def read_mjpeg_stream(url, queue, cam_index, shared_data, reconnect_delay=5):
    while not shared_data["exit_flag"].value:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[Camera {cam_index}] Failed to open. Retrying...")
            time.sleep(reconnect_delay)
            continue

        while not shared_data["exit_flag"].value:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if not queue.full():
                queue.put(frame)
        cap.release()

def read_video_file(path, queue, cam_index, shared_data):
    cap = cv2.VideoCapture(path)
    while cap.isOpened() and not shared_data["exit_flag"].value:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if not queue.full():
            queue.put(frame)
    cap.release()

def obd_process(shared_data):
    connection = obd.OBD("/dev/ttyUSB0", baudrate=38400)
    while not shared_data["exit_flag"].value:
        try:
            if connection.is_connected():
                speed_resp = connection.query(obd.commands.SPEED)
                if not speed_resp.is_null():
                    shared_data["host_speed"] = speed_resp.value.to("km/h").magnitude
        except:
            shared_data["host_speed"] = 0.0
        time.sleep(0.1)

def main():
    show_loading_screen()
    num_cams = len(VIDEO_URLS)
    queues = [multiprocessing.Queue(maxsize=1) for _ in range(num_cams)]
    shared_data = initialize_shared_data()
    test_folder = get_next_test_folder()

    start_video_sources(VIDEO_URLS, queues, shared_data)

    processes = []
    for idx in range(num_cams):
        p = multiprocessing.Process(target=camera_process, args=(idx, queues[idx], shared_data, test_folder))
        p.start()
        processes.append(p)

    mapping_process = multiprocessing.Process(target=run_mapping_process, args=(queues, shared_data))
    mapping_process.start()
    processes.append(mapping_process)

    obd_proc = multiprocessing.Process(target=obd_process, args=(shared_data,))
    obd_proc.start()
    processes.append(obd_proc)

    display_proc = multiprocessing.Process(target=camera_display_process, args=(shared_data, num_cams))
    display_proc.start()
    processes.append(display_proc)

    unified_display_loop(shared_data)

    for p in processes:
        p.terminate()
        p.join()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
