from multiprocessing import Manager

def initialize_shared_data():
    manager = Manager()
    shared_data = manager.dict()

    shared_data["track_positions"] = manager.dict()
    shared_data["last_speeds"] = manager.dict()
    shared_data["last_seen"] = manager.dict()
    shared_data["host_speed"] = 0.0
    shared_data["camera_frames"] = manager.dict()
    shared_data["camera_display_frame"] = None
    shared_data["bev_display_frame"] = None

    # ?? Kill switch
    shared_data["exit_flag"] = manager.Value('b', False)

    return shared_data
