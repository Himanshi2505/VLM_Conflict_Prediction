import numpy as np

previous_positions = {}

def calculate_speed_acceleration(track_id, x, y, time_stamp):
    global previous_positions
    if track_id in previous_positions:
        prev_x, prev_y, prev_time = previous_positions[track_id]
        dt = time_stamp - prev_time
        if dt > 0:
            speed = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) / dt * 3.6  # Convert m/s to km/h
            acceleration = (speed - previous_positions[track_id][2]) / dt  # m/s²
        else:
            speed, acceleration = 0, 0
    else:
        speed, acceleration = 0, 0

    previous_positions[track_id] = (x, y, time_stamp, speed)
    return speed, acceleration, 0  # Assuming lateral acceleration = 0 for simplicity
