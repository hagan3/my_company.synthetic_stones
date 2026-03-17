import socket
import json
import time
import csv
import os

UDP_IP = "127.0.0.1"
UDP_PORT = 12345

CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "busy-debug_14_detections.csv")

def get_stone_name(track_id, class_name):
    """
    Assign a stable stone name based on track_id and class_name.
    Examples: stone_y1, stone_r1, stone_y2, stone_r2, etc.
    """
    # Extract color prefix: 'yellow_stone' -> 'y', 'red_stone' -> 'r'
    if 'yellow' in class_name.lower():
        color_prefix = 'y'
    elif 'red' in class_name.lower():
        color_prefix = 'r'
    else:
        color_prefix = 'u'  # unknown
    
    return f"stone_{color_prefix}{track_id}"

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Reading CSV from: {CSV_FILE}")
    
    try:
        with open(CSV_FILE, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            last_timestamp = 0.0
            current_frame_stones = {}
            
            for row in reader:
                timestamp_sec = float(row['timestamp_sec'])
                track_id = int(row['track_id']) if row['track_id'] != '-1' else -1
                class_name = row['class_name']
                
                # Skip non-stone objects (e.g., hog_line, house) and invalid track_ids
                if track_id < 0 or 'stone' not in class_name.lower():
                    continue
                
                # Get bounding box coordinates (already in centimeters)
                x1 = float(row['x1'])
                y1 = float(row['y1'])
                x2 = float(row['x2'])
                y2 = float(row['y2'])
                
                # Calculate center in centimeters (no conversion needed)
                x_cm = (x1 + x2) / 2.0
                y_cm = (y1 + y2) / 2.0
                
                # Get stone name
                stone_name = get_stone_name(track_id, class_name)
                
                # Group stones by timestamp
                if timestamp_sec != last_timestamp and current_frame_stones:
                    # Send the previous frame's data
                    stones_list = list(current_frame_stones.values())
                    message = json.dumps(stones_list).encode()
                    sock.sendto(message, (UDP_IP, UDP_PORT))
                    print(f"[t={last_timestamp:.3f}] Sent {len(stones_list)} stones")
                    
                    # Sleep to simulate real-time playback
                    time.sleep(timestamp_sec - last_timestamp)
                    
                    # Reset for new frame
                    current_frame_stones = {}
                    last_timestamp = timestamp_sec
                
                # Add or update stone in current frame
                current_frame_stones[stone_name] = {
                    "id": stone_name,
                    "x_cm": x_cm,
                    "y_cm": y_cm
                }
            
            # Send the last frame
            if current_frame_stones:
                stones_list = list(current_frame_stones.values())
                message = json.dumps(stones_list).encode()
                sock.sendto(message, (UDP_IP, UDP_PORT))
                print(f"[t={last_timestamp:.3f}] Sent {len(stones_list)} stones (final frame)")
        
        print("CSV replay completed!")
        
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {CSV_FILE}")
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
