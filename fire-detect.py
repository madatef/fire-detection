import cv2
import numpy as np
import torch
from mss import mss
from ultralytics import YOLO
import time
import base64  # Added for Gemini image encoding
import io       # Added for image buffering
import requests # Added for Gemini API calls
from PIL import Image  # Added for image conversion

# === Gemini Validation State Tracking ===
last_fire_validation_time = 0
last_smoke_validation_time = 0
validation_interval_sec = 300  # 5 minutes

# === Gemini Flash Validation Function ===
def validate_with_gemini(image_np, label):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyCCFrl-nUDpN0iafFdS453O_flt6W5Mxsw"  # TODO: Move to env

    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode()

    prompt = f"Is there visible {label} in this image? Answer with only 'yes' or 'no'."

    data = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64
                }}
            ]
        }]
    }

    try:
        res = requests.post(url, json=data)
        response_text = res.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        return response_text == "yes"
    except Exception as e:
        print("Gemini validation error:", e)
        return False

def send_whatsapp_message(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = buffer.tobytes()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    receivers = ["201143117851", "201028968199"]  # TODO: Replace with your WhatsApp numbers
    try:
        upload_res = requests.post(
            "https://api.dragify.ai/general/uploader-secret",
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            headers={"x-api-key": "uLdiVUo67043G997lIua"}, # TODO: Move to env

        )
        upload_res.raise_for_status()
        image_url = upload_res.json().get("url")
        url = "https://waapi.app/api/v1/instances/65916/client/action/send-media"
        headers = { "Authorization": "Bearer grj8R8YuWGTnvtvo8knrPUrjeg0MsYOBPYOTZEEX8a6e362b"} # TODO: Move to env
        for receiver in receivers:
            data = {
                "chatId": f"{receiver}@c.us", #TODO: replace with your name
                "mediaCaption": f"Fire or smoke detected in this frame! Time: {ts}",
                "mediaUrl": image_url,
            }
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            status = response.json().get("status")
            if status == "success":
                print(f"WhatsApp message sent to {receiver} successfully!")
            else:
                print(f"Failed to send WhatsApp message: {response.json()}")
        
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

def get_center_screen_coordinates(width=608, height=608):
    with mss() as sct:
        monitor = sct.monitors[1]
        center_left = (monitor["width"] - width) // 2
        center_top = (monitor["height"] - height) // 2
        return center_left, center_top, width, height


def letterbox(image, expected_size):
    ih, iw = image.shape[:2]
    eh, ew = expected_size, expected_size
    scale = min(ew/iw, eh/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))
    return image_resized


def draw_bounding_box(frame, box, class_name, confidence):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    label = f"{class_name} {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_start_x = x1
    text_start_y = y1 - text_height if y1 > text_height + baseline else y1 + text_height
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(frame, (text_start_x, text_start_y - text_height), (text_start_x + text_width + baseline, text_start_y + text_height), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (text_start_x + baseline, text_start_y + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def detect_fire_and_smoke(model, frame, time_window=1000, detection_threshold=1):
    global last_fire_validation_time, last_smoke_validation_time

    results = model.predict(frame, verbose=False)
    fire_detected = False
    smoke_detected = False
    result = results[0]
    confidence_threshold = 0.001
    fire_detection_times = []

    for i in range(len(result.boxes.data)):
        cls_id = int(result.boxes.cls[i].item())
        conf = result.boxes.conf[i].item()
        if conf > confidence_threshold:
            bbox = result.boxes.xyxy[i].tolist()
            class_name = result.names[cls_id].lower()
            draw_bounding_box(frame, bbox, class_name, conf)
            if class_name == 'fire':
                fire_detection_times.append(time.time())
                fire_detection_times = [t for t in fire_detection_times if time.time() - t <= time_window]
                if len(fire_detection_times) >= detection_threshold:
                    fire_detected = True
            elif class_name == 'smoke':
                smoke_detected = True

    current_time = time.time()
    if fire_detected and current_time - last_fire_validation_time > validation_interval_sec:
        if validate_with_gemini(frame, "fire"):
            send_whatsapp_message(frame)
            print("\n🔥 [Gemini confirmed] Fire detected!")
            last_fire_validation_time = current_time

    if smoke_detected and current_time - last_smoke_validation_time > validation_interval_sec:
        if validate_with_gemini(frame, "smoke"):
            send_whatsapp_message(frame)
            print("\n💨 [Gemini confirmed] Smoke detected!")
            last_smoke_validation_time = current_time

    return fire_detected, smoke_detected


def capture_cctv_stream(model, rtsp_url, max_retries=3, retry_delay=5):
    retry_count = 0
    while retry_count < max_retries:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

        if not cap.isOpened():
            print(f"Failed to connect to CCTV stream. Retry {retry_count + 1}/{max_retries}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                raise IOError(f"Cannot connect to CCTV stream after {max_retries} attempts: {rtsp_url}")

        print(f"Connected to CCTV stream: {rtsp_url}")

        try:
            frame_count = 0
            last_fps_time = time.time()
            fps_counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Connection lost or stream ended. Attempting to reconnect...")
                    break

                frame_count += 1
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    print(f"CCTV Stream FPS: {fps:.2f}")
                    last_fps_time = current_time
                    fps_counter = 0

                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                detect_fire_and_smoke(model, frame)

                cv2.imshow('CCTV Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting CCTV capture...")
                    return

        except Exception as e:
            print(f"Error during stream processing: {e}")
            break
        finally:
            cap.release()

        retry_count += 1
        if retry_count < max_retries:
            print(f"Attempting to reconnect... ({retry_count}/{max_retries})")
            time.sleep(retry_delay)

    print(f"Failed to maintain connection after {max_retries} attempts")
    cv2.destroyAllWindows()


def capture_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Connected to webcam")
    
    try:
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_count += 1
            fps_counter += 1
            current_time = time.time()
            
            # FPS calculation and display
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                print(f"Webcam FPS: {fps:.2f}")
                last_fps_time = current_time
                fps_counter = 0

            # Ensure frame is in BGR format for YOLO processing
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Detect fire and smoke with Gemini validation
            detect_fire_and_smoke(model, frame)
            
            cv2.imshow('WebCam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam capture...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def screen_capture(model):
    model_input_size = 640
    left, top, width, height = get_center_screen_coordinates(width=model_input_size, height=model_input_size)
    mon = {
        "top": top,
        "left": left,
        "width": width,
        "height": height
    }

    print(f"Screen capture area: {width}x{height} at ({left}, {top})")
    
    with mss() as sct:
        try:
            frame_count = 0
            last_fps_time = time.time()
            fps_counter = 0
            
            while True:
                sct_img = sct.grab(mon)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                frame_count += 1
                fps_counter += 1
                current_time = time.time()
                
                # FPS calculation and display
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    print(f"Screen Capture FPS: {fps:.2f}")
                    last_fps_time = current_time
                    fps_counter = 0
                
                # Detect fire and smoke with Gemini validation
                detect_fire_and_smoke(model, frame)
                
                # Display the frame (convert back to BGRA for display)
                cv2.imshow('Screen', cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting screen capture...")
                    break
        finally:
            cv2.destroyAllWindows()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "runs\\detect\\train3\\weights\\best.pt"
    model = YOLO(model_path).to(device)
    
    print("Press 'q' to quit\n")

    capture = "webcam"  # Options: "screen", "webcam", "cctv"

    if capture == "screen":
        screen_capture(model)
    elif capture == "webcam":
        capture_webcam(model)
    elif capture == "cctv":
        # Example RTSP URLs for common camera brands:
        # Generic: rtsp://username:password@ip_address:port/stream
        # Hikvision: rtsp://username:password@ip_address:554/Streaming/Channels/101
        # Dahua: rtsp://username:password@ip_address:554/cam/realmonitor?channel=1&subtype=0
        # Axis: rtsp://username:password@ip_address:554/axis-media/media.amp
        
        rtsp_url = "rtsp://username:password@ip_address:554/Streaming/Channels/101"  # TODO: Replace with your camera's actual RTSP URL, env
        
        # You can also try without authentication if camera doesn't require it:
        # rtsp_url = "rtsp://192.168.1.100:554/stream"
        try:
            capture_cctv_stream(model, rtsp_url)
        except Exception as e:
            print(f"CCTV capture failed: {e}")


if __name__ == "__main__":
    main()


