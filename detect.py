import cv2
from ultralytics import YOLO
import argparse
import os
from datetime import datetime

# Default confidence threshold global variable for UI adjustments
CONFIDENCE_THRESHOLD = 0.50

def nothing(x):
    """Callback function for trackbar."""
    pass

def main(source, save_dir):
    # Determine absolute model path so it works regardless of terminal CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", "weapon_detection_model7", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Custom model not found at '{model_path}'. Using default 'yolov8m.pt' for demonstration.")
        model_path = "yolov8m.pt"
        
    # Load model
    print(f"Loading model '{model_path}'...")
    model = YOLO(model_path)
    
    # The augmented Roboflow dataset uses 'pistol' instead of 'gun'
    custom_classes = ['knife', 'gun', 'other', 'pistol'] # Our custom classes
    
    is_image = str(source).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    
    # Open video capture stream OR load static image
    if is_image:
        static_frame = cv2.imread(source)
        if static_frame is None:
            print(f"❌ Error: Could not load image '{source}'")
            return
        cap = None
    else:
        if source == '0':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source '{source}'")
            return

    # Create UI Window
    window_name = 'AI Weapon Detection'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Confidence', window_name, int(CONFIDENCE_THRESHOLD * 100), 100, nothing)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("✅ System ready. Starting live inference...")
    print("ℹ️ Press 'q' in the video window to quit.")

    frame_count = 0
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read the frame.")
                break
        else:
            # If it's a static image, just copy the original frame every iteration.
            # This allows the user to play with the Confidence Slider in real-time!
            frame = static_frame.copy()

        # Check if user closed the window via the 'X' button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user.")
            break

        # Get the threshold from the UI slider
        conf_thresh = cv2.getTrackbarPos('Confidence', window_name) / 100.0
        # Prevent completely 0 confidence to avoid freezing
        conf_thresh = max(0.01, conf_thresh) 
        
        # Run YOLO inference
        results = model(frame, conf=conf_thresh, verbose=False)
        
        weapon_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Retrieve class ID
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Check for "weapon" (modify this according to exact model labels trained)
                # Here we assume the user trained with ['knife', 'gun', 'other']
                # If falling back to yolov8n.pt (COCO), knife is 'knife', gun is not explicitly there but 'scissors' etc. exist
                if class_name in custom_classes or class_name == 'knife':
                    weapon_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw a red bounding box for critical alerts
                    color = (0, 0, 255) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name.upper()} {conf:.2f}"
                    # Label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Trigger actions on detection
        if weapon_detected:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ⚠️ ALERT: Weapon detected!")
            
            # Save frame every 30 frames (~1 sec) to prevent flooding disk
            if frame_count % 30 == 0:
                file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"alert_{file_ts}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📸 Saved alert frame: {save_path}")

        # Show the processed frame
        cv2.imshow(window_name, frame)
        frame_count += 1

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release hardware & destroy UI
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Weapon Detection System")
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam, or absolute path to a video file.')
    parser.add_argument('--save_dir', type=str, default='detected_frames', help='Directory to store frames when a weapon is detected.')
    args = parser.parse_args()
    
    main(args.source, args.save_dir)
