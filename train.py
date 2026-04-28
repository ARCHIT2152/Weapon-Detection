from ultralytics import YOLO

def main():
    # Load a pre-trained model as a starting point (Medium model for higher accuracy)
    model = YOLO("yolov8m.pt") 

    print("Starting training on weapon detection dataset...")
    
    # Train the model using our custom dataset
    results = model.train(
        data="weapon detection.v1i.yolov8 (1)/data.yaml",
        epochs=150,                  # Increased epochs for maximum learning
        imgsz=640,                  # Image size
        batch=16,                   # Batch size
        name="weapon_detection_model",
        device="0",               # Use "cpu" for laptop, change to "0" if you have an Nvidia GPU
        patience=25,                # Early stopping parameter (stops if no improvement for 25 epochs)
        save=True,                 # Save weights
        cache  = True,
        workers = 4

    )
    
    print("\nTraining complete! Results and weights saved to 'runs/detect/weapon_detection_model'.")
    print("Your best model weights are located at: 'runs/detect/weapon_detection_model/weights/best.pt'")

if __name__ == '__main__':
    main()
