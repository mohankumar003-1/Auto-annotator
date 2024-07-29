from ultralytics import YOLO
import cv2
import os

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your YOLOv8 weights file

# Define the class to detect
CLASS_TO_DETECT = 'person'
SHOW_ACCURACY = 85

# Available classes (Index 0 corresponds to 'person')
CLASSES_AVAIL = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

def get_index(classes_avail, class_to_detect):
    class_index = None
    for idx, label in classes_avail.items():
        if label == class_to_detect:
            class_index = idx
            break
    if class_index is None:
        raise ValueError(f"Class '{class_to_detect}' not found in classes_avail.")
    return class_index

def write_annotation(annotations, annotation_file_path):
    # Save annotations to text file
    with open(annotation_file_path, 'w') as file:
        for annotation in annotations:
            file.write(annotation + '\n')
    print(f'Annotations saved to {annotation_file_path}')

def process_image(image, image_path, output_dir, show_accuracy):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform inference
    results = model.predict(image_rgb)

    # Prepare for annotation text file
    annotations = []

    class_index = get_index(CLASSES_AVAIL, CLASS_TO_DETECT)
    # Process results
    for result in results:
        # Extract bounding boxes, scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Filter and draw bounding boxes for the specified class
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if score < show_accuracy / 100:
                continue

            if int(cls_id) == class_index:
                x1, y1, x2, y2 = box.astype(int)
                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f'{CLASS_TO_DETECT} {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Append to annotations
                annotations.append(f"{class_index} {score:.2f} {x1} {y1} {x2} {y2}")

    # Save annotations to a file
    annotation_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    write_annotation(annotations, annotation_file_path)

    # Save the resulting image
    output_image_path = os.path.join(output_dir, 'predicted_image.jpg')
    cv2.imwrite(output_image_path, image)
    print(f'Predicted image saved to {output_image_path}')



# # Example usage
# input_dir= '/home/mohan-si2708/ownprojects/auto_annotater/input_dir/'
# output_dir = '/home/mohan-si2708/ownprojects/auto_annotater/annotation/'



def process_all_images(input_dir, output_dir, show_accuracy):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            process_image(image,image_path, output_dir, show_accuracy)



# process_all_images(input_dir,output_dir,SHOW_ACCURACY)