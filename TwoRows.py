import os
import cv2
import torch
import gradio as gr
from ultralytics import YOLO
import numpy as np

#Two models , Ican't upload it because size > 25MB
plate_model = YOLO("/home/issa/Documents/PlateDetectionFINALLLY/runs/detect/train/weights/best.pt")
number_model = YOLO("/home/issa/Documents/NumbersLarge/runs/detect/train/weights/best.pt")

#Select Device {Cuda Please ^_^}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# create Folder to save process videos  
processed_folder = "processed_videos"
os.makedirs(processed_folder, exist_ok=True)

# Wiener filter "Please Keep it"   
def apply_wiener_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

# Processing 
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    processed_path = os.path.join(processed_folder, f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(processed_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plate_results = plate_model.predict(rgb_frame, device=DEVICE)

        for plate_result in plate_results:
            if plate_result.boxes:
                for box in plate_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_plate = rgb_frame[y1:y2, x1:x2]
                    deblurred_plate = apply_wiener_filter(cropped_plate)
                    number_results = number_model.predict(deblurred_plate, device=DEVICE)

                    upper_row, lower_row = [], []
                    for num_box in number_results[0].boxes:
                        _, ny1, _, ny2 = map(int, num_box.xyxy[0].tolist())
                        cy = (ny1 + ny2) // 2
                        if cy < (y2 - y1) / 2:
                            upper_row.append(num_box)
                        else:
                            lower_row.append(num_box)

                    upper_row = sorted(upper_row, key=lambda b: b.xyxy[0][0])
                    lower_row = sorted(lower_row, key=lambda b: b.xyxy[0][0])

                    combined_text = ''.join([number_model.names[int(box.cls[0])] for box in upper_row]) + ''.join([number_model.names[int(box.cls[0])] for box in lower_row])

                    cv2.putText(frame, combined_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return processed_path

# Gradio
with gr.Blocks() as demo:
    gr.Markdown("### 🚗 Automatic Video Plate Detection with Deblur")


    upload_input = gr.File(label="Upload Video", file_types=["video"], file_count="single")
    output_video = gr.Video(label="Processed Video")
    clear_button = gr.Button("Upload Another Video", visible=False)

    def process_and_show_video(file):
        processed_video = process_video(file.name)
        return processed_video, gr.update(visible=True)

    upload_input.change(fn=process_and_show_video, inputs=upload_input, outputs=[output_video, clear_button])

    def reset_inputs():
        return None, gr.update(visible=False)

    clear_button.click(fn=reset_inputs, outputs=[upload_input, clear_button])

demo.launch(share=True)
