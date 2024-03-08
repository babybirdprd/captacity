import subprocess
import tempfile
import math
import json
import sys
import cv2
import os

# Import faster-whisper in the specified format
from faster_whisper import WhisperModel

video_file = sys.argv[1]

current_dir = os.path.dirname(os.path.realpath(__file__))
output_file = os.path.join(current_dir, "with_transcript.avi")
temp_video_file = tempfile.NamedTemporaryFile(suffix=".avi").name
temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav").name


def write_line(text, frame, text_y, font, font_scale, white_color, black_color, thickness, border):
    # Calculate the position for centered text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # Center horizontally
    org = (text_x, text_y)  # Position of the text

    frame = cv2.putText(frame, text, org, font, font_scale, black_color, thickness + border * 2, cv2.LINE_AA)
    frame = cv2.putText(frame, text, org, font, font_scale, white_color, thickness, cv2.LINE_AA)

    return frame


def write_text(text, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    white_color = (255, 255, 255)
    black_color = (0, 0, 0)
    thickness = 10
    font_scale = 3
    border = 5
    margin = 18

    # Center multi-line text properly
    lines = text.splitlines()
    total_height = sum(cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines)
    text_y = (frame.shape[0] - total_height - (len(lines) - 1) * margin) // 2  # Center vertically

    for line in lines:
        frame = write_line(line, frame, text_y, font, font_scale, white_color, black_color, thickness, border)
        text_y += cv2.getTextSize(line, font, font_scale, thickness)[0][1] + margin

    return frame


def ffmpeg(command):
    return subprocess.run(command, capture_output=True)


def main():
    # Extract audio from video
    ffmpeg([
        'ffmpeg',
        '-y',
        '-i', video_file,
        temp_audio_file
    ])

    # Load faster-whisper model (replace "base" with desired model size)
    model = WhisperModel("base")  # Using the specified format

    # Transcribe the audio (adjust parameters as needed)
    transcription = model.transcribe(
        audio=temp_audio_file,
        word_timestamps=True,
        fp16=False,  # Set to True if you have a GPU
    )

    segments = transcription["segments"]

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    framerate = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_file, fourcc, framerate, (int(cap.get(3)), int(cap.get(4))))

    time = 0.0
    lookahead = 2.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        text_to_use = ""
        time_chunk = math.ceil(time / lookahead) * lookahead

        for segment in segments:
            for word in segment["words"]:
                if word["start"] >= time_chunk - lookahead and word["start"] < time_chunk:
                    text_to_use += word["word"]

        frame = write_text(text_to_use, frame)
        out.write(frame)

        time += 1 / framerate

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Close all OpenCV windows (if any)
    cv2.destroyAllWindows()

    ffmpeg([
        'ffmpeg',
        '-y',
        '-i', temp_video_file,
        '-i', temp_audio_file,
        '-map', '0:v',  # Map video from the first input
        '-map', '1:a',  # Map audio from the second input
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'aac',  # AAC audio codec
        '-strict', 'experimental',
        output_file
    ])


if __name__ == "__main__":
    main()
