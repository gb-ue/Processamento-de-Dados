import os
import whisper
from moviepy.utils import VideoFileClip, AudioFileClip

os.environ["PATH"] += os.pathsep + os.path.abspath("./ffmpeg/bin")

whisper = whisper.load_model("large")

def format_time(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s = int(s % 60)
        ms = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

def audio_fetch(file):
    if not has_audio(file, "audio") and not has_audio(file, "video"):
        return "O arquivo não contém áudio."
    result = whisper.transcribe(audio=file, language="pt")
    output = ""  

    for i, seg in enumerate(result['segments'], start=1):
        start = seg['start']
        end = seg['end']
        text = seg['text']

        output += f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
      
    return output

def has_audio(file_path, type):
    if type == "audio":
        try:
            clip = AudioFileClip(file_path)
            return clip is not None
        except Exception:
            return False
    elif type == "video":
        try:
            clip = VideoFileClip(file_path)
            return clip.audio is not None
        except Exception:
            return False
        
print(audio_fetch("Integrantes de facção enviam áudios para mãe de adolescente encontrada morta.mp3"))