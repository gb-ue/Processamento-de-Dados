import os
import whisper
from moviepy import VideoFileClip, AudioFileClip


def audio_fetch(audio, file_type):
  str_audio = os.path.splitext(os.path.basename(audio))[0]
  result = whisper.transcribe(audio = audio, language="pt")
  folder_type = ''
  if file_type == "audio":
      folder_type = 'Audios-Drive-Extraidos/'
  elif file_type == "video":
      folder_type = 'Videos-Extraidos/'
  with open(folder_type + str_audio + ".srt", 'w', encoding="utf-8") as file:
    for i, seg in enumerate(result['segments'], start=1):
      start = seg['start']
      end = seg['end']

      def format_time(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s = int(s % 60)
        ms = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

      file.write(f"{i}\n")
      file.write(f"{format_time(start)} --> {format_time(end)}\n")
      file.write(f"{seg['text']}\n\n")

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