import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable

def extract_video_id(url_or_id: str) -> str:

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id

def fetch_youtube_transcript(video_id_or_url: str, save_prefix: str = "data5") -> str | None:

    video_id = extract_video_id(video_id_or_url)

    try:
        raw_data = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        print("Transcript is disabled for this video.")
        return None
    except VideoUnavailable:
        print("Video is unavailable.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    json_path = f"data/raw/{save_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    full_text = " ".join([segment.get("text", "") for segment in raw_data]).replace("\n", " ")

    txt_path = f"data/raw/{save_prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text
