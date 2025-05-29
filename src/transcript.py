import json
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound

def extract_video_id(url_or_id: str) -> str:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound

def extract_video_id(url_or_id: str) -> str:
    import re
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id

def fetch_youtube_transcript(video_id_or_url: str, save_prefix: str = "data5") -> str | None:
    video_id = extract_video_id(video_id_or_url)

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])
        raw_data = transcript.fetch()  # raw_data is FetchedTranscript object (iterable of FetchedTranscriptSnippet)

        # Convert to list of dicts:
        raw_data_list = [
            {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
            for snippet in raw_data
        ]

    except TranscriptsDisabled:
        print("Transcript is disabled for this video.")
        return None
    except VideoUnavailable:
        print("Video is unavailable.")
        return None
    except NoTranscriptFound:
        print("No transcript found for this video.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    os.makedirs("data/raw", exist_ok=True)

    json_path = f"data/raw/{save_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data_list, f, ensure_ascii=False, indent=4)  # now it's a list of dicts

    full_text = " ".join([snippet["text"] for snippet in raw_data_list]).replace("\n", " ")

    txt_path = f"data/raw/{save_prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Transcript saved to {json_path} and {txt_path}")
    return full_text


