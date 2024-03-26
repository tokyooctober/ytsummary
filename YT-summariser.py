#
# Thanks to Ana Bildea for the base code. https://medium.com/@anna.bildea
#
import whisper
from pytube import YouTube
from transformers import pipeline
import os
from typing import List
import logging
import sys
import nltk
import argparse

logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


URL = "https://www.youtube.com/watch?v=QxMa79dvq-w"
VIDEO_NAME="MacroVoice-328"

# parser = argparse.ArgumentParser()
# parser.add_argument("--url", type=str, help="provide YouTube Url")
# parser.add_argument("--name", type=str, help="provide name of saved summary")

# args = parser.parse_args()

# if args.url:
#     URL = args.url
# if args.name:
#     VIDEO_NAME = args.name


# print(f"URL: {URL}")

def download_audio_from_youtube(url: str, video_name: str) -> str:
    #"""Download the audio from a YouTube video and save it as an MP3 file."""
    video_url= YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename

def load_whisper_model(model_name: str = "medium"):
    """Load the medium multilingual Whisper model."""
    return whisper.load_model(model_name)

def transcribe_audio_to_text(model, audio_path: str, language: str = "English"):
    """Transcribe the audio using the Whisper model."""
    return model.transcribe(audio_path, fp16=False, language=language)

def save_text_to_file(text: str, file_name: str):
    """Save the transcribed text to a file."""
    try:
        with open(file_name, "w+") as file:
            file.write(text)
    except (IOError, OSError, FileNotFoundError, PermissionError) as e:
        logging.debug(f"Error in file operation: {e}")

def get_text(url: str, video_name: str) -> None:
    model = load_whisper_model()
    audio_path = download_audio_from_youtube(url, video_name)
    result = transcribe_audio_to_text(model, audio_path)
    save_text_to_file(result["text"], video_name + ".txt")

    
get_text(url=URL, video_name=VIDEO_NAME)
   


nltk.download('punkt')

def read_file(file_name: str) -> str:
    try:
        with open(file_name + ".txt", "r", encoding="utf8") as file:
            return file.read()
    except FileNotFoundError as e:
        logging.error(f"{e}: File '{file_name}.txt' not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return ""

def split_text_into_chunks(document: str, max_tokens: int) -> List[str]:
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence)

            if current_length + sentence_length < max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [sentence], sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        return []

long_text = read_file(VIDEO_NAME)
if long_text:
    text_chunks = split_text_into_chunks(long_text, max_tokens=4000)
    logging.info(f"Text chunks: {text_chunks}")
else:
    logging.error("Error: Unable to process the text.")


from transformers import pipeline
from typing import Callable, List, Dict
import logging
bart_params = {
    "max_length": 1024,
    "min_length": 30,
    "do_sample": False, 
    "truncation": True,
    "repetition_penalty": 2.0,
}


def create_summarizer(model: str) -> Callable:
    summarizer = pipeline("summarization", model=model)
    return summarizer


def get_summary_bart(
    list_chunks: List[str], summarizer: Callable, summarization_params: Dict[str, int]
) -> str:
    # Generate summaries for each text chunk
    try:
        summaries = [
            summarizer(chunk, **summarization_params)[0]["summary_text"]
            for chunk in list_chunks
        ]
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error generating summaries: {e}")
        return ""


def save_summary_to_file(summary: str, file_name: str) -> None:
    try:
        # Save the summary to a file
        with open(f"{file_name}.txt", "a") as fp:
            fp.write(summary)
        fp.close()
    except Exception as e:
        logging.error(f"Error saving summary to file: {e}")
    


# Assume text_chunks is already defined and contains the chunks of text from the previous steps
summarizer = create_summarizer("facebook/bart-large-cnn")
summary = get_summary_bart(text_chunks, summarizer, bart_params)
save_summary_to_file(summary, f"summary_{VIDEO_NAME}")

if len(summary) > 5000:
    # If the summary is to long we can reapply the summarization function
    text_chunks = split_text_into_chunks(summary, max_tokens=1000)
    short_summary = get_summary_bart(text_chunks, summarizer, bart_params)
    save_summary_to_file(short_summary, f"short_summary_{VIDEO_NAME}")
    logging.info("Summary saved to file.")
else:
    logging.info("Summary is not applied again.")

    
import anthropic

start_str = "your task is to analyse the following text:\n<text>\n"
end_str = "\n</text>\nSummarize this text in a concise and clear manner, and identify key topics as questions. Organise the summary in a questions and answers style.  For each question, expand further the justification and rational in the answer.  Make sure to include all as much as details in your summary and analysis"
prompt = start_str + long_text + end_str


client = anthropic.Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"                
                }
            ]
        }
    ]
)
print(message.content)
save_summary_to_file(message.content, f"claude_summary_{VIDEO_NAME}")