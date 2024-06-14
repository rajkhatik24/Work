import numpy as np
import torch
import scipy.signal as signal
import re
from openai import OpenAI
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("laoding model")
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "en"}
)
print("model Loaded")
client = OpenAI()


def apply_filters(audio: np.ndarray, sample_rate: int, lowpass: float = 8000, highpass: float = 75) -> np.ndarray:
    """
    Apply low pass and high pass filters to the audio data.

    Args:
        audio (np.ndarray): The input audio data as a numpy array.
        sample_rate (int): The sample rate of the audio data.
        lowpass (float): The cutoff frequency of the low pass filter in Hz.
        highpass (float): The cutoff frequency of the high pass filter in Hz.

    Returns:
        np.ndarray: The filtered audio data.
    """
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio

    audio_mono = audio_mono.astype(np.float32)
    audio_mono /= np.max(np.abs(audio_mono))

    nyquist_rate = sample_rate / 2.0
    normal_cutoff_low = lowpass / nyquist_rate
    b_low, a_low = signal.butter(5, normal_cutoff_low, btype='low', analog=False)
    low_passed_audio = signal.filtfilt(b_low, a_low, audio_mono)

    normal_cutoff_high = highpass / nyquist_rate
    b_high, a_high = signal.butter(5, normal_cutoff_high, btype='high', analog=False)
    filtered_audio = signal.filtfilt(b_high, a_high, low_passed_audio)

    return filtered_audio.copy()

# def transcribe(stream: np.ndarray, new_chunk: tuple) -> tuple:
#     """
#     Transcribe the audio stream using a pre-trained model.

#     Args:
#         stream (np.ndarray): The audio stream data.
#         new_chunk (tuple): The new audio chunk data (sample_rate, data).

#     Yields:
#         tuple: Updated audio stream and the transcribed text.
#     """
#     sr, y = new_chunk

#     if len(y.shape) > 1 and y.shape[1] > 1:
#         y_mono = np.mean(y, axis=1)
#     else:
#         y_mono = y

#     y_mono = y_mono.astype(np.float32)
#     y_mono /= np.max(np.abs(y_mono))

#     if stream is not None:
#         stream = np.concatenate([stream, y_mono])
#     else:
#         stream = y_mono

#     stream = apply_filters(stream, sr)

#     stream_copy = stream.copy()
    
#     result = transcriber({"sampling_rate": sr, "raw": stream_copy})["text"]
#     sentence_endings = re.compile(r'[.!?]')
#     last_sentence_end = 0

#     if result:
#         new_sentence =""
#         for match in sentence_endings.finditer(result):
#             end_idx = match.end()
#             new_sentence = result[last_sentence_end:end_idx].strip()
#             last_sentence_end = end_idx
#             if new_sentence:
#                 print("new sentence:", new_sentence)
#                 yield stream, new_sentence,new_sentence
                
#             else:
#                 yield stream, ""

#     remaining_text = result[last_sentence_end:].strip()
#     if remaining_text:
#         yield stream, remaining_text



import numpy as np
import re


import numpy as np
import re

def transcribe(stream: np.ndarray, new_chunk) -> tuple:
    """
    Transcribe the audio stream using a pre-trained model.

    Args:
        stream (np.ndarray): The audio stream data.
        new_chunk (tuple): The new audio chunk data (sample_rate, data).

    Yields:
        tuple: Updated audio stream and the transcribed text.
    """
    sr, y = new_chunk

    if len(y.shape) > 1 and y.shape[1] > 1:
        y_mono = np.mean(y, axis=1)
    else:
        y_mono = y

    y_mono = y_mono.astype(np.float32)
    y_mono /= np.max(np.abs(y_mono))

    if stream is not None:
        stream = np.concatenate([stream, y_mono])
    else:
        stream = y_mono

    # Assume apply_filters is defined elsewhere
    # stream = apply_filters(stream, sr)
    stream = apply_filters(stream, sr)
    stream_copy = stream.copy()

    # Assuming transcriber is defined elsewhere
    result = transcriber({"sampling_rate": sr, "raw": stream_copy})["text"]
    sentence_endings = re.compile(r'[.!?]')
    sentences = sentence_endings.split(result)
    
    if len(sentences) > 1:
        last_complete_sentence = ""
        for sentence in sentences[:-1]:
            if sentence.strip():
                last_complete_sentence = sentence.strip()
        
        if last_complete_sentence:
            print("last sentence:", last_complete_sentence)
            yield stream, last_complete_sentence, last_complete_sentence
    else:
        yield stream, ""

    # Update the stream to exclude the yielded sentences
    stream = stream_copy[len(result):]
    yield stream, ""


# Example usage
# stream = None
# for chunk in audio_chunks:
#     stream, text, _ = next(transcribe(stream, chunk))
#     if not stream:
#         break

# def transcribe(stream: np.ndarray, new_chunk) -> tuple:
#     """
#     Transcribe the audio stream using a pre-trained model.

#     Args:
#         stream (np.ndarray): The audio stream data.
#         new_chunk (tuple): The new audio chunk data (sample_rate, data).

#     Yields:
#         tuple: Updated audio stream and the transcribed text.
#     """
#     sr, y = new_chunk

#     if len(y.shape) > 1 and y.shape[1] > 1:
#         y_mono = np.mean(y, axis=1)
#     else:
#         y_mono = y

#     y_mono = y_mono.astype(np.float32)
#     y_mono /= np.max(np.abs(y_mono))

#     if stream is not None:
#         stream = np.concatenate([stream, y_mono])
#     else:
#         stream = y_mono

#     # Assume apply_filters is defined elsewhere
#     # stream = apply_filters(stream, sr)
#     stream = apply_filters(stream,sr)
#     stream_copy = stream.copy()

#     # Assuming transcriber is defined elsewhere
#     result = transcriber({"sampling_rate": sr, "raw": stream_copy})["text"]
#     sentence_endings = re.compile(r'[.!?]')
#     sentences = sentence_endings.split(result)
#     if len(sentences) > 1:
#         last_sentence = sentences[-2].strip()
#         if last_sentence:
#             print("last sentence:", last_sentence)
#             yield stream, last_sentence, last_sentence
#     else:
#         yield stream, ""

#     stream = stream_copy[len(last_sentence):]
#     yield stream, ""

def translate(stream: str) -> str:
    """
    Translate the transcribed text.

    Args:
        stream (str): Transcribed text.

    Returns:
        str: Translated text.
    """
    if stream.strip() == "" or stream.lower().strip() == "stop":
        return "", ""

    try:
        translated_text = stream
        # translated_text = client.completions.create(
        #     model='gpt-3.5-turbo-instruct',
        #     prompt=f"Return the following stream as it is :{stream}. Only return text and nothing else.",
        #     echo=False,
        #     n=1,
        #     stream=False).choices[0].text
        print(f"translated: {translated_text}")
        return translated_text,translated_text
    except Exception as e:
        return "",""

def speak(translated_text: str) -> np.ndarray:
    """
    Generate TTS audio in chunks from the translated text.

    Args:
        translated_text (str): Translated text to convert to speech.

    Yields:
        np.ndarray: TTS audio chunk.
    """
    print("switch")
    if translated_text.strip() == "":
        yield None
        return

    try:
        with client.with_streaming_response.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=translated_text,
            ) as response:
            for chunk in response.iter_bytes():
                yield chunk
    except Exception as e:
        yield None
