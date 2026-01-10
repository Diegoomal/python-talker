
print("Starting application...")

print("Importing libraries...")
import whisper                                      # type: ignore

from ollama import chat                             # type: ignore
from ollama import ChatResponse                     # type: ignore

import torch                                        # type: ignore
from transformers import BarkModel, AutoProcessor   # type: ignore

import scipy.io.wavfile as wav                      # type: ignore

print("Loading models...")

print("Loading Whisper model...")
model_whisper = whisper.load_model("base")

print("Loading Bark model...")
device = "cuda"
model_suno = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark")



def get_transcribed_text(audio):
    waveform = audio.squeeze().numpy()
    result = model_whisper.transcribe(waveform, fp16=False)
    return result["text"]

def get_chat_response(message: str) -> ChatResponse:
    system_prompt = "You are a helpful, polite, and clear assistant in all your responses."
    messages=[
        { "role": "system", "content": system_prompt },
        { 'role': 'user', 'content': message },
    ],
    response = chat( model="phi3:instruct", messages=messages)
    return response

def generate_audio_from_text(text_prompt: str):
    inputs = processor(text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        speech_output = model_suno.generate(**inputs)
    return speech_output

def save_audio_to_wav(speech_tensor):
    audio = speech_tensor.cpu().numpy().squeeze()
    wav.write("output.wav", model_suno.generation_config.sample_rate, audio)

def main():

    print("Running main application...")

    print("Loading audio file...")
    # carrega o áudio
    sr, audio = wav.read("../../dataset/output.wav")
    audio = torch.from_numpy(audio).float()

    print("Processing audio file... transcribed audio to text")
    # transcreve
    text = get_transcribed_text(audio)

    print("LLM processing...")
    # envia ao LLM
    response = get_chat_response(text)
    reply = response["message"]["content"]

    print("Generating audio response...")
    # gera áudio da resposta
    speech = generate_audio_from_text(reply)

    print("Saving audio response to output.wav...")
    # salva
    save_audio_to_wav(speech)



if __name__ == "__main__":
    main()
