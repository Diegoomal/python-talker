
# print("Starting application...")

# print("Importing libraries...")

# import os
# import whisper                                      # type: ignore

# from ollama import chat                             # type: ignore
# from ollama import ChatResponse                     # type: ignore

# import torch                                        # type: ignore
# from transformers import BarkModel, AutoProcessor   # type: ignore

# import scipy.io.wavfile as wav                      # type: ignore

# print("Loading models...")

# print("Loading Whisper model...")

# model_whisper = whisper.load_model("base")

# print("Loading Bark model...")

# device = "cuda"
# model_suno = BarkModel.from_pretrained("suno/bark-small").to(device)
# processor = AutoProcessor.from_pretrained("suno/bark")

# def get_transcribed_text(audio):
#     print('get_transcribed_text')
#     waveform = audio.squeeze().numpy()
#     result = model_whisper.transcribe(waveform, fp16=False)
#     print("Human:", result["text"])
#     return result["text"]

# def get_chat_response(message: str) -> ChatResponse:
#     print('get_chat_response')
#     system_prompt = "You are a helpful, polite, and clear assistant in all your responses."
#     messages = [
#         { "role": "system", "content": system_prompt },
#         { "role": "user", "content": message },
#     ]
#     response = chat(model="phi3:instruct", messages=messages)
#     print("AI-Assistent:", response["message"]["content"])
#     return response

# def generate_audio_from_text(text_prompt: str):
#     print('generate_audio_from_text')
#     inputs = processor(text_prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         speech_output = model_suno.generate(**inputs)
#     return speech_output

# def save_audio_to_wav(speech_tensor, filename):
#     print('save_audio_to_wav')
#     audio = speech_tensor.cpu().numpy().squeeze()
#     wav.write(filename, model_suno.generation_config.sample_rate, audio)

# print("Running main application...")

# # minimal pipeline: 3 workers (ASR -> LLM -> TTS)
# import queue, threading

# asr_q = queue.Queue(maxsize=4)
# llm_q = queue.Queue(maxsize=4)
# out_q = queue.Queue(maxsize=4)

# def asr_worker(audio_paths):
#     print('asr_worker')
#     for p in audio_paths:
#         audio = whisper.load_audio(p)            # keep your preprocessing
#         audio = whisper.pad_or_trim(audio)
#         audio = torch.from_numpy(audio).float().to(device)
#         with torch.inference_mode():
#             text = get_transcribed_text(audio)   # keep lightweight
#         asr_q.put(text)
#     asr_q.put(None)

# def llm_worker():
#     print('llm_worker')
#     while True:
#         text = asr_q.get()
#         if text is None:
#             llm_q.put(None); break
#         # request smaller/quantized model or streaming if available
#         resp = get_chat_response(text)
#         llm_q.put(resp["message"]["content"])

# def tts_worker(output_dir):
#     print('tts_worker')
#     idx = 0
#     while True:
#         reply = llm_q.get()
#         if reply is None:
#             break
#         # generate in mixed precision, no_grad
#         with torch.inference_mode():
#             speech = generate_audio_from_text(reply)
#         out_path = f"{output_dir}/out_{idx}.wav"
#         save_audio_to_wav(speech, out_path)
#         idx += 1
#         out_q.put(out_path)
#     out_q.put(None)

# # start threads

# base = os.path.dirname(os.path.abspath(__file__))
# human_dir = os.path.join(base, "human_simulation")

# audio_files = [os.path.join(human_dir, f"{i}.wav") for i in range(5)]

# # audio_files = [f"human_simulation/{i}.wav" for i in range(5)]


# threads = [
#     threading.Thread(target=asr_worker, args=(audio_files,)),
#     threading.Thread(target=llm_worker),
#     threading.Thread(target=tts_worker, args=(".",)),
# ]
# for t in threads: t.start()
# for t in threads: t.join()
