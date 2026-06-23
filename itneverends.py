import io
import os
import sys
import time
import gc
from openai import OpenAI

# Suppress pygame welcome banner
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# 1. Initialize API Client
# Point this to your local llama.cpp server (default port is usually 8080 or 5000)
llm_client = OpenAI(base_url="http://10.0.50.21:11211/v1", api_key="llama-cpp")
LLM_MODEL = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

# llm_client = OpenAI(base_url="http://iacitm:11434/v1", api_key="llama-cpp")
# LLM_MODEL = "qwen3-vl:4b"

# Configuration
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

SPEAKER_CAPITALIST = "dylan" 
SPEAKER_ACTIVIST = "ryan"   

pygame.mixer.init()

# 2. Define Character Prompts
PROMPT_CAPITALIST = (
    "You are a fierce advocate for laissez-faire capitalism. You believe government intervention "
    "like minimum wage and working hour limits hurts the economy. Respond directly to your opponent's "
    "arguments. Keep your response strict, precise, and under 80 words. Do not include the word length in the answer."
)

PROMPT_ACTIVIST = (
    "You are a passionate labor activist. You believe labor protections and the right to strike "
    "are vital achievements won through historical worker struggles. Respond directly to your opponent's "
    "arguments. Keep your response sharp, urgent, and under 80 words. Do not include the word length in the answer."
)

topic = "Should the government enforce a minimum wage and limit weekly working hours?"

history_capitalist = [
    {"role": "system", "content": PROMPT_CAPITALIST},
    {"role": "user", "content": f"The debate topic is: {topic}. Give your opening statement."}
]
history_activist = [{"role": "system", "content": PROMPT_ACTIVIST}]


# 3. Memory-Managed TTS Context Manager
class ManagedTTS:
    """Dynamically loads Qwen-TTS into memory and cleanly unloads it on exit."""
    def __enter__(self):
        # Load the model right before audio generation
        self.tts_model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2",
        )
        return self.tts_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Destroy the model reference
        del self.tts_model
        # Force Python to clear unused objects
        gc.collect()
        # Force PyTorch to release VRAM back to the OS/llama.cpp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_llm_response(messages):
    """Fetches response from your local llama_cpp engine."""
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n🛑 llama_cpp Connection Error: {e}")
        sys.exit(1)


def play_speech_and_unload(text, speaker_name):
    """Safely handles the lifecycle of the audio generation and playback."""
    try:
        print("📥 Loading TTS model into memory...")
        with ManagedTTS() as tts:
            audio_array, sample_rate = tts.generate_custom_voice(text, language="English", speaker=speaker_name)
        
        # The model is now completely unloaded from hardware at this line!
        print("📤 TTS model unloaded. Playing audio and freeing VRAM for llama_cpp...")
        
        # Play the generated audio data from RAM
        buffer = io.BytesIO()
        sf.write(buffer, audio_array[0], sample_rate, format='WAV')
        buffer.seek(0)
        
        pygame.mixer.music.load(buffer)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
            
    except Exception as e:
        print(f"⚠️ TTS Playback failed: {e}")


# 4. Running Continuous Loop Simulation
print("\n=== LIVE VRAM-OPTIMIZED DEBATE ===")
print(f"Topic: {topic}\nPress Ctrl+C to stop.\n" + "-"*50)

# --- Opening Statement ---
capitalist_reply = get_llm_response(history_capitalist)
print(f"💰 CAPITALIST LIBERAL ({SPEAKER_CAPITALIST}):\n{capitalist_reply}\n")
play_speech_and_unload(capitalist_reply, SPEAKER_CAPITALIST)
print("-" * 50)

history_capitalist.append({"role": "assistant", "content": capitalist_reply})
history_activist.append({"role": "user", "content": f"Your opponent opened with: {capitalist_reply}"})

round_counter = 1
try:
    while True:
        # Sliding Window Trim
        if len(history_activist) > 7:
            history_activist = [history_activist[0]] + history_activist[-6:]
        if len(history_capitalist) > 7:
            history_capitalist = [history_capitalist[0]] + history_capitalist[-6:]

        # Turn: Activist
        activist_reply = get_llm_response(history_activist)
        print(f"✊ LABOUR ACTIVIST ({SPEAKER_ACTIVIST} - Round {round_counter}):\n{activist_reply}\n")
        play_speech_and_unload(activist_reply, SPEAKER_ACTIVIST)
        print("-" * 50)
        
        history_activist.append({"role": "assistant", "content": activist_reply})
        history_capitalist.append({"role": "user", "content": activist_reply})
        
        # Turn: Capitalist
        capitalist_reply = get_llm_response(history_capitalist)
        print(f"💰 CAPITALIST LIBERAL ({SPEAKER_CAPITALIST} - Round {round_counter}):\n{capitalist_reply}\n")
        play_speech_and_unload(capitalist_reply, SPEAKER_CAPITALIST)
        print("-" * 50)
        
        history_capitalist.append({"role": "assistant", "content": capitalist_reply})
        history_activist.append({"role": "user", "content": capitalist_reply})
        
        round_counter += 1

except KeyboardInterrupt:
    pygame.mixer.music.stop()
    print("\n=== DEBATE GRACEFULLY ENDED ===")
