import io
import os
import sys
import time
import gc
import threading
import queue
from openai import OpenAI
import httpx

# Suppress pygame welcome banner
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import soundfile as sf
import torch
import numpy as np
from kokoro import KPipeline

# 1. Configuration
#LLAMA_CPP_BASE_URL = "http://xerrameca:11434" 
#LLM_MODEL_ID = "ministral-3:8B"                  
LLAMA_CPP_BASE_URL = "http://127.0.0.1:1234" 
LLM_MODEL_ID = "lmstudio-communuty/qwen3.5-4b"


#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

print("⚙️ Initializing Kokoro-82M Pipeline...")
# 'a' stands for American English flavor phonemes
tts_pipeline = KPipeline(lang_code='a', device=DEVICE)

# Select premium Kokoro-82M built-in voice models
SPEAKER_CAPITALIST = "am_fenrir" 
SPEAKER_ACTIVIST = "af_heart"   

# Initialize API Client and Pygame
llm_client = OpenAI(base_url=f"{LLAMA_CPP_BASE_URL}/v1", api_key="llama-cpp")
pygame.mixer.init()

# Thread-safe queue to hold the next pre-generated turn
playback_queue = queue.Queue(maxsize=1) 

# --- Character Prompts ---
PROMPT_CAPITALIST = (
    "You are a fierce advocate for laissez-faire capitalism. You believe government intervention "
    "like minimum wage limits hurts the economy. Respond directly to your opponent's "
    "arguments. Keep your response strict, precise, and around 150 words. /no_think"
)

PROMPT_ACTIVIST = (
    "You are a passionate labor activist. You believe labor protections and the right to strike "
    "are vital achievements won through historical worker struggles. Respond directly to your opponent's "
    "arguments. Keep your response sharp, urgent, and around 150 words. /no_think"
)

topic = "Should the government enforce a minimum wage and limit weekly working hours?  /no_think"

history_capitalist = [
    {"role": "system", "content": PROMPT_CAPITALIST},
    {"role": "user", "content": f"The debate topic is: {topic}. Give your opening statement."}
]
history_activist = [{"role": "system", "content": PROMPT_ACTIVIST}]


# 2. VRAM Cleanup Helpers (Kept commented out per your structural setup)
def unload_llama_cpp_model():
    """Tells the llama.cpp server to eject the LLM from VRAM immediately."""
    try:
        url = f"{LLAMA_CPP_BASE_URL}/models/unload"
        response = httpx.post(url, json={"model": LLM_MODEL_ID}, timeout=5.0)
    except Exception:
        pass 


def generate_audio_buffer(text, speaker_name):
    """Generates audio arrays from Kokoro generator chunks and compiles a WAV stream buffer."""
    start_time = time.time()
    generator = tts_pipeline(text, voice=speaker_name, speed=1.0)
    all_audio_chunks = []
    
    for graphemes, phonemes, audio_chunk in generator:
        if audio_chunk is not None:
            all_audio_chunks.append(audio_chunk)
            
    # Concatenate the split text fragment arrays into one fluid array
    combined_audio = np.concatenate(all_audio_chunks)
    
    buffer = io.BytesIO()
    # Kokoro outputs at 24000Hz sampling rate
    sf.write(buffer, combined_audio, 24000, format='WAV')
    buffer.seek(0)
    duration = time.time() - start_time
    print(f"⏱️ Audio generation took {duration:.2f} seconds")
    return buffer


# 3. Background Pipeline Engine
def generation_worker():
    """Background thread that computes the NEXT turn ahead of time while audio plays."""
    global history_capitalist, history_activist
    
    current_turn = "capitalist"  
    round_counter = 1
    
    print("⏳ [Pipeline] Generating opening statement in background...")
    try:
        # LLM Call
        start_llm = time.time()
        response = llm_client.chat.completions.create(model=LLM_MODEL_ID, messages=history_capitalist, temperature=0.7)
        text = response.choices[0].message.content.strip()
        print(f"First LLM generation done. Took {time.time() - start_llm:.2f} seconds")
        
        # TTS Call
        buffer = generate_audio_buffer(text, SPEAKER_CAPITALIST)
        
        # Update histories
        history_capitalist.append({"role": "assistant", "content": text})
        history_activist.append({"role": "user", "content": f"Your opponent opened with: {text}"})
        
        playback_queue.put({"speaker": "💰 CAPITALIST LIBERAL", "text": text, "buffer": buffer})
        current_turn = "activist"
    except Exception as e:
        print(f"🛑 Background generation initialization error: {e}")
        return

    # Continuous generation loop
    while True:
        try:
            if current_turn == "activist":
                # Manage Context sliding window: keep system prompt + last 5 messages (must start/end with user)
                if len(history_activist) > 7:
                    history_activist = [history_activist[0]] + history_activist[-5:]

                # 1. LLM text generation
                start_llm = time.time()
                response = llm_client.chat.completions.create(model=LLM_MODEL_ID, messages=history_activist, temperature=0.7)
                text = response.choices[0].message.content.strip()
                print(f"LLM generation done. Took {time.time() - start_llm:.2f} seconds")

                # 2. Convert to Speech
                buffer = generate_audio_buffer(text, SPEAKER_ACTIVIST)
                
                # 3. Sync dialogue loop history
                history_activist.append({"role": "assistant", "content": text})
                history_capitalist.append({"role": "user", "content": text})
                
                # Push object to playback loop
                playback_queue.put({"speaker": f"✊ LABOUR ACTIVIST (Round {round_counter})", "text": text, "buffer": buffer})
                current_turn = "capitalist"
                
            else:
                # Capitalist's Turn
                # Manage Context sliding window: keep system prompt + last 5 messages (must start/end with user)
                if len(history_capitalist) > 7:
                    history_capitalist = [history_capitalist[0]] + history_capitalist[-5:]

                start_llm = time.time()
                response = llm_client.chat.completions.create(model=LLM_MODEL_ID, messages=history_capitalist, temperature=0.7)
                text = response.choices[0].message.content.strip()
                print(f"LLM generation done. Took {time.time() - start_llm:.2f} seconds")
                buffer = generate_audio_buffer(text, SPEAKER_CAPITALIST)
                
                history_capitalist.append({"role": "assistant", "content": text})
                history_activist.append({"role": "user", "content": text})
                
                playback_queue.put({"speaker": f"💰 CAPITALIST LIBERAL (Round {round_counter})", "text": text, "buffer": buffer})
                current_turn = "activist"
                round_counter += 1

        except Exception as e:
            print(f"⚠️ Background pipeline pipeline glitch: {e}")
            time.sleep(2) 


# 4. Main Thread Execution Loop (Playback)
print("\n=== PIPELINE AUDIO DEBATE INITIALIZED ===")
print(f"Topic: {topic}\nPress Ctrl+C to stop.\n" + "-"*50)

# Start the background pipeline thread
bg_thread = threading.Thread(target=generation_worker, daemon=True)
bg_thread.start()

try:
    while True:
        print("🎭 Waiting for background engine to compile assets...")
        next_turn = playback_queue.get() 
        
        print(f"\n{next_turn['speaker']}:\n{next_turn['text']}\n")
        
        pygame.mixer.music.load(next_turn['buffer'])
        pygame.mixer.music.play()
        print("🔊 [Playing Live Audio]")
        print("-" * 50)
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        print("🔊 [Done speaking]")
        time.sleep(0.4) 

except KeyboardInterrupt:
    pygame.mixer.music.stop()
    print("\n=== DEBATE GRACEFULLY ENDED ===")
