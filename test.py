import os 
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write#for saving the recorded audio as a wav file
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import time
import queue

class EnvConfig: 
    def __init__(self):
        #initializing dotenv 
        load_dotenv()
        self._key = os.getenv("API_KEY")
        if not self.key: 
            raise ValueError("missing api key")
        
    @property
    def key(self): 
        return self._key
    


class GroqAPI(EnvConfig): 
    def __init__(self):
        super().__init__()#fix parent class constructor
        #initializing groq api
        self.client = Groq(
            api_key=self.key
        )
        self.converstation_history = [{
            "role": "system", 
            "content": "you are my cool genz bestfriend bro gangsta"
        }]

    def convert_speech_to_text(self): 
        self.filename = os.path.dirname(__file__) + '/voice.wav'

        #open audio file 
        with open(self.filename, "rb") as file: 
            transcription = self.client.audio.transcriptions.create(
                file=file, # Required audio file
                model="whisper-large-v3-turbo", # Required model to use for transcription
                prompt="Specify context or spelling",  # Optional
                response_format="verbose_json",  # Optional
                timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
                language="en",  # Optional
                temperature=0.0  # Optional
            )
            return transcription.text#will be pass to ai response
        

    def talk_to_ai(self): 
        self.user_prompt = self.convert_speech_to_text()
        self.converstation_history.append({
            "role": "user", 
             "content": self.user_prompt
        })

        #initialize ai text generation
        self.chat = self.client.chat.completions.create(
            messages=self.converstation_history,
            model="llama-3.3-70b-versatile"
        )
        ai_message = self.chat.choices[0].message.content
        self.converstation_history.append({
            "role": "assistant", 
            "content": ai_message
        })

        print(f'user : {self.user_prompt}\n\n')
        print(f'ai : {ai_message}')
        return ai_message

    
    def convert_text_to_speech(self): 
        self.file_path = "aiVoice.wav"
        self.model = "playai-tts"
        self.voice = 'Arista-PlayAI'
        self.text = self.talk_to_ai()
        self.response_format = 'wav'

        self.response = self.client.audio.speech.create(
            model=self.model, 
            voice=self.voice, 
            input=self.text, 
            response_format=self.response_format
        )
        #saving the speech to the wav file 
        self.response.write_to_file(self.file_path)



class RealTimeRecorder:
    def __init__(self, device_index=5):
        self.fs = 16000
        self.channels = 1
        self.silence_thresh = 500 
        self.silence_timeout = 1.5  # seconds of silence before stopping
        self.recording = False
        self.q = queue.Queue()
        # self.audio_chunks = []

        # Set default input device globally
        sd.default.device = (device_index, None)

    def callback(self, indata, frames, time_info, status):
        """Audio callback - runs in a background thread."""
        self.q.put(indata.copy())

    def record(self):
        print("🎤 Start speaking...")
        self.recording = False#to reset the voice detected from the prev call of the function
        # self.audio_chunks = []# to override the previous value of chunks

        with sd.InputStream(samplerate=self.fs, channels=self.channels, dtype='int16', callback=self.callback):
            silence_timer = None

            while True:
                data = self.q.get()
                volume = np.linalg.norm(data)

                # print(f"🔊 Volume: {volume:.2f}")

                if volume > self.silence_thresh:
                    if not self.recording:
                        print("🎙️ Voice detected. Recording...")
                        self.recording = True
                        self.audio_chunks = []
                    self.audio_chunks.append(data)
                    silence_timer = None  # Reset silence timer
                elif self.recording:
                    self.audio_chunks.append(data)
                    if silence_timer is None:
                        silence_timer = time.time()
                    elif time.time() - silence_timer > self.silence_timeout:
                        print("🛑 Silence detected. Stopping recording.")
                        break

    def save(self):
        if not self.audio_chunks:
            print("⚠️ No audio recorded.")
            return
        audio_data = np.concatenate(self.audio_chunks, axis=0)
        write("voice.wav", self.fs, audio_data)
        print("✅ File saved as final_output.wav")


class ElevenLabsConfig(GroqAPI): 
    def __init__(self):
        super().__init__()
        self.elevenlabs = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )

    def generate_audio(self): 
        self.audio = self.elevenlabs.text_to_speech.convert(
            
        )

def main():
    recorder = RealTimeRecorder(device_index=5)  # Use your actual working mic index
    ai = GroqAPI()

    while True: 
        try: 
            recorder.record()
            recorder.save()
            ai.convert_text_to_speech()

        except Exception as e: 
            print(f'An error occured : {e}')

        #to cool down the the conversation 
        # time.sleep(5)

if __name__ == "__main__":
    main()
