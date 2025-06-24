import os 
from groq import Groq
import json
from dotenv import load_dotenv
import sounddevice
from scipy.io.wavfile import write#for saving the recorded audio as a wav file

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


class SpeechFileConverter: 
    def __init__(self):
        self.fs = 44100 #crucial for audio quality to define the number of rate in audio 

    def record_audio(self, audio): 
        

        
