from scipy.io.wavfile import write 
import numpy as np 
import sounddevice as sd 

fs = 16000
silence_thresh = 