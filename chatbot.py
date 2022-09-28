import deepspeech
from halo import Halo
import numpy as np
from stt import VADAudio
from utils import get_response
import pandas as pd
import pyttsx3

def getChatbotTextResponse(inputText, movieData, engine):
    if len(inputText) == 0:
        return
    print("Input text detected: " + inputText)
    print()
    print("Chatbot Response: ")
    response = get_response(inputText, movieData)
    engine.say(response[0])
    engine.runAndWait()
    print()

def main():
    # Load DeepSpeech model
    model = 'deepspeech-0.9.3-models.pbmm'
    scorer = 'deepspeech-0.9.3-models.scorer'

    print('Initializing model...')
    print("model: %s", model)
    model = deepspeech.Model(model)
    if scorer:
        print("scorer: %s", scorer)
        model.enableExternalScorer(scorer)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=2,
                        device=None,
                        input_rate=DEFAULT_SAMPLE_RATE,
                        file=None)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()
    
    # Stream from microphone to DeepSpeech using VAD

    # Just the spinning halo for decorative purposes
    spinner = Halo(spinner='line')

    stream_context = model.createStream()
    wav_data = bytearray()

    # Read movie data for chatbot reference
    movieData = pd.read_csv('./data/movie_data.csv')

    # Prepare text to speech engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)

    # Listen for audio, feed to NLU, play response
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
        else:
            if spinner: spinner.stop()
            text = stream_context.finishStream()

            getChatbotTextResponse(text, movieData, engine)
            stream_context = model.createStream()
 
if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000
    main()