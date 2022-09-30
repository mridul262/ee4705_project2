import deepspeech
from halo import Halo
import numpy as np
from stt import VADAudio
from utils import get_response
import pandas as pd
import pyttsx3

def getChatbotTextResponse(inputText, movieData, intent_state, engine):
    if len(inputText) == 0:
        return intent_state, movieData
    print("Input text detected: " + inputText)
    print()
    print("Chatbot Response: ")
    response, movieData, intent_state = get_response(inputText, movieData, intent_state)
    engine.say(response)
    engine.runAndWait()
    print()
    return intent_state, movieData

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
    movieData['Year'] = pd.to_numeric(movieData['Year'])
    movieData['Rating'] = pd.to_numeric(movieData['Rating'])
    originalData = movieData.copy()
    # Prepare text to speech engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)

    # For Intent Tracking
    intent_state = 0
    # Listen for audio, feed to NLU, play response
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
        else:
            if spinner: spinner.stop()
            text = stream_context.finishStream()

            intent_state, movieData = getChatbotTextResponse(text, movieData, intent_state, engine)
            if intent_state == 1:
                movieData = originalData
            print(movieData)
            stream_context = model.createStream()
 
if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000
    main()