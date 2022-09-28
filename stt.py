import collections
import contextlib
import glob
import wave
import webrtcvad
from deepspeech import Model
import deepspeech
import pyaudio
import queue
from halo import Halo
import numpy as np

def read_wave(path):
   """Reads a .wav file.
 
   Takes the path, and returns (PCM audio data, sample rate).
   """
   with contextlib.closing(wave.open(path, 'rb')) as wf:
       num_channels = wf.getnchannels()
       assert num_channels == 1
       sample_width = wf.getsampwidth()
       assert sample_width == 2
       sample_rate = wf.getframerate()
       assert sample_rate in (8000, 16000, 32000)
       frames = wf.getnframes()
       pcm_data = wf.readframes(frames)
       duration = frames / sample_rate
       return pcm_data, sample_rate, duration

class Frame(object):
   """Represents a "frame" of audio data."""
   def __init__(self, bytes, timestamp, duration):
       self.bytes = bytes
       self.timestamp = timestamp
       self.duration = duration
 
def frame_generator(frame_duration_ms, audio, sample_rate):
   """Generates audio frames from PCM audio data.
 
   Takes the desired frame duration in milliseconds, the PCM data, and
   the sample rate.
 
   Yields Frames of the requested duration.
   """
   n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
   offset = 0
   timestamp = 0.0
   duration = (float(n) / sample_rate) / 2.0
   while offset + n < len(audio):
       yield Frame(audio[offset:offset + n], timestamp, duration)
       timestamp += duration
       offset += n

def vad_collector(sample_rate, frame_duration_ms,
                 padding_duration_ms, vad, frames):
   """Filters out non-voiced audio frames.
 
   Given a webrtcvad.Vad and a source of audio frames, yields only
   the voiced audio.
 
   Uses a padded, sliding window algorithm over the audio frames.
   When more than 90% of the frames in the window are voiced (as
   reported by the VAD), the collector triggers and begins yielding
   audio frames. Then the collector waits until 90% of the frames in
   the window are unvoiced to detrigger.
 
   The window is padded at the front and back to provide a small
   amount of silence or the beginnings/endings of speech around the
   voiced frames.
 
   Arguments:
 
   sample_rate - The audio sample rate, in Hz.
   frame_duration_ms - The frame duration in milliseconds.
   padding_duration_ms - The amount to pad the window, in milliseconds.
   vad - An instance of webrtcvad.Vad.
   frames - a source of audio frames (sequence or generator).
 
   Returns: A generator that yields PCM audio data.
   """
   num_padding_frames = int(padding_duration_ms / frame_duration_ms)
   # We use a deque for our sliding window/ring buffer.
   ring_buffer = collections.deque(maxlen=num_padding_frames)
   # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
   # NOTTRIGGERED state.
   triggered = False
 
   voiced_frames = []
   for frame in frames:
       is_speech = vad.is_speech(frame.bytes, sample_rate)
 
       if not triggered:
           ring_buffer.append((frame, is_speech))
           num_voiced = len([f for f, speech in ring_buffer if speech])
           # If we're NOTTRIGGERED and more than 90% of the frames in
           # the ring buffer are voiced frames, then enter the
           # TRIGGERED state.
           if num_voiced > 0.9 * ring_buffer.maxlen:
               triggered = True
               # We want to yield all the audio we see from now until
               # we are NOTTRIGGERED, but we have to start with the
               # audio that's already in the ring buffer.
               for f, s in ring_buffer:
                   voiced_frames.append(f)
               ring_buffer.clear()
       else:
           # We're in the TRIGGERED state, so collect the audio data
           # and add it to the ring buffer.
           voiced_frames.append(frame)
           ring_buffer.append((frame, is_speech))
           num_unvoiced = len([f for f, speech in ring_buffer if not speech])
           # If more than 90% of the frames in the ring buffer are
           # unvoiced, then enter NOTTRIGGERED and yield whatever
           # audio we've collected.
           if num_unvoiced > 0.9 * ring_buffer.maxlen:
               triggered = False
               yield b''.join([f.bytes for f in voiced_frames])
               ring_buffer.clear()
               voiced_frames = []
   # if triggered:
   #     pass
   # If we have any leftover voiced audio when we run out of input,
   # yield it.
   if voiced_frames:
       yield b''.join([f.bytes for f in voiced_frames])

'''
Generate VAD segments. Filters out non-voiced audio frames.
@param waveFile: Input wav file to run VAD on.0
 
@Retval:
Returns tuple of
   segments: a bytearray of multiple smaller audio frames
             (The longer audio split into multiple smaller ones)
   sample_rate: Sample rate of the input audio file
   audio_length: Duration of the input audio file
 
'''
def vad_segment_generator(wavFile, aggressiveness):
   print("Caught the wav file @: %s" % (wavFile))
   audio, sample_rate, audio_length = read_wave(wavFile)
   assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
   vad = webrtcvad.Vad(int(aggressiveness))
   frames = frame_generator(30, audio, sample_rate)
   frames = list(frames)
   segments = vad_collector(sample_rate, 30, 300, vad, frames)
 
   return segments, sample_rate, audio_length

'''
Load the pre-trained model into the memory
@param models: Output Graph Protocol Buffer file
@param scorer: Scorer file
 
@Retval
Returns a list [DeepSpeech Object, Model Load Time, Scorer Load Time]
'''
def load_model(models, scorer):
   model_load_start = timer()
   ds = Model(models)
   model_load_end = timer() - model_load_start
   print("Loaded model in %0.3fs." % (model_load_end))
 
   scorer_load_start = timer()
   ds.enableExternalScorer(scorer)
   scorer_load_end = timer() - scorer_load_start
   print('Loaded external scorer in %0.3fs.' % (scorer_load_end))
 
   return [ds, model_load_end, scorer_load_end]
 
'''
Resolve directory path for the models and fetch each of them.
@param dirName: Path to the directory containing pre-trained models
 
@Retval:
Retunns a tuple containing each of the model files (pb, scorer)
'''
def resolve_models(dirName):
   pb = glob.glob(dirName + "/*.pbmm")[0]
   print("Found Model: %s" % pb)
 
   scorer = glob.glob(dirName + "/*.scorer")[0]
   print("Found scorer: %s" % scorer)
 
   return pb, scorer

'''
Run Inference on input audio file
@param ds: Deepspeech object
@param audio: Input audio for running inference on
@param fs: Sample rate of the input audio file
 
@Retval:
Returns a list [Inference, Inference Time, Audio Length]
 
'''
def stt(ds, audio, fs):
   inference_time = 0.0
   audio_length = len(audio) * (1 / fs)
 
   # Run Deepspeech
   print('Running inference...')
   inference_start = timer()
   output = ds.stt(audio)
   inference_end = timer() - inference_start
   inference_time += inference_end
   print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
 
   return [output, inference_time]


class VADAudio(object):
    """Filter & segment audio with voice activity detection."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50
   
    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None, aggressiveness=3):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(aggressiveness)

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                            input_rate=self.input_rate)
 
    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                        |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()