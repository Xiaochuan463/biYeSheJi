import pvporcupine
import sounddevice as sd
import numpy as np

ACCESS_KEY = "gG0Oynu4Rw7MYFq7nSn53XrCaEyNJrAbqtdOCU3+AtKKZO6cG9jKCA=="

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=["jarvis"]
)

sample_rate = porcupine.sample_rate
frame_length = porcupine.frame_length

print("等待唤醒词...")

with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        blocksize=frame_length) as stream:

    while True:
        pcm, _ = stream.read(frame_length)
        pcm = pcm.flatten()

        result = porcupine.process(pcm)

        if result >= 0:
            print("唤醒成功!")