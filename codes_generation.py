from encodec import EncodecModel
from encodec.utils import convert_audio
from audiocraft.models import MultiBandDiffusion
import torch
from audiocraft.data.audio import audio_read, audio_write
import os

bandwidth = 3.0  # 1.5, 3.0, 6.0
mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
encodec_24khz = EncodecModel.encodec_model_24khz()

def get_degraded_codes(path='19_198_000000_000000_downsampled_12000.wav'):
    with torch.no_grad():
        wav, sr = audio_read(path)
        wav = convert_audio(wav, sr, encodec_24khz.sample_rate, encodec_24khz.channels)
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            encoded_frames = encodec_24khz.encode(wav)

        # encoded_frames is a tuple (codes, None), so we return the first element.
        return encoded_frames[0][0], sr

def upsample_24kHzcodes_with_mbd(encoded_frames_12khz):
    with torch.no_grad():
        # Step 1: Use MBD to generate a 24kHz waveform from the 12kHz codes.
        reconstructed_wav_24khz = mbd.tokens_to_wav(encoded_frames_12khz)

        # Step 2: Use the 24kHz EnCodec model to get the new codes.
        encoded_frames_24khz = encodec_24khz.encode(reconstructed_wav_24khz)

    return reconstructed_wav_24khz, encoded_frames_24khz[0][0]

def generate_using_mbd(encoded_frames):
    with torch.no_grad():
        # Get the codes from the tuple
        reconstructed_wav_24khz, encoded_frames_24khz = upsample_24kHzcodes_with_mbd(encoded_frames)

    audio_write(
        'mbd_generated_from_codes',
        reconstructed_wav_24khz.squeeze(0).cpu(),
        mbd.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )

    print(f"Audio generated from latent codes and saved to 'mbd_generated_from_codes.wav'.")

    return reconstructed_wav_24khz, encoded_frames_24khz

# encoded_codes, sr = get_degraded_codes()
# reconstructed_wav, upsampled_encoded_codes = generate_using_mbd(encoded_codes)


# traverse training_data folder
training_data_path = 'training_data/'
codes = dict()

for dirpath, dirnames, filenames in os.walk(training_data_path):
    for filename in filenames:
        target = f"{dirpath}{filename}"
        if 'downsampled' in filename:
            degraded_codes, sr = get_degraded_codes(target)
            reconstructed_wav, upsampled_codes = generate_using_mbd(degraded_codes)
            codes[filename] = {'degraded':degraded_codes, 'target':upsampled_codes}
        break
print(codes)
