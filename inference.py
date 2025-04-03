import torch
import torchaudio
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor
processor = WhisperProcessor.from_pretrained("./whisper-small-en", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-en").to("cuda:6")

# Ensure no forced decoder IDs for transcription task
model.generation_config.forced_decoder_ids = None

# Function to resample audio to 16kHz
def resample_audio(waveform, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(torch.tensor(waveform))
    return waveform, 16000

# Transcription function
def transcribe(audio):
    try:
        print("Received audio for transcription.")
        print(f"Audio type: {type(audio)}")

        if audio is None:
            print("No audio received. Returning.")
            return "No audio received."
        print(audio)
        # Audio is a tuple (waveform as numpy array, sample rate)
        sample_rate, waveform = audio
        print(waveform.dtype)
        if waveform.dtype == "int16":
            waveform = waveform.astype("float32") / 32768.0
            
        # Resample the audio to 16kHz
        waveform, sample_rate = resample_audio(waveform, sample_rate)
        print(f"Resampled audio sample rate: {sample_rate}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Waveform dtype: {waveform.dtype}")
        print(f"Waveform device: {waveform.device}")

        # Ensure waveform is a 1D tensor
        waveform = torch.tensor(waveform).squeeze()



        # Process audio to get input features
        inputs = processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # Move input features to GPU
        input_features = inputs['input_features'].to("cuda:6")

        # Generate token IDs using input_features directly
        predicted_ids = model.generate(input_features=input_features)

        # Decode token IDs to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Error during transcription: {str(e)}"

# Gradio Interface
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="numpy", label="Record Your Voice"),
    outputs=gr.Textbox(label="Transcription"),
    title="Whisper ASR Transcription",
    description="Upload or record an audio file and get the transcription.",
    theme="default",
    allow_flagging="never",
    live=True,  # Enables real-time updates
).launch(share=True)
