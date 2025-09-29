"""
Text2Audio.py - Text-to-Audio (Music Generation) with HuggingFace Transformers

This script demonstrates how to generate audio (music) from a text prompt using a pre-trained generative model from the HuggingFace Transformers library. It is designed for production-quality, end-to-end text-to-audio synthesis, and can be adapted for other prompts, models, or output formats.

Overview:
---------
- Loads a pre-trained text-to-audio (music generation) model ("facebook/musicgen-small") using the HuggingFace pipeline abstraction.
- Accepts a text prompt describing the desired music style or content.
- Generates an audio waveform based on the prompt.
- Saves the generated audio as a WAV file using SciPy.

Key Components:
---------------
1. Model Selection and Loading:
	- The "facebook/musicgen-small" model is a generative model trained to synthesize music from natural language descriptions.
	- The model is loaded using the `pipeline` function, which abstracts away tokenization, model input formatting, and output post-processing.
	- The `task` is set to "text-to-audio" to indicate the type of generation.

2. Audio Generation:
	- The pipeline is called with a text prompt (e.g., "lo-fi music with a soothing melody") to generate music matching the description.
	- The `forward_params` argument allows for sampling-based generation (`do_sample=True`), which introduces randomness and variety in the output.
	- The output is a dictionary containing:
	  - `audio`: The generated audio waveform as a NumPy array.
	  - `sampling_rate`: The sample rate (Hz) of the audio.

3. Output:
	- The generated audio is saved as 'musicgen_out.wav' in the current working directory using `scipy.io.wavfile.write`.
	- The output file can be played with any standard audio player.

Best Practices:
---------------
- For production use, validate the input prompt and handle exceptions during model loading and inference.
- To use a different model or generate speech instead of music, change the model name to a compatible text-to-audio or text-to-speech model from HuggingFace Hub.
- For batch processing, loop over a list of prompts and save each generated audio file with a unique filename.
- For web or interactive applications, return the audio data directly instead of saving to disk.

Dependencies:
-------------
- transformers >= 4.x
- torch (PyTorch backend)
- scipy

References:
-----------
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextToAudioPipeline
- MusicGen model: https://huggingface.co/facebook/musicgen-small

"""
#%% packages
from transformers import pipeline
import scipy

#%% model selection
task = "text-to-audio"
model = "facebook/musicgen-small"

# %%
synthesiser = pipeline(task, model)
music = synthesiser("lo-fi music with a soothing melody", forward_params=
{"do_sample": True})
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"],
data=music["audio"])