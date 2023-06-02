import IPython
import soundfile as sf
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai_api_key = os.getenv('OPENAI_API_KEY')


def play_audio(audio):
    sf.write("speech_converted.wav", audio.numpy(), samplerate=16000)
    return IPython.display.Audio("speech_converted.wav")


agent_name = "OpenAI (API Key)"

if agent_name == "OpenAI (API Key)":
    from transformers.tools import OpenAiAgent
    agent = OpenAiAgent(model="gpt-3.5-turbo", api_key=openai_api_key)
    print("OpenAI is initialized 💪")

audio = agent.run("Read out loud the summary of https://www.newscientist.com/article/mg25834383-000-why-virtual"
                  "-particles-dont-exist-but-do-explain-reality-for-now/ in no less than 500 words")
play_audio(audio)