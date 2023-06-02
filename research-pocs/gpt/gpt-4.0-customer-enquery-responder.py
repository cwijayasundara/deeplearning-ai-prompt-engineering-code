import openai
import os
from datasets import load_dataset  # use the HuggingFace Banking77 dataset
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')

# MODEL = "gpt-4" # 8K context	$0.03 / 1K tokens	$0.06 / 1K tokens (Expensive!)
MODEL = "gpt-3.5-turbo"  # gpt-3.5-turbo	$0.002 / 1K tokens

dataset = load_dataset("banking77")

# Sort the dataset by the length of the customer texts
sorted_data = sorted(dataset['train'], key=lambda x: len(x['text']), reverse=True)

# Extract the longest 5 customer texts
longest_five_texts = [entry["text"] for entry in sorted_data[:5]]

# Print the longest 5 customer texts
for i, text in enumerate(longest_five_texts):
    print(f"Longest Customer Text {i + 1} (Length: {len(text)}): {text}")


def get_completion(prompt_to_gpt, model=MODEL):
    messages = [{"role": "user", "content": prompt_to_gpt}]
    response_from_gpt = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response_from_gpt.choices[0].message["content"]


for i in range(len(longest_five_texts)):
    print(f"Longest Customer Text {i + 1} (Length: {len(longest_five_texts[i])}): {longest_five_texts[i]}")

    prompt_0 = f"""
    Identify the following items from the customer chat text: 
    - Intent of the customer
    - Any other requests

    The review is delimited with triple backticks. \
    Format your response as a JSON object with \
    "Intent" and "description" as the keys.
    Make your response as short as possible.

    customer text: '''{longest_five_texts[i]}'''
    """
    intent = get_completion(prompt_0)
    print(f"intent of ", i + 1, "text is -", intent, "\n")

    prompt_1 = f"""
    What is the sentiment of the following comments, 
    which is delimited with triple backticks?

    Give your answer as a single word, either "positive" \
    or "negative".

    Review text: '''{longest_five_texts[i]}'''
    """
    sentiment = get_completion(prompt_1)
    print(f"sentiment of ", i + 1, "text is -", sentiment, "\n")

    prompt_2 = f"""
    Your task is to generate a short summary of the customer inquiry. \

    Summarize the inquiry below, delimited by triple 
    backticks, in at most 15 words, and focusing on any aspects \
    that mention banking products like debit card, credit card and accounts. 

    summary text: ```{longest_five_texts[i]}```
    """
    summary = get_completion(prompt_2)
    # print(i, summary, "\n")
    print(f"summary of ", i + 1, "text is - ", summary, "\n")

    prompt_3 = f"""
    You are a customer service AI assistant.
    Your task is to send an email reply to a valued customer.
    Given the customer email delimited by ```, \
    Generate a reply to thank the customer for their review.
    If the sentiment is positive or neutral, thank them for \
    their review.
    If the sentiment is negative, apologize and suggest that \
    they can reach out to customer service. 
    Make sure to use specific details from the review.
    Write in a concise and professional tone.
    Sign the email as `AI customer agent`.
    Customer review: ```{longest_five_texts[i]}```
    Review sentiment: {sentiment}
    """
    response = get_completion(prompt_3)
    print(f"Response from the  GPT Email Bot for the  ", i + 1, "text is - ", response, "\n")
    print("******************", "\n")
