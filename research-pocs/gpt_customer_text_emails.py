import openai

import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL = "gpt-3.5-turbo"


def get_completion(prompt_to_gpt, model=MODEL):
    messages = [{"role": "user", "content": prompt_to_gpt}]
    response_from_gpt = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response_from_gpt.choices[0].message["content"]


comment_1 = "Please help me with my card. It won't activate., I tired but an unable to activate my card., I want to " \
            "start using my card., How do I verify my new card?, I tried activating my plug-in and it didn't piece of" \
            " work"
comment_2 = "Hi, I have an apple watch. How do I use it to top up my card?, Can I use google pay to top up?, " \
            "why top up is not working even if I got my American Express in Apple Bay?, Can I deposit money using " \
            "Apple Pay?, Can I use google pay for topping -up"
comment_3 = "is there something blocking me from making transfers, What are the reasons for my beneficiary not being " \
            "allowed?, Is there something wrong with the transferring functions? I keep trying to transfer funds and " \
            "only get an error message., What are the reasons a beneficiary would be denied?, A transfer to my " \
            "account was denied"
comments = [comment_1, comment_2, comment_3]

for i in range(len(comments)):
    prompt_1 = f"""
  What is the sentiment of the following comments, 
  which is delimited with triple backticks?

  Give your answer as a single word, either "positive" \
  or "negative".

  Review text: '''{comments[i]}'''
  """
    sentiment = get_completion(prompt_1)
    print(i, sentiment, "\n")

    prompt_2 = f"""
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
  Customer review: ```{comments[i]}```
  Review sentiment: {sentiment}
  """
    response = get_completion(prompt_2)
    print(i, response, "\n")
