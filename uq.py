import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from uqlm import WhiteBoxUQ
from uqlm.utils import load_example_dataset, math_postprocessor, plot_model_accuracies, Tuner

import asyncio
# Load example dataset (gsm8k)
gsm8k = load_example_dataset("gsm8k", n=100)
gsm8k.head()

MATH_INSTRUCTION = "When you solve this math problem only return the answer with no additional text.\n"
prompts = [MATH_INSTRUCTION + prompt for prompt in gsm8k.question]

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI

from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings)

os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", logprobs=True)


wbuq = WhiteBoxUQ(llm=llm)

async def say_hello():
    results = await wbuq.generate_and_score(prompts=prompts)

if __name__ == "__main__":
    results = asyncio.run(say_hello())

    result_df = results.to_df()
    print(result_df.head())
