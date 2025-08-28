# %% [markdown]
"""
# AI Agent

Taking the bull by the horns

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

For this practical work, you need the following python packages:

-   `openai`
-   `python-dotenv`
-   `faiss-cpu`
-   `numpy`

# Hello World

Make work the example of the course.

``` python
from openai import OpenAI
client = OpenAI()

chat_response = client.chat.completions.create(
    model= "gpt-4o",
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)
```

Look at the documentation of the `OpenAI()` constructor in order to take
your own model. Modify the model name accordingly.

> **Important**
>
> **Never, ever** put your API key in the code. Use environment
> variables instead. For example, use python `dotenv` to load the API
> key from a `.env` file.

> **Tip**
>
> The openai compatible endpoint for mistral.ai is
> `https://api.mistral.ai/v1`. `mistral-small-latest` as the model
> should be sufficient.

> **Tip**
>
> If you have locally installed llm with ollama/lmstudio for example,
> don’t hesitate to adapt the code to use your local model.

# Build a RAG Agent

## Split the document in chunks

Get the alice.txt and split it in 2048 characters.

## Encode the chunks

A simple function to get an embedding is:
"""

# %%
from time import sleep

def get_text_embedding(input):
    sleep(1) # Rate-limiting
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response.data[0].embedding


# %% [markdown]
"""
It does return a 1024 list of float (the embedding of the input).

make a numpy array of all chunk embeddings from the text.

(Should take one and half minute)

## Store embeddings in vector database
"""

# %%
import faiss

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# %% [markdown]
"""
# Query

## Make an example query

Make an embedding for a question like “À quels obstacles est confrontée
Alice?”

## Search for the most similar chunks
"""

# %%
D, I = index.search(question_embeddings, k=2) # distance, index

# %% [markdown]
"""
## Retrieve the chunks
"""

# %%
#! tags: [solution]
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

# %% [markdown]
"""
## RAG prompt

Make the RAG query with the following prompt
"""

# %%
prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""


# %%
def run_mistral(user_message, model="mistral-large-latest"):
    sleep(1) # Rate-limit
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

run_mistral(prompt)

# %% [markdown]
"""
# Final

Make a function for any question about the book.

##
"""
