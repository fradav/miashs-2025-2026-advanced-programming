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

and also a nicely format function for markdown output of LLM:
"""

# %%
from IPython.display import Markdown, display

def format_markdown_output(response):
    # Create a boxed effect using HTML and CSS
    boxed_content = f"""
<div style="border: 2px solid #f15006ff; padding: 10px; border-radius: 5px; background-color: #493f3fff;">
    {response}
</div>
"""
    return display(Markdown(boxed_content))


# %% [markdown]
"""
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

Get the
[alice.txt](https://fradav.github.io/miashs-2025-2026-advanced-programming/docs-resources/Notebooks/materials/alice.txt)
and split it in 2048 characters.

## Encode the chunks

A simple function to get an embedding is:
"""

# %%
from time import sleep

def get_text_embedding(input):
    sleep(0.5) # Rate-limiting
    embeddings_response = client.embeddings.create(
          model="mistral-embed",
          input=input
      )
    return embeddings_response.data[0].embedding


# %% [markdown]
"""
It does return a 1024 list of float (the embedding of the input).

Make a `numpy` array of all chunk embeddings from the text. Save the
array to a file to avoid recomputing it.

(Should take one and half minute)

> **Tip**
>
> An optimal version of this should use the [batch
> feature](https://platform.openai.com/docs/guides/batch) of OpenAI APi

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

## Search for the most semantically similar chunks
"""

# %%
D, I = index.search(question_embeddings, k=2) # distance, index

# %% [markdown]
"""
## Retrieve the chunks

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
def run_mistral(user_message, model="mistral-small-latest"):
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

format_markdown_output(run_mistral(prompt))

# %% [markdown]
"""
# Put it together

Make a function for any question about the book.

## Areas for improvement

-   Chunking strategies: chunk sizes, overlap, metadata
-   LLM parameters: context size, temperature, top_p, etc.
-   reranking
-   hybrid retrieval

# MCP server

## How to test/install a python MCP server

As simple mcp server requires to be just a complete one-liner script to
run, simply create a new python file (e.g. `mcp_server.py`) and add the
following code:

``` python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyMCPServer")

@mcp.tool()
def my_tool():
    """
    This is my tool, it does nothing.

    Args:
        None

    Returns:
        dict: A dictionary with the status of the tool.
    """
    return {"message": "Hello from my_tool!"}
```

``` bash
uv run --with fastmcp mcp run path/to/mcp_server.py
```

in a yaml/json file for mcp server:

## YAML

``` yaml
    command: uv
    args:
      - run
      - --with
      - fastmcp
      - mcp
      - run
      - mcp_server.py
```

## JSON

``` json
{
  "command": "uv",
  "args": [
    "run",
    "--with",
    "fastmcp",
    "mcp",
    "run",
    "mcp_server.py"
  ]
}
```

## First draft

We want to make an MCP server which logs useful stuff in standard
logging format to a file specified by environment variable
`MCP_LOGGING_FILE`.

Use [FastMCP](https://gofastmcp.com/getting-started/welcome) as MCP SDK.

We should use standard `logging` python package.

## Refining

We want to let the LLM choose the logging level

## Playing with it

Make a rule to use the tool at each LLM request, test it. Pay attention
to the workflow, especially with devstral.

<span class="proof-title">*Solution*. </span>

``` markdown
# Logging each answer

Each time your answer:
- Choose a concise summary of your answer
- Choose a level of importance `INFO`, `WARNING`, `ERROR`
- log that
- after logging, if successful, just say "Logged."

DO NOT FORGET TO LOG EACH TIME
```
"""
