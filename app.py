import os
import requests
import json
import math
import numpy as np
from groq import Groq, Stream
# No need for ChatCompletionChunk import if not directly used for typing
from dotenv import load_dotenv
import time
import traceback # Import traceback for better error logging

# --- Configuration ---
load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BRAVE_API_KEY:
    raise ValueError("BRAVE_API_KEY not found in environment variables.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# --- Step 1: Integrate Brave Search API (Optimized) ---
def brave_search(query: str, count: int = 3) -> list[dict] | None: # Return None on failure
    """
    Performs a search using the Brave Search API (optimized for speed).
    Returns a list of results or None if a critical error occurs.
    """
    print(f"--- Performing Brave Search (Max {count} results)...")
    t_start = time.time()
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
        "text_decorations": False,
        "safesearch": "moderate",
    }
    if not BRAVE_API_KEY:
         print("Error: BRAVE_API_KEY is not set!")
         return None # Indicate failure
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=5 # Slightly increased timeout for reliability
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        search_data = response.json()
        results = []
        if "web" in search_data and "results" in search_data["web"]:
            for item in search_data["web"]["results"]:
                if item.get("description"): # Prioritize results with snippets
                     results.append({
                        "title": item.get("title", "No Title"),
                        "url": item.get("url", "No URL"),
                        "snippet": item.get("description")
                     })
        t_end = time.time()
        print(f"--- Brave Search completed in {t_end - t_start:.2f}s ({len(results)} results with snippets) ---")
        return results # Return list (could be empty)

    except requests.exceptions.Timeout:
        print(f"Error: Brave Search API call timed out.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"Error: Brave Search HTTP Error: {http_err}")
        print(f"Response body: {response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Error: Brave Search Request Error: {req_err}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from Brave Search.")
        print(f"Response text: {response.text if 'response' in locals() else 'Response object not available'}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Brave Search: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        return None

# --- Step 2: Preprocess Retrieved Documents (Remains Fast) ---
def preprocess_context(search_results: list[dict] | None) -> str:
    """
    Formats the search results into a single string context.
    Handles None input gracefully.
    """
    print("--- Preprocessing context...")
    t_start = time.time()
    if not search_results: # Checks for None or empty list
        print("--- No search results provided for context. ---")
        return "No relevant context found from search."

    context = "Relevant Information:\n"
    for i, result in enumerate(search_results):
        # Basic sanitation - remove excessive newlines/whitespace from snippet
        snippet = ' '.join(result.get('snippet', '').split())
        title = result.get('title', 'Source')
        context += f"\nSource {i+1} ({title}):\n"
        context += f"{snippet}\n"

    t_end = time.time()
    print(f"--- Context preprocessing completed in {t_end - t_start:.4f}s (length {len(context)}) ---")
    return context.strip()

# --- Step 3: Generate Responses with Groq API (Streaming) ---
def generate_with_groq_streaming(
    user_query: str,
    context: str,
    model: str = "llama3-8b-8192", # Corrected model name
    max_tokens: int = 512
) -> Stream | None:
    """
    Initiates a streaming response generation using the Groq API.
    Returns a Stream object or None on failure.
    """
    print(f"--- Initiating Groq stream ({model}, max_tokens={max_tokens})... ---")
    client = Groq(api_key=GROQ_API_KEY)

    # Consider making the prompt slightly more robust
    prompt = f"""Please provide a comprehensive and informative answer to the user's query based *primarily* on the Relevant Information provided below. Cite sources using [Source X] notation where the information is used. If the context does not contain the answer, state that you couldn't find the information in the provided sources and answer based on your general knowledge, clearly indicating the shift. Be concise.

Relevant Information:
{context}

---
User Query: {user_query}

Answer:"""

    try:
        print(f"DEBUG: Attempting Groq call with model: {model}")
        stream = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a large language model trained by Perplexity AI. Write an accurate answer concisely for a given question, citing the search results as needed. Your answer must be correct, high-quality, and written by an expert using an unbiased and journalistic tone. Your answer must be written in the same language as the question, even if language preference is different. Cite search results using [index] at the end of sentences when needed, for example \"Ice is less dense than water.[1][2]\" NO SPACE between the last word and the citation. Cite the most relevant results that answer the question. Avoid citing irrelevant results. Write only the response. Use markdown for formatting.\n\nUse markdown to format paragraphs, lists, tables, and quotes whenever possible.\nUse markdown code blocks to write code, including the language for syntax highlighting.\nUse LaTeX to wrap ALL math expression. Always use double dollar signs $$, for example \n\n$$x^4 = x - 3.$$\n\nDO NOT include any URL's, only include citations with numbers, eg [1].\nDO NOT include references (URL's at the end, sources).\nUse footnote citations at the end of applicable sentences(e.g, [1][2]).\nWrite more than 100 words (2 paragraphs).\nIn the response avoid referencing the citation directly\nPrint just the response text."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=1,
            stop=None,
            stream=True,
            logprobs=False,
        )
        print("DEBUG: Groq stream object created successfully.")
        return stream

    except Exception as e:
        print(f"ERROR during Groq API call/stream initiation: {e}")
        traceback.print_exc() # Print detailed traceback
        return None

# --- Step 4: Compute Perplexity (REMOVED if logprobs=False) ---
# If you set logprobs=False in generate_with_groq_streaming,
# you cannot calculate perplexity. Remove or comment out this function
# and related calls if speed is the absolute priority.

# def calculate_perplexity(logprobs: list[float] | None) -> float | None:
#     """ Calculates perplexity... (Keep only if logprobs=True) """
#     # ... (implementation from before) ...
#     pass

# --- Step 5: Optimization Guidance (REMOVED if logprobs=False) ---
# Remove or comment out if perplexity isn't calculated.

# def explain_optimization(perplexity: float | None):
#     """ Provides guidance... (Keep only if logprobs=True) """
#     # ... (implementation from before) ...
#     pass


# --- Main Execution (Handles Streaming) ---
if __name__ == "__main__":
    user_query = input("Enter your query: ")

    if not user_query:
        print("Query cannot be empty.")
    else:
        # --- Initialize all variables used later ---
        total_start_time = time.time()
        search_results = None
        context = "Error during context preparation or search failure." # Default error context
        stream = None
        full_response = ""
        first_token_time = None
        stream_start_time = None # Initialize here
        stream_end_time = None   # Initialize here
        # collected_logprobs = [] # Only needed if logprobs=True
        # logprobs_available_post_stream = False # Only needed if logprobs=True

        print("DEBUG: Starting RAG process...")

        # --- 1. Retrieve ---
        print("DEBUG: Calling brave_search...")
        search_results = brave_search(user_query, count=3)
        print(f"DEBUG: brave_search returned type: {type(search_results)}")
        if search_results is not None:
            print(f"DEBUG: brave_search returned {len(search_results)} results.")
        else:
            print("DEBUG: brave_search returned None (search failed).")


        # --- 2. Preprocess (only if search succeeded) ---
        if search_results is not None: # Check if search didn't fail critically
            print("DEBUG: Calling preprocess_context...")
            context = preprocess_context(search_results) # Handles empty list gracefully
            print(f"DEBUG: preprocess_context completed.")
        else:
            print("DEBUG: Skipping context preprocessing due to search failure.")


        # --- 3. Generate (only if context seems okay) ---
        # We proceed even if search returned [] but not None, using default context
        if context != "Error during context preparation or search failure.":
            print("DEBUG: Calling generate_with_groq_streaming...")
            stream = generate_with_groq_streaming(user_query, context, model="llama3-8b-8192", max_tokens=512)
            print(f"DEBUG: generate_with_groq_streaming returned: {'Stream object' if stream else 'None'}")
        else:
             print("DEBUG: Skipping generation due to error in search/context phase.")


        # --- Process Stream ---
        if stream:
            print("\n--- Groq Response (Streaming) ---")
            stream_start_time = time.time() # Start timer just before loop
            try:
                for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    delta = choice.delta if choice else None
                    content = delta.content if delta else None

                    if content:
                        if first_token_time is None: # Record time of first content
                            first_token_time = time.time()
                            print(f"\n--- Time to first token: {first_token_time - stream_start_time:.3f}s ---")
                        print(content, end="", flush=True)
                        full_response += content

                    # --- Logprob collection logic (REMOVED for simplicity/speed) ---
                    # if logprobs=True, you would add logic here to parse
                    # chunk.choices[0].logprobs and append to collected_logprobs
                    # This adds latency and complexity.

            except Exception as e:
                print(f"\n--- ERROR during streaming loop: {e} ---")
                traceback.print_exc() # See where the error happened in the loop
            finally:
                print() # Ensure newline after stream output
                stream_end_time = time.time() # Record end time after loop finishes/errors
                if stream_start_time: # Avoid calculating duration if loop never started
                     print(f"--- Streaming finished or errored in {stream_end_time - stream_start_time:.2f}s ---")
                # --- Post-stream logprob access (REMOVED for simplicity/speed) ---
                # if logprobs=True: attempt to access aggregated logprobs here if SDK provides it


        else: # Handle stream initiation failure
            if search_results is None:
                 print("\nERROR: Failed to initiate Groq stream because Brave Search failed.")
            elif context == "Error during context preparation or search failure.":
                 print("\nERROR: Failed to initiate Groq stream due to context preparation error.")
            else:
                 print("\nERROR: Failed to initiate Groq stream. Check Groq API key, model name ('llama3-8b-8192'), and connection.")


        # --- Final Output & Timings ---
        total_end_time = time.time()
        print(f"\n--- Full response assembled (length: {len(full_response)}) ---")
        # print(full_response) # Optionally print the full response again if needed

        print(f"\n--- Total RAG process time: {total_end_time - total_start_time:.2f}s ---")

        # --- 4/5. Perplexity Calculation & Optimization (REMOVED if logprobs=False) ---
        # if collected_logprobs: # Only if logprobs=True and collection was implemented
        #      perplexity_score = calculate_perplexity(collected_logprobs)
        #      if perplexity_score is not None:
        #          explain_optimization(perplexity_score)
        #      else:
        #          print("\nPerplexity calculated as None.")
        # else:
        print("\nPerplexity calculation skipped (logprobs=False or collection failed).")