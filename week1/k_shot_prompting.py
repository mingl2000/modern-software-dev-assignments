import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
#YOUR_SYSTEM_PROMPT = "You are a prefect assistant that reverses words accurately. Check your work by reversing the word again to ensure you get the input string."
YOUR_SYSTEM_PROMPT ="""
You reverse a word by mapping each character position to its reversed position.

For any word W = w1 w2 w3 ... wn  
you MUST output:  wn ... w3 w2 w1

To ensure correctness, follow these exact examples:

apple
=> elppa

robot
=> tobor

coffee
=> eeffoc

matrix
=> xirtam

httpstatus
=> sutatsptth

RULES:
- Output MUST be only the reversed characters.
- NO explanation.
- NO sentences.
- NO punctuation.
- NO extra spaces.
- NO arrow, no labels.
- Only the reversed string.

"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            #model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)