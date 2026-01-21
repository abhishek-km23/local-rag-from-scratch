import sys
import os

# Ensure project root is on Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm import load_llm

try:
    llm = load_llm()

    response = llm.invoke("Say hello in one sentence.")

    print("LLM response:")
    print(response)

except Exception as e:
    print("ERROR:", e)
