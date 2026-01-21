from langchain_ollama import OllamaLLM

def load_llm(model_name="llama3", temperature=0.0):
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=temperature
        )
        return llm

    except Exception as e:
        raise RuntimeError(f"Failed to load LLM '{model_name}': {str(e)}")
