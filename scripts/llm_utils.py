from config import Config
from llama_cpp import Llama

cfg = Config()
llm = Llama(model_path=cfg.smart_llm_model, n_ctx=2048, embedding=True)

def create_chat_completion(messages, model=None, temperature=0.36, max_tokens=0) -> str:
    """Generate a reply using the complete conversation and its message roles."""
    role_labels = {"system": "System", "user": "Human", "assistant": "Assistant"}
    # Keep every system entry: these also contain memories and command results.
    prompt = "\n\n".join(
        f"### {role_labels.get(message['role'], message['role'].capitalize())}:\n"
        f"{message['content']}"
        for message in messages
    ) + "\n\n### Assistant:\n"
    response = llm(
        prompt,
        stop=["Q:", "### Human:"],
        echo=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["text"]
