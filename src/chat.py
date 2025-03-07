import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model.
# Change the model_name if needed (e.g. "meta-llama/Llama-3.1-8B-Instruct")

def chat_with_llama(prompt, model, tokenizer, chat_history=None, max_new_tokens=50):
    """
    Chat with a Llama 3.1 model using the transformers generate() method.

    Parameters:
      prompt (str): The user's new message.
      chat_history (list): Conversation history, a list of dicts with "role" and "content".
                           If None, the conversation starts with a default system prompt.
      max_length (int): Maximum token length for the generated output.

    Returns:
      assistant_reply (str): The new response generated by the model.
      chat_history (list): Updated conversation history including the new messages.
    """
    # Start with a default system prompt if no history exists.
    if chat_history is None:
        chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Append the user's new prompt.
    chat_history.append({"role": "user", "content": prompt})
    
    # Build the conversation text from history.
    conversation = ""
    for message in chat_history:
        if message["role"] == "system":
            conversation += "System: " + message["content"] + "\n"
        elif message["role"] == "user":
            conversation += "User: " + message["content"] + "\n"
        elif message["role"] == "assistant":
            conversation += "Assistant: " + message["content"] + "\n"
    
    # Encode the conversation as input IDs.
    input_ids = tokenizer.encode(conversation, return_tensors="pt")
    with torch.no_grad():
        
        # Generate a continuation using the model.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id  # Ensure proper padding.
        )
    
    # Decode the generated tokens.
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract the new assistant response.
    # We assume the assistant’s reply starts after the last "Assistant:" marker.
    if "Assistant:" in generated_text:
        assistant_reply = generated_text.split("Assistant:")[-1].strip()
    else:
        # Fallback: take the text that comes after the input prompt.
        assistant_reply = generated_text[len(conversation):].strip()
    
    # Append the assistant's reply to the conversation history.
    chat_history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply

