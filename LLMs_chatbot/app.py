import gradio as gr
import os
from openai import OpenAI

client = OpenAI(
    api_key="binhphap5",
    base_url="http://localhost:8000/v1",
)

# Chat function
def respond(
    message, history, system_message, max_tokens, temperature, top_p
):
    """
    A chat function that responds to user input based on the given history and
    system message. The chat function uses the OpenAI API to generate text
    based on the input message and the given history and system message.

    :param message: The user's input message
    :param history: A list of previous messages in the conversation
    :param system_message: The system message for the chat
    :param max_tokens: The maximum number of tokens to generate
    :param temperature: The temperature to use when generating text
    :param top_p: The top p value to use when generating text
    :return: A generator that yields a dictionary for each new message
    """
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model="binhphap5/Qwen2.5-3B-Instruct-Chat-RLHF",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            response += token
        yield {"role": "assistant", "content": response}

# UI
chat = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly chatbot.", label="System message"),
        gr.Slider(
            1, 2048, value=512, step=1, label="Max new tokens"
        ),
        gr.Slider(
            0.1, 4.0, value=0.7, step=0.1, label="Temperature"
        ),
        gr.Slider(
            0.1, 1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"
        ),
    ],
    type="messages",
)

with gr.Blocks() as demo:
    chat.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
