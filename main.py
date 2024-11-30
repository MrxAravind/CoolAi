import os
import gradio as gr
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


GitHub = os.getenv('Token') 


# List of available models
MODELS = [
    "Meta-Llama-3-8B-Instruct",
    "Cohere-command-r-plus-08-2024",
    "Ministral-3B",
    "AI21-Jamba-1.5-Large",
    "Phi-3.5-MoE-instruct",
    "gpt-4o-mini",
    "gpt-4o"
]

def run_inference(model, query, temperature, max_tokens):
    """Run inference on selected model"""
    client = ChatCompletionsClient(
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(GitHub),
    )

    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful AI assistant."),
                UserMessage(content=query),
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            model_dropdown = gr.Dropdown(MODELS, label="Select Model")
            temperature_slider = gr.Slider(0, 1, value=0.8, label="Temperature")
            max_tokens_slider = gr.Slider(64, 4096, value=2048, label="Max Tokens")
        
        query_input = gr.Textbox(label="Enter your query")
        submit_btn = gr.Button("Generate Response")
        output = gr.Textbox(label="Model Response")

        submit_btn.click(
            fn=run_inference,
            inputs=[model_dropdown, query_input, temperature_slider, max_tokens_slider],
            outputs=output
        )

    return demo

def main():
    demo = gradio_interface()
    demo.launch(server_port=80)

if __name__ == "__main__":
    main()
