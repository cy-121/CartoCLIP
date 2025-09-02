import gradio as gr
import torch
import clip
import os
import warnings
import numpy as np
from PIL import Image

# Ignore warning.
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

# Load the device and model.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the weights of the CartoCLIP model.
model_path = './model/CartoCLIP.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Switch to evaluation mode

# Prepare the texts of cartographic representations.
texts = [
    "This thematic map is a choropleth map.",
    "This thematic map is a quality base map.",
    "This thematic map is a flow map.",
    "This thematic map is a proportional symbol map.",
]
text_tokens = clip.tokenize(texts).to(device)

# Texts for display
display_texts = [
    "This thematic map is a choropleth map.",
    "This thematic map is a quality-based map.",
    "This thematic map is a flow map.",
    "This thematic map is a proportional symbol map.",
]

# Encapsulate identification function.
def identify_expression_method(query_image):
    # Convert NumPy arrays to PIL images.
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray(query_image)
    
    # Prepare the images for query.
    query_image = preprocess(query_image).unsqueeze(0).to(device)

    # Extract the embedding vectors of the query images.
    with torch.no_grad():
        query_features = model.encode_image(query_image)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    # Extract text embeddings.
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # Normalize the embedding vectors.
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity.
    similarity = (query_features @ text_features.T).cpu().numpy().flatten()

    # Obtain the highest similarity.
    max_similarity_index = np.argmax(similarity)
    max_similarity = similarity[max_similarity_index]
    expression_method = texts[max_similarity_index]

    # Construct the output results.
    table_rows = []
    for i, text in enumerate(display_texts):
        table_rows.append(f'<tr><td style="font-size: 20px;">{text}</td><td style="font-size: 24px;">{similarity[i]:.4f}</td></tr>')
    table_html = f'<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">' \
                 f'<thead><tr><th style="font-size: 24px; text-align: left; background-color: #eeeeee;">Cartographic Representation</th><th style="font-size: 24px; text-align: left; background-color: #eeeeee;">Similarity</th></tr></thead>' \
                 f'<tbody>{"".join(table_rows)}</tbody></table>'

    result_phrase = display_texts[max_similarity_index].split(" is ")[1].strip(".")
    summary = f'<p style="font-size: 28px; margin-top: 40px; font-weight: bold;">So this thematic map is {result_phrase}.</p>'

    return table_html + "\n" + summary

# Input and output components
input_component = gr.Image(label="Upload a Thematic Map")
output_expression_method = gr.HTML(label="Cartographic Representation")

# Load all images in the examples folder as examples.
examples_folder = './examples'
example_files = [os.path.join(examples_folder, f) for f in os.listdir(examples_folder) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]

# Create a Gradio interface.
demo = gr.Interface(
    fn=identify_expression_method,
    inputs=input_component,
    outputs=output_expression_method,
    examples=example_files,
    title="Cartographic Representation Identification",
    description="Upload a thematic map to identify the cartographic representation."
)

# Launch the Gradio application.
demo.launch(server_name="0.0.0.0")