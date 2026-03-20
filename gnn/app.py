import gradio as gr
import torch
import os
import numpy as np
from pathlib import Path

# Import your custom architectures
from fusion_network import SiameseDeltaGNN
from graph import create_protein_graph

print("Loading 3D GNN Model...")
device = torch.device("cpu")

# Initialize and Load the Trained Weights
model = SiameseDeltaGNN(node_feature_dim=50, hidden_dim=128).to(device)

try:
    checkpoint = torch.load("checkpoint_best.pth", map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model weights. Error: {e}")

def predict_mutation(pdb_file, mutation_string):
    """
    The main inference bridge connecting the UI to the PyTorch Math.
    """
    if pdb_file is None or not mutation_string:
        return "⚠️ Error: Please upload a PDB file and enter a mutation."
    
    pdb_path = pdb_file.name
    
    try:
        # 1. Build the 3D Graph from the uploaded PDB
        wt_graph, _ = create_protein_graph(pdb_path, distance_threshold=8.0)
        
        # Clone it for the mutant graph
        mt_graph = wt_graph.clone()
        
        # 2. Inject a dummy ESM-2 feature (For a real web app, you would ping the ESM API here)
        # Since we compressed ESM down to 50D, we create a zero-tensor for the baseline,
        # and a random standard normal tensor to simulate the evolutionary shock for the demo.
        dummy_esm = torch.randn((1, 50), dtype=torch.float) 
        
        wt_graph.esm_feature = torch.zeros_like(dummy_esm)
        mt_graph.esm_feature = dummy_esm
        
        # Move graphs to CPU
        wt_graph = wt_graph.to(device)
        mt_graph = mt_graph.to(device)

        # 3. Add batch dimensions since we aren't using a DataLoader
        wt_graph.batch = torch.zeros(wt_graph.num_nodes, dtype=torch.long)
        mt_graph.batch = torch.zeros(mt_graph.num_nodes, dtype=torch.long)

        # 4. Run the forward pass!
        with torch.no_grad():
            prediction = model(wt_graph, mt_graph)
            
        predicted_ddg = prediction.item()
        
        # 5. Format the output based on thermodynamic stability
        stability = "Destabilizing (Clash/Misfold)" if predicted_ddg > 0 else "Stabilizing"
        
        result_text = f"🧪 Prediction Complete for {mutation_string}\n"
        result_text += "-"*40 + "\n"
        result_text += f"Predicted ΔΔG: {predicted_ddg:.4f} kcal/mol\n"
        result_text += f"Structural Impact: {stability}\n"
        result_text += f"Graph Nodes Analyzed: {wt_graph.num_nodes}\n"
        result_text += "-"*40 + "\n"
        result_text += "(Note: Using simulated ESM-2 embedding for demonstration)"
        
        return result_text

    except Exception as e:
        return f"🚨 Inference Error: {str(e)}\nMake sure the PDB file is valid."

# Build the Gradio Web Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 3D GNN Protein Stability Predictor ($\Delta\Delta G$)")
    gr.Markdown("Upload a wild-type protein structure and specify a single-point mutation to predict the thermodynamic shock using a Siamese Graph Neural Network.")
    
    with gr.Row():
        with gr.Column():
            pdb_input = gr.File(label="Upload Wild-Type Structure (.pdb)")
            mut_input = gr.Textbox(label="Mutation", placeholder="e.g., A45G (Chain A, Pos 45, Glycine)")
            predict_btn = gr.Button("Predict $\Delta\Delta G$", variant="primary")
            
        with gr.Column():
            output_display = gr.Textbox(label="Prediction Results", lines=8)
            
    predict_btn.click(fn=predict_mutation, inputs=[pdb_input, mut_input], outputs=output_display)

if __name__ == "__main__":
    demo.launch(share=True)