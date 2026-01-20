# === Imports & Setup ===
import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import networkx as nx
import uuid

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# === Gemini Caller ===
def call_gemini(system_prompt, user_prompt, language, temperature=0.5, max_output=400):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    user_prompt = f"{user_prompt}\n\nRespond in {language}."
    response = model.generate_content(
        [system_prompt, user_prompt],
        generation_config={"temperature": temperature, "max_output_tokens": max_output},
        stream=True
    )
    full_text = ""
    for chunk in response:
        if chunk.text:
            full_text += chunk.text
            yield full_text
    yield full_text

# === Agents ===
def run_ceo(problem, constraints, language, temp):
    return call_gemini("You are CEO Agent. Generate concise product ideas.",
                       f"Problem: {problem}\nConstraints: {constraints}", language, temp)

def run_researcher(idea, language, temp):
    return call_gemini("You are Research Agent. Provide background and competitors.",
                       f"Idea to research: {idea}", language, temp)

def run_coder(idea, research, language, temp):
    return call_gemini("You are Coder Agent. Suggest an implementation approach.",
                       f"Idea: {idea}\nResearch: {research}", language, temp)

def run_critic(idea, research, codeplan, language, temp):
    return call_gemini("You are Critic Agent. Analyze risks and flaws.",
                       f"Idea: {idea}\nResearch: {research}\nCode Plan: {codeplan}", language, temp)

# === Pipeline ===
def multi_agent_pipeline(problem, constraints, language, temp):
    ceo_out = "".join(run_ceo(problem, constraints, language, temp))
    research_out = "".join(run_researcher(ceo_out, language, temp))
    coder_out = "".join(run_coder(ceo_out, research_out, language, temp))
    critic_out = "".join(run_critic(ceo_out, research_out, coder_out, language, temp))
    history_id = str(uuid.uuid4())[:8]
    return ceo_out, research_out, coder_out, critic_out, f"Session-{history_id}"

# === Visualization ===
def generate_workflow_graph():
    G = nx.DiGraph()
    edges = [("Problem", "CEO"), ("CEO", "Researcher"), ("Researcher", "Coder"), ("Coder", "Critic")]
    G.add_edges_from(edges)
    plt.figure(figsize=(6,4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="#90caf9", font_size=10, arrowsize=20, font_weight="bold")
    img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(img_file.name, bbox_inches="tight")
    plt.close()
    return img_file.name

# === Gradio UI ===
with gr.Blocks(css="""
    #title {text-align: center; font-size: 2em; font-weight: bold;
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .output-card {border-radius: 15px; padding: 15px; background: #f9f9f9; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);}
""") as demo:
    
    gr.Markdown("<div id='title'>🤖 Multi-Agent Problem Solver</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Input Settings")
            problem_input = gr.Textbox(label="Problem Statement", placeholder="Describe the problem here...")
            constraints_input = gr.Textbox(label="Constraints", placeholder="E.g., Budget < $5000")
            language_input = gr.Dropdown(
                choices=["English", "Hindi", "Marathi", "Spanish", "French", "German", "Chinese", "Japanese"],
                value="English", label="🌍 Output Language"
            )
            temp_slider = gr.Slider(0, 1, value=0.5, step=0.1, label="🎨 Creativity (Temperature)")
            run_btn = gr.Button("🚀 Run Agents", variant="primary")
        
    with gr.Column(scale=2):
        gr.Markdown("### 📊 Agent Outputs")
        with gr.Row():
            ceo_output = gr.Textbox(label="🧑‍💼 CEO", elem_classes="output-card")
            research_output = gr.Textbox(label="🔎 Researcher", elem_classes="output-card")
        with gr.Row():
            coder_output = gr.Textbox(label="💻 Coder", elem_classes="output-card")
            critic_output = gr.Textbox(label="🧐 Critic", elem_classes="output-card")

    with gr.Accordion("📜 Extra Features", open=False):
        history_box = gr.Textbox(label="Session ID")
        workflow_graph = gr.Image(label="Workflow Graph", type="filepath")

    # Button action
    run_btn.click(
        fn=multi_agent_pipeline,
        inputs=[problem_input, constraints_input, language_input, temp_slider],
        outputs=[ceo_output, research_output, coder_output, critic_output, history_box]
    ).then(
        fn=generate_workflow_graph,
        inputs=[],
        outputs=workflow_graph
    )

demo.launch()
