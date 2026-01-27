import gradio as gr
import os
from dotenv import load_dotenv
import tempfile
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import google.generativeai as genai

# ===================== ENV =====================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing")

genai.configure(api_key=GOOGLE_API_KEY)

# ===================== GEMINI =====================
def call_gemini(system_prompt, user_prompt, temperature):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"{system_prompt}\n\n{user_prompt}",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 2048,
        },
    )
    return response.text

# ===================== GRAPH =====================
def generate_workflow_image():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Problem", "CEO"),
        ("CEO", "Researcher"),
        ("Researcher", "Coder"),
        ("Coder", "Critic"),
    ])

    plt.figure(figsize=(8, 5))
    nx.draw(G, with_labels=True, node_size=3000)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name

# ===================== LOGIC =====================
def run_agents(problem, constraints, language, creativity):
    if not problem.strip():
        return "", "", "", "", "", None

    ceo = call_gemini("CEO agent", problem, creativity)
    research = call_gemini("Research agent", ceo, creativity)
    coder = call_gemini("Coder agent", research, creativity)
    critic = call_gemini("Critic agent", coder, creativity)

    return (
        ceo,
        research,
        coder,
        critic,
        f"Session-{uuid.uuid4().hex[:8]}",
        generate_workflow_image(),
    )

# ===================== UI =====================
with gr.Blocks(
    title="Multi-Agent Solver",
    analytics_enabled=False,   # 🔒 critical
) as demo:

    gr.Markdown("# 🤖 Multi-Agent Problem Solver")

    problem = gr.Textbox(label="Problem", lines=3)
    constraints = gr.Textbox(label="Constraints", lines=2)
    language = gr.Dropdown(["English", "Hindi"], value="English")
    creativity = gr.Slider(0, 1, value=0.7)

    run = gr.Button("Run")

    ceo_out = gr.Textbox(label="CEO")
    res_out = gr.Textbox(label="Research")
    code_out = gr.Textbox(label="Code")
    critic_out = gr.Textbox(label="Critic")
    session = gr.Textbox(label="Session ID")
    image = gr.Image(type="filepath")

    demo.queue(api_open=False)   # 🔒 critical

    run.click(
        fn=run_agents,
        inputs=[problem, constraints, language, creativity],
        outputs=[ceo_out, res_out, code_out, critic_out, session, image],
        api_name=False,          # 🔒 critical
    )

# ===================== LAUNCH =====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
