import gradio as gr
import google.generativeai as genai
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import time
import json
from datetime import datetime
import io
import base64
import numpy as np

# ─────────────────────────────────────────────
# Configure Gemini
# ─────────────────────────────────────────────
def get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# ─────────────────────────────────────────────
# Agent Definitions
# ─────────────────────────────────────────────
AGENTS = {
    "CEO": {
        "icon": "👔",
        "color": "#6C63FF",
        "role": "Strategic Planner & Coordinator",
        "system": (
            "You are the CEO Agent — a master strategist. Your job is to:\n"
            "1. Analyze the problem at a high level.\n"
            "2. Break it into clear sub-tasks for specialized agents.\n"
            "3. Define success criteria.\n"
            "4. Output a concise strategic plan (max 300 words) with numbered action items.\n"
            "Be executive: sharp, structured, decisive."
        ),
    },
    "Research": {
        "icon": "🔬",
        "color": "#00C9A7",
        "role": "Information Gatherer & Analyst",
        "system": (
            "You are the Research Agent — a deep-dive analyst. Given the CEO's strategic plan and the original problem:\n"
            "1. Identify key concepts, relevant knowledge, and potential approaches.\n"
            "2. Highlight important considerations, trade-offs, and edge cases.\n"
            "3. Provide relevant background that will help solve the problem.\n"
            "4. Output structured research findings (max 400 words).\n"
            "Be thorough but focused. Surface insights that matter."
        ),
    },
    "Coder": {
        "icon": "💻",
        "color": "#FF6B6B",
        "role": "Technical Implementer",
        "system": (
            "You are the Coder Agent — a pragmatic engineer. Given the problem, CEO plan, and research:\n"
            "1. Produce a concrete technical solution, code snippet, algorithm, or step-by-step implementation.\n"
            "2. If code is needed, write clean, commented, working code.\n"
            "3. If it's not a coding problem, provide a detailed technical/procedural solution.\n"
            "4. Include error handling and edge case handling where relevant.\n"
            "Be precise. Build something that works."
        ),
    },
    "Critic": {
        "icon": "🎯",
        "color": "#FFD93D",
        "role": "Reviewer & Quality Enhancer",
        "system": (
            "You are the Critic Agent — a rigorous quality reviewer. Given all previous agent outputs:\n"
            "1. Identify weaknesses, gaps, or flaws in the proposed solution.\n"
            "2. Suggest specific improvements.\n"
            "3. Provide a final synthesized, polished answer that incorporates the best of all agents.\n"
            "4. Rate the overall solution quality (1-10) with justification.\n"
            "Be constructive but honest. Elevate the final output."
        ),
    },
}

# ─────────────────────────────────────────────
# Agent Runner
# ─────────────────────────────────────────────
def run_agent(agent_name: str, problem: str, context: dict, model) -> str:
    agent = AGENTS[agent_name]
    
    context_str = ""
    if agent_name == "Research":
        context_str = f"\n\n--- CEO Strategic Plan ---\n{context.get('CEO', '')}"
    elif agent_name == "Coder":
        context_str = (
            f"\n\n--- CEO Strategic Plan ---\n{context.get('CEO', '')}"
            f"\n\n--- Research Findings ---\n{context.get('Research', '')}"
        )
    elif agent_name == "Critic":
        context_str = (
            f"\n\n--- CEO Strategic Plan ---\n{context.get('CEO', '')}"
            f"\n\n--- Research Findings ---\n{context.get('Research', '')}"
            f"\n\n--- Technical Solution ---\n{context.get('Coder', '')}"
        )

    prompt = (
        f"{agent['system']}\n\n"
        f"=== PROBLEM ===\n{problem}{context_str}\n\n"
        f"=== YOUR RESPONSE ==="
    )

    response = model.generate_content(prompt)
    return response.text

# ─────────────────────────────────────────────
# Graph Visualization
# ─────────────────────────────────────────────
def build_agent_graph(completed: list[str]) -> str:
    """Build a directed agent workflow graph and return as base64 PNG."""
    G = nx.DiGraph()
    nodes = ["CEO", "Research", "Coder", "Critic", "Solution"]
    edges = [("CEO", "Research"), ("CEO", "Coder"), ("Research", "Coder"), ("Coder", "Critic"), ("Critic", "Solution")]
    
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(*edge)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.patch.set_facecolor('#0F0F1A')
    ax.set_facecolor('#0F0F1A')

    pos = {
        "CEO": (0.1, 0.5),
        "Research": (0.35, 0.75),
        "Coder": (0.6, 0.5),
        "Critic": (0.8, 0.5),
        "Solution": (1.0, 0.5),
    }

    node_colors_map = {**{k: AGENTS[k]["color"] for k in AGENTS}, "Solution": "#FFFFFF"}
    node_icons = {**{k: AGENTS[k]["icon"] for k in AGENTS}, "Solution": "✅"}

    for node in nodes:
        x, y = pos[node]
        color = node_colors_map.get(node, "#888")
        alpha = 1.0 if node in completed or node == "Solution" else 0.25
        size = 0.09

        circle = plt.Circle((x, y), size, color=color, alpha=alpha, zorder=3)
        ax.add_patch(circle)

        if node in completed:
            glow = plt.Circle((x, y), size + 0.015, color=color, alpha=0.15, zorder=2)
            ax.add_patch(glow)

        icon = node_icons.get(node, "")
        label = node if node != "Solution" else "Final\nSolution"
        ax.text(x, y + 0.005, icon, ha='center', va='center', fontsize=16, zorder=4)
        ax.text(x, y - size - 0.05, label, ha='center', va='top',
                fontsize=8, color='white', alpha=alpha,
                fontfamily='monospace', fontweight='bold')

    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        shrink = 0.09 / length
        x1s = x1 + dx * shrink
        y1s = y1 + dy * shrink
        x2s = x2 - dx * shrink
        y2s = y2 - dy * shrink

        active = u in completed
        ax.annotate("",
            xy=(x2s, y2s), xytext=(x1s, y1s),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#6C63FF" if active else "#333355",
                lw=2 if active else 1,
                mutation_scale=15,
            ),
            zorder=1,
        )

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0.1, 1.0)
    ax.axis('off')
    ax.set_title("Agent Collaboration Graph", color='white', fontsize=11,
                 fontfamily='monospace', pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0F0F1A', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# Main Solver Pipeline
# ─────────────────────────────────────────────
def solve_problem(problem: str, api_key: str, progress=gr.Progress()):
    if not api_key.strip():
        yield (
            "⚠️ Please enter your Gemini API key.", "", "", "", "", None,
            "❌ No API key provided.", None
        )
        return

    if not problem.strip():
        yield (
            "⚠️ Please enter a problem to solve.", "", "", "", "", None,
            "❌ No problem provided.", None
        )
        return

    try:
        model = get_model(api_key.strip())
    except Exception as e:
        yield (f"❌ Failed to initialize Gemini: {e}", "", "", "", "", None, "Error", None)
        return

    results = {}
    agent_order = ["CEO", "Research", "Coder", "Critic"]
    completed = []

    status_log = []

    for i, agent_name in enumerate(agent_order):
        progress((i + 0.1) / len(agent_order), desc=f"Running {agent_name} Agent...")
        agent = AGENTS[agent_name]
        status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent['icon']} {agent_name} Agent working...")

        status_str = "\n".join(status_log)
        graph_buf = build_agent_graph(completed)

        yield (
            results.get("CEO", ""),
            results.get("Research", ""),
            results.get("Coder", ""),
            results.get("Critic", ""),
            status_str,
            graph_buf,
            f"⏳ Running {agent_name} Agent ({i+1}/{len(agent_order)})...",
            None,
        )

        try:
            output = run_agent(agent_name, problem, results, model)
            results[agent_name] = output
            completed.append(agent_name)
            status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {agent_name} Agent complete.")
        except Exception as e:
            results[agent_name] = f"❌ Error: {e}"
            status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {agent_name} Agent failed: {e}")

        time.sleep(0.3)

    # Final synthesis
    progress(0.95, desc="Synthesizing final answer...")
    status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Synthesizing final answer...")
    graph_buf = build_agent_graph(completed)

    final_prompt = (
        f"You are a synthesis engine. Given the following multi-agent analysis of the problem:\n\n"
        f"PROBLEM: {problem}\n\n"
        f"CEO PLAN:\n{results.get('CEO','')}\n\n"
        f"RESEARCH:\n{results.get('Research','')}\n\n"
        f"TECHNICAL SOLUTION:\n{results.get('Coder','')}\n\n"
        f"CRITIC REVIEW:\n{results.get('Critic','')}\n\n"
        f"Write a clean, final, polished answer to the original problem. "
        f"This should be the definitive response that a user would want to read. "
        f"Format it clearly with sections if needed."
    )

    try:
        final_resp = model.generate_content(final_prompt)
        final_answer = final_resp.text
        status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Done! All agents complete.")
    except Exception as e:
        final_answer = f"❌ Final synthesis failed: {e}"

    graph_buf = build_agent_graph(completed + ["Solution"])

    yield (
        results.get("CEO", ""),
        results.get("Research", ""),
        results.get("Coder", ""),
        results.get("Critic", ""),
        "\n".join(status_log),
        graph_buf,
        "✅ All agents complete!",
        final_answer,
    )

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg: #080812;
    --surface: #0F0F1E;
    --surface2: #161628;
    --border: #2A2A4A;
    --accent: #6C63FF;
    --accent2: #00C9A7;
    --accent3: #FF6B6B;
    --accent4: #FFD93D;
    --text: #E8E8F0;
    --muted: #6B6B8A;
    --radius: 12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container { max-width: 1300px !important; margin: 0 auto !important; }

/* Header */
.header-box {
    background: linear-gradient(135deg, #0F0F1E 0%, #1A1A35 50%, #0F0F1E 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(108,99,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(0,201,167,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.app-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6C63FF, #00C9A7, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0 !important;
}
.app-subtitle {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    font-family: 'Space Mono', monospace !important;
    margin: 0 !important;
}

/* Agent badges */
.agent-badges {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 20px;
}
.agent-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: 1px solid;
}

/* Inputs */
label { color: var(--text) !important; font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important; }

textarea, input[type="text"], input[type="password"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
    outline: none !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, var(--accent), #9B93FF) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.05em !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.4) !important;
}

button.secondary {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Tabs */
.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Agent output panels */
.agent-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.block { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* Status log */
.status-log textarea {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--accent2) !important;
    background: #050510 !important;
    border-color: #1A2A1A !important;
}

/* Status label */
.status-badge {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    text-align: center !important;
}

/* Final answer */
.final-answer textarea {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    min-height: 200px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""

AGENT_ORDER = ["CEO", "Research", "Coder", "Critic"]

with gr.Blocks(css=CSS, title="Multi-Agent Problem Solver") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="header-box">
        <h1 class="app-title">⚡ Multi-Agent Problem Solver</h1>
        <p class="app-subtitle">// Powered by Google Gemini 2.5 Flash · Collaborative AI Reasoning</p>
        <div class="agent-badges">
            <span class="agent-badge" style="color:#6C63FF; border-color:#6C63FF33; background:rgba(108,99,255,0.08)">👔 CEO · Strategist</span>
            <span class="agent-badge" style="color:#00C9A7; border-color:#00C9A733; background:rgba(0,201,167,0.08)">🔬 Research · Analyst</span>
            <span class="agent-badge" style="color:#FF6B6B; border-color:#FF6B6B33; background:rgba(255,107,107,0.08)">💻 Coder · Implementer</span>
            <span class="agent-badge" style="color:#FFD93D; border-color:#FFD93D33; background:rgba(255,217,61,0.08)">🎯 Critic · Reviewer</span>
        </div>
    </div>
    """)

    # ── Input Row ──
    with gr.Row():
        with gr.Column(scale=3):
            problem_input = gr.Textbox(
                label="🧩 PROBLEM STATEMENT",
                placeholder="Describe the problem you want to solve... e.g. 'Design a scalable microservices architecture for a real-time chat application'",
                lines=4,
                elem_classes=["problem-input"],
            )
        with gr.Column(scale=2):
            api_key_input = gr.Textbox(
                label="🔑 GEMINI API KEY",
                placeholder="AIza...",
                type="password",
                lines=1,
            )
            with gr.Row():
                solve_btn = gr.Button("⚡ SOLVE", variant="primary", scale=3)
                clear_btn = gr.Button("↺ Clear", variant="secondary", scale=1)

    # ── Status ──
    status_label = gr.Textbox(
        value="Ready. Enter a problem and API key, then click SOLVE.",
        label="STATUS",
        interactive=False,
        elem_classes=["status-badge"],
        lines=1,
    )

    # ── Graph ──
    graph_image = gr.Image(
        label="🕸 AGENT COLLABORATION GRAPH",
        type="filepath",
        interactive=False,
        height=280,
    )

    # ── Agent Outputs ──
    gr.HTML("<hr style='border-color:#2A2A4A; margin:8px 0;'>")

    with gr.Tabs():
        agent_outputs = {}
        tab_labels = {
            "CEO": "👔 CEO Agent",
            "Research": "🔬 Research Agent",
            "Coder": "💻 Coder Agent",
            "Critic": "🎯 Critic Agent",
        }
        for agent_name in AGENT_ORDER:
            with gr.Tab(tab_labels[agent_name]):
                gr.HTML(f"""
                <div style="padding:10px 0 6px 0; font-family:'Space Mono',monospace; font-size:0.75rem; 
                     color:{AGENTS[agent_name]['color']}; opacity:0.8;">
                  {AGENTS[agent_name]['icon']} {AGENTS[agent_name]['role'].upper()}
                </div>
                """)
                agent_outputs[agent_name] = gr.Textbox(
                    label="",
                    lines=14,
                    interactive=False,
                    placeholder=f"Waiting for {agent_name} Agent to run...",
                    elem_classes=["agent-panel"],
                )

        with gr.Tab("📋 Status Log"):
            status_log = gr.Textbox(
                label="",
                lines=14,
                interactive=False,
                placeholder="Agent activity log will appear here...",
                elem_classes=["status-log"],
            )

    # ── Final Answer ──
    gr.HTML("<hr style='border-color:#2A2A4A; margin:8px 0;'>")
    gr.HTML("""
    <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#6C63FF; 
         margin-bottom:8px; letter-spacing:0.05em;">
      ✅ FINAL SYNTHESIZED ANSWER
    </div>
    """)
    final_answer = gr.Textbox(
        label="",
        lines=12,
        interactive=False,
        placeholder="The final synthesized answer will appear here after all agents complete...",
        elem_classes=["final-answer"],
    )

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; margin-top:24px; padding:16px; 
         font-family:'Space Mono',monospace; font-size:0.7rem; color:#3A3A5A;">
        Multi-Agent Problem Solver · Gemini 2.5 Flash · Built with Gradio · Deploy on Hugging Face Spaces
    </div>
    """)

    # ── Event Handlers ──
    def clear_all():
        return ("", "", "", "", "", "", "Ready. Enter a problem and API key, then click SOLVE.", None, "")

    clear_btn.click(
        fn=clear_all,
        outputs=[
            problem_input,
            agent_outputs["CEO"],
            agent_outputs["Research"],
            agent_outputs["Coder"],
            agent_outputs["Critic"],
            status_log,
            status_label,
            graph_image,
            final_answer,
        ],
    )

    def save_graph_and_return(buf):
        """Save graph buffer to temp file and return path."""
        if buf is None:
            return None
        path = "/tmp/agent_graph.png"
        with open(path, "wb") as f:
            f.write(buf.read())
        return path

    def solve_wrapper(problem, api_key, progress=gr.Progress()):
        for outputs in solve_problem(problem, api_key, progress):
            ceo, research, coder, critic, log, graph_buf, status, final = outputs
            graph_path = save_graph_and_return(graph_buf) if graph_buf else None
            yield ceo, research, coder, critic, log, graph_path, status, final

    solve_btn.click(
        fn=solve_wrapper,
        inputs=[problem_input, api_key_input],
        outputs=[
            agent_outputs["CEO"],
            agent_outputs["Research"],
            agent_outputs["Coder"],
            agent_outputs["Critic"],
            status_log,
            graph_image,
            status_label,
            final_answer,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
