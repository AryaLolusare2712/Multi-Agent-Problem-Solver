# ⚡ Multi-Agent Problem Solver

A Generative AI-powered multi-agent system that collaboratively solves complex problems using **4 specialized agents** powered by **Google Gemini 2.5 Flash**.

## 🧠 How It Works

The system pipelines your problem through 4 specialized agents in sequence:

| Agent | Role | Responsibility |
|-------|------|---------------|
| 👔 **CEO Agent** | Strategic Planner | Breaks the problem into a high-level plan with clear action items |
| 🔬 **Research Agent** | Information Analyst | Gathers relevant knowledge, identifies trade-offs and edge cases |
| 💻 **Coder Agent** | Technical Implementer | Produces concrete solutions, code, algorithms, or procedures |
| 🎯 **Critic Agent** | Quality Reviewer | Reviews all outputs, identifies gaps, and synthesizes improvements |

After all agents complete, a **Final Synthesis** pass produces the definitive polished answer.

## 🛠 Tech Stack

- **Python** — Core backend
- **Gradio** — Interactive web UI
- **Google Gemini 2.5 Flash** — LLM powering all agents
- **NetworkX + Matplotlib** — Agent collaboration graph visualization
- **Hugging Face Spaces** — Deployment platform

## 🚀 Usage

1. Enter your [Google Gemini API Key](https://aistudio.google.com/app/apikey)
2. Type your problem in the text box
3. Click **⚡ SOLVE**
4. Watch agents work in real-time across the tabs
5. Read the final synthesized answer at the bottom

## 💡 Example Problems

- *"Design a scalable microservices architecture for a real-time chat app"*
- *"How do I implement a recommendation engine with limited user data?"*
- *"Explain the trade-offs between SQL and NoSQL for an e-commerce platform"*
- *"Write a Python script to scrape and analyze stock prices"*
- *"How can a small startup compete with large tech companies in AI?"*

## 🔑 API Key

Get your free Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey).  
Your key is never stored — it's used only within your session.

## 📄 License

MIT License
