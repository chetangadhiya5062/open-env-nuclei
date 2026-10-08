"""Gradio demo: pick a task, seed and agent, then watch the episode step by step.

Mounted at ``/demo`` by the FastAPI app (same process / same Hugging Face Space). The demo runs episodes
in-process, so it never touches other users' sessions.
"""

import json
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from .server.datagen import TASKS, generate

STEP_COLUMNS = ["step", "action", "reward", "quality", "result"]


def available_agents() -> Dict[str, object]:
    """Agent name -> factory. The LLM agent is only offered when HF_TOKEN is configured."""
    from agents import DoNothingAgent, LLMAgent, RandomAgent, RuleBasedAgent

    agents = {"rule-based": RuleBasedAgent, "random": RandomAgent, "do-nothing": DoNothingAgent}
    if LLMAgent.available():
        agents["llm"] = LLMAgent
    return agents


def run_demo(
    task_id: str, seed: int, agent_name: str
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]]:
    """Yield ``(steps_table, before_table, after_table, summary_markdown)`` after every step."""
    from agents import LocalEnv

    agent = available_agents()[agent_name]()
    seed = int(seed)
    before, _ = generate(task_id, seed)
    env = LocalEnv()
    agent.reset(seed)
    obs = env.reset(task_id, seed)
    rows: List[list] = []
    summary = f"Start: quality **{obs.quality_score:.3f}**, {obs.row_count} rows, {obs.duplicate_count} duplicates."
    yield pd.DataFrame(rows, columns=STEP_COLUMNS), before, before, summary

    for _ in range(TASKS[task_id].max_steps + 5):
        if obs.done:
            break
        action = agent.act(obs)
        obs = env.step(action)
        result = obs.last_message if obs.last_action_ok else f"REJECTED: {obs.last_error}"
        rows.append(
            [
                obs.step_count,
                json.dumps(action.to_payload(), separators=(",", ":")),
                round(float(obs.reward or 0.0), 3),
                round(obs.quality_score, 3),
                result,
            ]
        )
        summary = (
            f"Step {obs.step_count}: quality **{obs.quality_score:.3f}**, total reward **{obs.total_reward:.2f}**."
        )
        yield pd.DataFrame(rows, columns=STEP_COLUMNS), before, env._env.df.copy(), summary

    b = obs.score_breakdown or {}
    summary = (
        f"**Done.** Grader score **{obs.final_score:.3f}** "
        f"(cells {b.get('cell_accuracy', 0):.3f}, rows {b.get('row_fidelity', 0):.3f}, "
        f"schema {b.get('schema_score', 0):.3f}) | total reward {obs.total_reward:.2f} | "
        f"{obs.step_count} steps | invalid actions {obs.invalid_action_count}"
    )
    yield pd.DataFrame(rows, columns=STEP_COLUMNS), before, env._env.df.copy(), summary
    env.close()


def build_demo():
    import gradio as gr

    agents = list(available_agents())
    with gr.Blocks(title="Data Cleaning Environment") as demo:
        gr.Markdown(
            "# Data Cleaning Environment\n"
            "Pick a task, a seed and an agent, then watch it clean a dirty table. "
            "The grader score compares the result with a hidden ground truth."
        )
        with gr.Row():
            task = gr.Dropdown(list(TASKS), value="hard-clean", label="Task")
            seed = gr.Number(value=0, precision=0, label="Seed")
            agent = gr.Dropdown(agents, value=agents[0], label="Agent")
            run = gr.Button("Run episode", variant="primary")
        summary = gr.Markdown("Press **Run episode**.")
        steps = gr.Dataframe(headers=STEP_COLUMNS, label="Steps (action, reward, quality, result)", wrap=True)
        with gr.Row():
            before = gr.Dataframe(label="Before (dirty table)", wrap=True)
            after = gr.Dataframe(label="After (agent's table)", wrap=True)
        run.click(run_demo, inputs=[task, seed, agent], outputs=[steps, before, after, summary])
    return demo
