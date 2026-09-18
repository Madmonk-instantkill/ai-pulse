"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import PaperDigest


@CrewBase
class ReduceCrew:
    """Reduce stage of Content Crew: merges one paper's chunk summaries
    into a single document-level digest.

    Invoked once per paper via crew().kickoff(inputs={"arxiv_id": ...,
    "chunk_summaries": ...}), where chunk_summaries comes from
    reduce.format_summaries_for_reduce(). Separate from MapCrew because
    Map runs once per chunk while Reduce runs once per paper.
    """

    agents_config = "config/reduce_agents.yaml"
    tasks_config = "config/reduce_tasks.yaml"

    @agent
    def reduce_digester_agent(self) -> Agent:
        """The Reduce Digester: pure LLM reasoning, no tools -- merging
        partial summaries and resolving contradictions requires
        judgment, not a deterministic lookup."""
        return Agent(
            config=self.agents_config["reduce_digester_agent"],
            verbose=True,
        )

    @task
    def reduce_digest_task(self) -> Task:
        """Merge whichever paper's summaries are passed in via
        kickoff(inputs={...}) into one PaperDigest."""
        return Task(
            config=self.tasks_config["reduce_digest_task"],
            output_pydantic=PaperDigest,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
