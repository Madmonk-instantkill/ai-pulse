"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import PaperSection


@CrewBase
class WriterCrew:
    """Writer stage of Editorial Crew: turns one paper's digest into a
    beginner-friendly section plus that paper's claim ledger.

    Invoked once per paper via crew().kickoff(inputs={"arxiv_id": ...,
    "digest": ...}), where digest is the PaperDigest serialized as JSON.
    """

    agents_config = "config/writer_agents.yaml"
    tasks_config = "config/writer_tasks.yaml"

    @agent
    def paper_writer_agent(self) -> Agent:
        """The Paper Writer Agent: pure LLM reasoning, no tools --
        explaining a paper for beginners requires judgment."""
        return Agent(
            config=self.agents_config["paper_writer_agent"],
            verbose=True,
        )

    @task
    def write_paper_section_task(self) -> Task:
        """Write the section and claim ledger for whichever paper's
        digest is passed in via kickoff(inputs={...})."""
        return Task(
            config=self.tasks_config["write_paper_section_task"],
            output_pydantic=PaperSection,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
