"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import GlossaryResult


@CrewBase
class GlossaryCrew:
    """Glossary stage of Content Crew: picks 3-5 terms from one paper's
    digest and explains them for a beginner.

    Invoked once per paper via crew().kickoff(inputs={"arxiv_id": ...,
    "digest": ...}), where digest is the PaperDigest serialized as JSON.
    The entries feed straight into the report; there is no cross-run
    glossary archive.
    """

    agents_config = "config/glossary_agents.yaml"
    tasks_config = "config/glossary_tasks.yaml"

    @agent
    def glossary_agent(self) -> Agent:
        """The Glossary Agent: pure LLM reasoning, no tools -- choosing
        which terms matter and rewording them for a beginner requires
        judgment."""
        return Agent(
            config=self.agents_config["glossary_agent"],
            verbose=True,
        )

    @task
    def glossary_task(self) -> Task:
        """Build glossary entries from whichever paper's digest is
        passed in via kickoff(inputs={...})."""
        return Task(
            config=self.tasks_config["glossary_task"],
            output_pydantic=GlossaryResult,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
