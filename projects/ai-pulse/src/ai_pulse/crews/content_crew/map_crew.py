"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import ChunkSummary


@CrewBase
class MapCrew:
    """Map stage of Content Crew: summarizes one chunk of a paper.

    map_summarize_task is invoked once per chunk via repeated
    crew().kickoff(inputs={...}) calls, not multiple @task methods --
    the number of chunks varies per paper, so it can't be a fixed set of
    named tasks the way Discovery Crew's two tasks were. That looping
    lives in ai_pulse_flow.py, not here.

    Map and Reduce are separate crew classes (with separate YAML files)
    because they run at different cadences -- N times per paper vs once
    per paper -- and a crew runs all of its tasks on every kickoff.
    """

    agents_config = "config/map_agents.yaml"
    tasks_config = "config/map_tasks.yaml"

    @agent
    def map_summarizer_agent(self) -> Agent:
        """The Map Summarizer: pure LLM reasoning, no tools -- reading a
        chunk's text and extracting structured content requires
        judgment, not a deterministic lookup."""
        return Agent(
            config=self.agents_config["map_summarizer_agent"],
            verbose=True,
        )

    @task
    def map_summarize_task(self) -> Task:
        """Summarize whichever chunk's text/page numbers are passed in
        via kickoff(inputs={...}) -- description/expected_output contain
        {chunk_text}/{start_page}/{end_page}/{arxiv_id} placeholders
        filled in fresh on each call."""
        return Task(
            config=self.tasks_config["map_summarize_task"],
            output_pydantic=ChunkSummary,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
