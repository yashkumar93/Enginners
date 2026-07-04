"""Knowledge Sources for CrewAI agents.

Defines the text file knowledge sources that wrap the project scope,
requirements, and other documentation for Retrieval-Augmented Generation (RAG).
"""

from pathlib import Path
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Resolve absolute paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG Knowledge Source for Project Requirements and Scope
requirements_source = TextFileKnowledgeSource(
    file_paths=[
        _PROJECT_ROOT / "project_scope.txt",
        _PROJECT_ROOT / "requirements/non_functional_requirements.txt",
    ]
)
