from pydantic import BaseModel
from typing import TypeVar
from abc import ABC, abstractmethod
from pydantic_ai import Agent
from pydantic_ai.models import Model
from litellm import acompletion

import os
import json

T = TypeVar("T", bound=BaseModel)


class Parser[T](ABC):
    def __init__(self, output_type: type[T]):
        """Initialize the Parser.
        
        Args:
            output_type: The Pydantic model to parse the document into.
        """
        self.output_type = output_type

    @abstractmethod
    async def parse(self, document: str) -> T:
        """Parse a document into a Pydantic model.
        
        Args:
            document: The document to parse.

        Returns:
            The parsed document as a Pydantic model.
        """
        pass


class SimpleAgentParser(Parser[T]):
    """
    A simple agent parser that uses a simple PydanticAI agent to parse a document into a Pydantic model.
    """
    def __init__(self, output_type: type[T], model: Model):
        """Initialize the SimpleAgentParser.
        
        Args:
            output_type: The Pydantic model to parse the document into.
            model: The AI LLM model to use for the agent.
        """
        super().__init__(output_type)
        _system_prompt = f"""
            You are a document parsing assistant. 
            You will be given a document and you will need to parse it into a Pydantic model.
        """
        self.agent = Agent(
            model=model,
            output_type=self.output_type,
            system_prompt=_system_prompt,
        )

    async def parse(self, document: str) -> T:
        """Parse a document into a Pydantic model.
        
        Args:
            document: The document to parse.
        """
        nodes = []
        async with self.agent.iter(document) as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)

        return nodes[-1].data.output


class OneShotParser(Parser[T]):
    """
    A one-shot parser that uses LiteLLM with structured outputs to parse a document into a Pydantic model.
    Makes a single LLM call with the Pydantic model schema for structured output.
    """
    def __init__(self, output_type: type[T], model: str):
        """Initialize the OneShotParser.

        Args:
            output_type: The Pydantic model to parse the document into.
            model: The LLM model to use (LiteLLM format, e.g., "gemini/gemini-1.5-flash").
        """
        super().__init__(output_type)
        self.model = model
        self.system_prompt = """You are a document parsing assistant.
You will be given a document and you will need to parse it into structured data.
Extract all relevant information accurately according to the provided schema."""

    async def parse(self, document: str) -> T:
        """Parse a document into a Pydantic model using a single LLM call.

        Args:
            document: The document to parse.

        Returns:
            The parsed document as a Pydantic model.
        """
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": document}
            ],
            response_format=self.output_type
        )

        return self.output_type.model_validate_json(
            response.choices[0].message.content
        )