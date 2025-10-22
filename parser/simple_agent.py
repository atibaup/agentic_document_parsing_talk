from typing import Callable, Tuple, Union
import asyncio
import inspect

from litellm import acompletion
from pydantic import ValidationError as PydanticValidationError

from .base import Parser, T


class SimpleAgentParser(Parser[T]):
    """
    A simple agent parser that uses a simple reflection agentic pattern to parse a document.

    """
    def __init__(self, output_type: type[T], model: str, validation_functions: list[Callable[[str, T], Tuple[bool, str]]] = [], max_attempts: int = 5):
        """Initialize the SimpleAgentParser.

        Args:
            output_type: The Pydantic model to parse the document into.
            model: The AI LLM model to use for the agent.
            validation_functions: The list of semantic validation functions to use for the agent. 
                Validation functions must take the document and the parsed data and return a 
                tuple with a boolean indicating if the validation was successful and a string
                with the error message if the validation was not successful.
        """
        super().__init__(output_type)

        def is_valid_schema(document: str, parsed: T) -> Tuple[bool, str]:
            """Validate the schema of the parsed data."""
            try:
                output_type.model_validate(parsed)
            except PydanticValidationError as e:
                serialized_errors = ",".join(
                    [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                )
                error_msg = f"The returned tool call doesn't meet the semantic validation requirements, errors: {serialized_errors}"
                return False, error_msg
            return True, "Validation successful"
            
        self.validation_functions = [is_valid_schema] + validation_functions
        self.output_type = output_type
        self.model = model
        self.max_attempts = max_attempts
        self._system_prompt = f"""
You are a document parsing agent.
You will be given a document and you will need to parse it into the provided schema.

Instructions:
- Make sure to parse all the information required by the schema.
- Retrieve the information from the document exactly as it is in the document.
        """
        if len(self.validation_functions) > 0:
            self._system_prompt += "\n- Before you return the parsed data you MUST validate it using the provided `validate_` tool."

    
    async def _validate_data(self, document: str, data: T) -> Tuple[bool, str]:
        """Validate the parsed data against the validation functions."""
        for validation_function in self.validation_functions:
            # Check if the validation function is async
            if inspect.iscoroutinefunction(validation_function):
                success, error = await validation_function(document, data)
            else:
                success, error = validation_function(document, data)
            if not success:
                return False, error
        return True, "Validation successful"

    @staticmethod
    def _add_feedback(messages: list[dict], attempt: dict, feedback: str):
        """Update the messages with the feedback."""
        messages.append({
            "role": "assistant",
            "content": attempt
        })
        messages.append({
            "role": "tool",
            "name": "validate_data",
            "content": feedback
        })

    async def _attempt_parse(self, messages: list[dict]) -> str:
        """Attempt to parse the document into a Pydantic model."""
        response = await acompletion(
            model=self.model,
            messages=messages,
            response_format=self.output_type
        )
        return response.choices[0].message.content

    async def parse(self, document: str) -> T:
        """Parse a document into a Pydantic model.

        Args:
            document: The document to parse.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": document}
        ]
        n_attempts = 0
        while n_attempts < self.max_attempts:
            print(f"Attempt {n_attempts + 1} of {self.max_attempts}")
            attempt = await self._attempt_parse(messages)

            # Schema validation
            try:
                parsed = self.output_type.model_validate_json(attempt)
            except PydanticValidationError as e:
                serialized_errors = ",".join(
                    [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                )
                schema_error = f"The parsed object doesn't meet the schema validation requirements, errors: {serialized_errors}"
                print(f"Schema error: {schema_error}")
                self._add_feedback(messages, attempt, schema_error)
                n_attempts += 1
                continue

            # Semantic validation
            success, semantic_error = await self._validate_data(document, parsed)
            if success:
                return parsed
            else:
                print(f"Semantic error: {semantic_error}")
                self._add_feedback(messages, attempt, semantic_error)

            n_attempts += 1

        raise ValueError(f"Failed to parse the document after {self.max_attempts} attempts")
