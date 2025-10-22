from litellm import acompletion

from .base import Parser, T


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
