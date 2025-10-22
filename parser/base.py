from pydantic import BaseModel
from typing import TypeVar
from abc import ABC, abstractmethod

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
