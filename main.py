from argparse import ArgumentParser
from dotenv import load_dotenv
import asyncio
import json
from pydantic import BaseModel, Field, field_validator
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel

from parser import SimpleAgentParser, OneShotParser

import os

load_dotenv()

async def main():
    args = ArgumentParser()
    args.add_argument("--file", type=str, required=True)
    args = args.parse_args()

    model_name = "gemini-2.5-flash"

    provider = GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))
    model = GoogleModel(model_name, provider=provider)

    with open(args.file, "r") as file:
        document = file.read()

    class DocumentModel(BaseModel):
        title: str = Field(description="The title of the document")
        summary: str = Field(description="The summary of the document")
        author: str = Field(description="The author of the document")
        date: str = Field(description="The date of the document")

    class CitatedText(BaseModel):
        start: int = Field(description="The start line of the citated text in the document")
        end: int = Field(description="The end line of the citated text in the document")
        text: str = Field(description="The citated text")

    class DocumentModelWithCitation(BaseModel):
        title: CitatedText = Field(description="The title of the document")
        summary: CitatedText = Field(description="The summary of the document")
        author: CitatedText = Field(description="The author of the document")
        date: CitatedText = Field(description="The date of the document")

    class DocumentModelWithCitationsVerified(DocumentModelWithCitation):
        @field_validator("title", "summary", "author", "date")
        def verify_citations(cls, v):
            return v

    parser = SimpleAgentParser(DocumentModel, model)
    result = await parser.parse(document)

    print("Simple Agent Parser (DocumentModel):\n===========================================")
    print(result.model_dump_json(indent=4))

    parser = OneShotParser(DocumentModel, f"gemini/{model_name}")
    result = await parser.parse(document)

    print("One Shot Parser (DocumentModel):\n===========================================")
    print(result.model_dump_json(indent=4))

    parser = SimpleAgentParser(DocumentModelWithCitation, model)
    result = await parser.parse(document)

    print("Simple Agent Parser (DocumentModelWithCitation):\n===========================================")
    print(result.model_dump_json(indent=4))

if __name__ == "__main__":
    asyncio.run(main())