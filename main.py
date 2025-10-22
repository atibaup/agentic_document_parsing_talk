from argparse import ArgumentParser
from dotenv import load_dotenv
import asyncio
import json
from typing import Tuple, Generic, TypeVar, Annotated
from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from parser import SimpleAgentParser, OneShotParser

import os

load_dotenv()

T = TypeVar("T")

class CitedFact(BaseModel, Generic[T]):
    passage: str = Field(description="The passage of the document where the fact is obtained, verbatim exactly as it is in the document, including punctuation and whitespace.")
    fact: T = Field(description="The cited fact")

class DocumentModel(BaseModel):
    title: Annotated[CitedFact[str], Field(description="The title of the document")]
    author: Annotated[CitedFact[str], Field(description="The author of the document")]
    date: Annotated[CitedFact[date], Field(description="The date of the document in %Y-%m-%d format")]


def validate_citations(document: str, data: DocumentModel) -> Tuple[bool, str]:
    """Validate the citations in the document."""
    clean_document = document.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ")
    if data.title.passage not in clean_document:
        return False, f"`Title` passage not found in document, passage={data.title.passage}"
    if data.author.passage not in clean_document:
        return False, f"`Author` passage not found in document, passage={data.author.passage}"
    if data.date.passage not in clean_document:
        return False, f"`Date` passage not found in document, passage={data.date.passage}"
    return True, "Citations validated successfully"
    

async def main():
    args = ArgumentParser()
    args.add_argument("--file", type=str, required=True)
    args = args.parse_args()

    model_name = "gemini/gemini-2.5-flash"

    with open(args.file, "r") as file:
        document = file.read()

    print("One Shot Parser (DocumentModel):\n===========================================")
    parser = OneShotParser(DocumentModel, model_name)
    try:
        result = await parser.parse(document)

        print(result.model_dump_json(indent=4))
    except Exception as e:
        print(f"Error parsing document: {e}")

    print("Simple Agent Parser (DocumentModel):\n===========================================")

    parser = SimpleAgentParser(DocumentModel, model_name, [validate_citations])
    result = await parser.parse(document)

    print(result.model_dump_json(indent=4))

if __name__ == "__main__":
    asyncio.run(main())