from argparse import ArgumentParser
from xxlimited import Str
from dotenv import load_dotenv
import asyncio
import difflib
from typing import Tuple, Generic, TypeVar, Annotated
from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from parser import SimpleAgentParser, OneShotParser
import litellm

import os

load_dotenv()

T = TypeVar("T")


class CitedFact(BaseModel, Generic[T]):
    passage: str = Field(description="The passage of the document where the fact is obtained, verbatim exactly as it is in the document, including punctuation and whitespace.")
    fact: T = Field(description="The cited fact")

class Location(BaseModel):
    municipality: str = Field(description="The municipality where the document is located")
    province: str = Field(description="The province where the document is located")

class DocumentModel(BaseModel):
    title: Annotated[CitedFact[str], Field(description="The title of the document")]
    author: Annotated[CitedFact[str], Field(description="The author of the document")]
    date: Annotated[CitedFact[date], Field(description="The date of the document in %Y-%m-%d format")]
    locations: Annotated[Location, Field(description="The locations cited in the document")]


async def validate_passages(document: str, data: DocumentModel) -> Tuple[bool, str]:
    """Validate the citations in the document using LLM-based assessment."""

    passages_to_validate = [
        ("Title", data.title.passage),
        ("Author", data.author.passage),
        ("Date", data.date.passage),
    ]

    for field_name, passage in passages_to_validate:
        prompt = f"""You are a document validation assistant. Your task is to determine if a given passage exists in a document.

Document:
{document}

Passage to find:
{passage}

Does the passage appear in the document?
Answer with only "YES" or "NO", followed by a brief explanation if NO."""

        response = await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip().upper()

        if not answer.startswith("YES"):
            return False, f"`{field_name}` passage not found in document according to LLM validation. Passage='{passage}'. LLM response: {answer}"

    return True, "All passages validated successfully by LLM"


async def validate_author(document: str, data: DocumentModel) -> Tuple[bool, str]:
    """Validate the author of the document."""
    prompt = f"""You are a document validation assistant. Your task is to determine if the extracted author is the correct one.

Document:
{document}

Author to find:
{data.author.fact}

- Does the author appear in the document?
- Is the author the correct one?
- Does the author appear to be a person's name?
Answer with only "YES" if all the above are true, otherwise answer with "NO" and explain why."""

    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().upper()

    if not answer.startswith("YES"):
        return False, f"`author` not found in document according to LLM validation. author='{data.author.fact}'. LLM response: {answer}"

    return True, "author validated successfully by LLM"


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
    parser = SimpleAgentParser(DocumentModel, model_name, [validate_passages, validate_author])
    result = await parser.parse(document)
    print(result.model_dump_json(indent=4))

if __name__ == "__main__":
    asyncio.run(main())