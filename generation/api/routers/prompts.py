import os
import httpx
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])

@router.get("/{prompt_identifier}/versions_full", response_model=List[Dict])
async def list_prompt_versions_full(prompt_identifier: str, limit: int = 50, cursor: str = None):
    """
    Fetches prompt version list via Phoenix REST API for a given prompt identifier.
    """
    base_url = os.getenv("PHOENIX_BASE_URL", "http://localhost:6006")
    api_path = f"{base_url}/v1/prompts/{prompt_identifier}/versions"
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor

    async with httpx.AsyncClient() as client:
        resp = await client.get(api_path, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Error: {resp.text}")
    body = resp.json()
    return body.get("data", [])

