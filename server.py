#!/usr/bin/env python3
"""Web server template for Caminu-generated agents"""

import asyncio
import os
import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from agent import main_agent
from agents import Runner
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Style-Safe Letter Writer Orchestrator",
    description="Users want to quickly generate high-quality, personalized letters based on a topic, optionally in the recognizable style of a public figure (e.g., Joe Rogan), while maintaining ethical safeguards (no deceptive impersonation) and supporting rapid iteration, structured outputs, and export.",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    prompt: str = "Please demonstrate your functionality."

class AgentResponse(BaseModel):
    response: str
    status: str = "success"

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the UI interface"""
    try:
        # Try to serve the index.html file
        html_path = Path(__file__).parent / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        else:
            # Fallback: Basic API info if no UI file found
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head><title>Style-Safe Letter Writer Orchestrator</title></head>
            <body>
                <h1>Style-Safe Letter Writer Orchestrator</h1>
                <p>Users want to quickly generate high-quality, personalized letters based on a topic, optionally in the recognizable style of a public figure (e.g., Joe Rogan), while maintaining ethical safeguards (no deceptive impersonation) and supporting rapid iteration, structured outputs, and export.</p>
                <p>API Endpoints:</p>
                <ul>
                    <li>POST /generate - Generate a response from a prompt</li>
                    <li>GET /health - Health check</li>
                </ul>
            </body>
            </html>
            """, status_code=200)
    except Exception as e:
        logger.error(f"Error serving UI: {e}")
        return HTMLResponse(content="<h1>UI Error</h1><p>Could not load interface</p>", status_code=500)

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Welcome to Style-Safe Letter Writer Orchestrator",
        "endpoints": {
            "GET /": "User interface",
            "POST /generate": "Generate a response from a prompt",
            "POST /chat": "Chat endpoint for conversational agents",
            "GET /health": "Health check",
            "GET /api": "This API information"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "style-safe-letter-writer-orchestrator"}

@app.post("/generate", response_model=AgentResponse)
async def generate(request: AgentRequest):
    try:
        logger.info(f"Processing request: {request.prompt[:100]}...")
        result = await Runner().run(main_agent, request.prompt)
        return AgentResponse(
            response=result.final_output,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    """Chat endpoint for conversational agents"""
    try:
        logger.info(f"Processing chat request: {request.prompt[:100]}...")
        result = await Runner().run(main_agent, request.prompt)
        return AgentResponse(
            response=result.final_output,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )