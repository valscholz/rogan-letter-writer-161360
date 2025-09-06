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
from agents import Runner, Session
import os
import threading
import psutil
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session for simple chat continuity
# For production, you'd want per-user session management
try:
    chat_session = Session()
except Exception as e:
    logger.warning(f"Could not create session: {e}")
    chat_session = None

# Thread pool for controlled concurrency
max_workers = min(4, os.cpu_count() or 1)
thread_pool = ThreadPoolExecutor(max_workers=max_workers)
logger.info(f"Initialized thread pool with {max_workers} workers")

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
    try:
        memory_info = psutil.virtual_memory()
        thread_count = threading.active_count()
        
        # Determine health status
        status = "healthy"
        if memory_info.percent > 85:
            status = "warning_high_memory"
        if thread_count > 50:
            status = "warning_high_threads" 
        if memory_info.percent > 95 or thread_count > 100:
            status = "unhealthy"
            
        return {
            "status": status,
            "service": "style-safe-letter-writer-orchestrator",
            "resources": {
                "memory_percent": memory_info.percent,
                "memory_available_mb": memory_info.available // 1024 // 1024,
                "thread_count": thread_count,
                "cpu_count": os.cpu_count()
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "service": "style-safe-letter-writer-orchestrator", "error": str(e)}

@app.post("/generate", response_model=AgentResponse)
async def generate(request: AgentRequest):
    try:
        logger.info(f"Processing request: {request.prompt[:100]}...")
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Run the agent
        result = await Runner().run(main_agent, request.prompt)
        return AgentResponse(
            response=result.final_output,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Log system resource info for debugging  
        try:
            memory_info = psutil.virtual_memory()
            thread_count = threading.active_count()
            logger.error(f"System state - Memory: {memory_info.percent}% used, Threads: {thread_count}")
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    """Chat endpoint for conversational agents"""
    try:
        logger.info(f"Processing chat request: {request.prompt[:100]}...")
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Try with session first (for conversational agents)
        try:
            if chat_session:
                result = await Runner().run(main_agent, request.prompt, session=chat_session)
            else:
                result = await Runner().run(main_agent, request.prompt)
        except Exception as session_error:
            logger.warning(f"Session-based run failed: {session_error}, trying without session")
            # Fallback to no session (for simple agents)
            result = await Runner().run(main_agent, request.prompt)
            
        return AgentResponse(
            response=result.final_output,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        # Log system resource info for debugging
        try:
            memory_info = psutil.virtual_memory()
            thread_count = threading.active_count()
            logger.error(f"System state - Memory: {memory_info.percent}% used, Threads: {thread_count}")
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    workers = min(2, os.cpu_count() or 1)  # Limit workers to reduce thread usage
    logger.info(f"Starting server on port {port} with {workers} workers")
    
    # Log startup system info
    try:
        memory_info = psutil.virtual_memory() 
        logger.info(f"System startup - Memory: {memory_info.total // 1024 // 1024}MB total, {memory_info.available // 1024 // 1024}MB available")
        logger.info(f"CPUs available: {os.cpu_count()}")
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Use single worker to reduce memory/thread usage
        log_level="info",
        access_log=False,  # Reduce logging overhead
        limit_concurrency=16,  # Limit concurrent connections
        limit_max_requests=1000,  # Restart worker after 1000 requests to free memory
    )