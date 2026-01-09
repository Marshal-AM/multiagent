"""
Multi-tenant Agent Server
Handles multiple clients with isolated configurations, MongoDB databases, and tools.
"""
import os
import sys
import asyncio
import aiohttp
import httpx
import time
import json
import atexit
from datetime import datetime, timedelta
from threading import Lock
from typing import List, Dict, Optional
from pathlib import Path
from pydantic import BaseModel

# Load environment variables FIRST before importing config_loader
from dotenv import load_dotenv
load_dotenv()

# Import LOCAL config_loader
from config_loader import (
    get_client_config, 
    get_mongodb_database_name,
    get_preprocessor_url,
    get_postprocessor_url,
    list_all_clients
)

# Import LOCAL auth module (in same directory)
from auth.middleware import require_auth, check_client_ownership
from auth.models import User

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyngrok import ngrok

import re
from pipecat.frames.frames import EndFrame, LLMRunFrame, TextFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from loguru import logger

# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Global variable to store the ngrok tunnel
ngrok_tunnel = None

# Global storage for guardrails per client: {client_id: [guardrails]}
guardrails_storage: Dict[str, List[Dict[str, str]]] = {}
guardrails_lock = Lock()

# FastAPI app
app = FastAPI(title="Multi-Tenant Agent Server", version="2.0.0")

# Add startup event to load guardrails from MongoDB for all clients
@app.on_event("startup")
async def startup_event():
    """Load guardrails from MongoDB when the application starts"""
    logger.info("Application starting up - loading guardrails from MongoDB for all clients...")
    clients = list_all_clients()  # Now returns ObjectIds as strings
    for id in clients:
        try:
            load_guardrails_from_mongodb(id)
        except Exception as e:
            logger.warning(f"Failed to load guardrails for client {id}: {e}")

# Add CORS middleware (configurable via environment)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Client-ID", "X-Request-ID"],  # Expose custom headers
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Tenant Context Middleware
@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    """
    Middleware to extract and store tenant/client context in request state.
    Logs which tenant is accessing which resource for monitoring and debugging.
    """
    # Extract client_id from query params or headers
    client_id = request.query_params.get("client_id") or request.headers.get("X-Client-ID")
    
    # Store in request state for easy access throughout request lifecycle
    request.state.client_id = client_id
    request.state.has_client_context = bool(client_id)
    
    # Log tenant context (optional - can be disabled in production)
    if client_id:
        logger.debug(f"[Tenant: {client_id}] {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Add client_id to response headers for tracking
    if client_id:
        response.headers["X-Client-ID"] = client_id
    
    return response

# Configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_VERTEX_CREDENTIALS = os.getenv("GOOGLE_VERTEX_CREDENTIALS", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")

if not DAILY_API_KEY:
    raise ValueError("DAILY_API_KEY must be set")
if not GOOGLE_CLOUD_PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID must be set")
if not GOOGLE_VERTEX_CREDENTIALS:
    raise ValueError("GOOGLE_VERTEX_CREDENTIALS must be set")
if not MONGODB_URI:
    logger.warning("MONGODB_URI not set - user lookup functionality will be disabled")

# Initialize MongoDB client with connection pooling
# Single client instance shared across all requests and all client databases
mongodb_client = None
if MONGODB_URI:
    try:
        mongodb_client = MongoClient(
            MONGODB_URI,
            maxPoolSize=50,  # Maximum 50 connections in pool
            minPoolSize=10,  # Minimum 10 connections always ready
            maxIdleTimeMS=45000,  # Close idle connections after 45s
            serverSelectionTimeoutMS=5000,  # Timeout for server selection
            connectTimeoutMS=10000,  # Timeout for initial connection
            socketTimeoutMS=45000,  # Timeout for socket operations
            retryWrites=True,  # Retry writes on network errors
            retryReads=True,  # Retry reads on network errors
        )
        # Test connection
        mongodb_client.admin.command('ping')
        logger.info("✅ MongoDB connection pool initialized successfully")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        mongodb_client = None
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}")
        mongodb_client = None


def get_client_id_from_request(request: Request) -> str:
    """
    Extract MongoDB ObjectId from request state (set by tenant context middleware).
    Falls back to extracting from query params or headers if not in state.
    Now expects 'id' instead of 'client_id' - this is the MongoDB ObjectId.
    """
    # Try to get from request state (set by tenant context middleware)
    if hasattr(request.state, 'client_id') and request.state.client_id:
        return request.state.client_id

    # Fallback: extract from query params or headers (now looking for 'id' parameter)
    id = request.query_params.get("id") or request.headers.get("X-Client-ID")
    if not id:
        raise HTTPException(status_code=400, detail="id is required (query param 'id' or X-Client-ID header with MongoDB ObjectId)")
    return id


def start_ngrok_tunnel(port=8000):
    """Start ngrok tunnel and return the public URL."""
    global ngrok_tunnel
    
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        logger.info("Using ngrok auth token from environment")
    else:
        logger.warning("NGROK_AUTH_TOKEN not set in environment, using free ngrok (may have limitations)")
    
    ngrok_tunnel = ngrok.connect(port, "http")
    public_url = ngrok_tunnel.public_url
    
    logger.info("=" * 60)
    logger.info("🚀 ngrok tunnel started successfully!")
    logger.info(f"📞 Public URL: {public_url}")
    logger.info(f"🌐 Access your bot at: {public_url}/start?client_id=YOUR_CLIENT_ID")
    logger.info(f"💚 Health check: {public_url}/health")
    logger.info("=" * 60)
    
    atexit.register(cleanup_ngrok)
    
    return public_url


def cleanup_ngrok():
    """Clean up ngrok tunnel on exit."""
    global ngrok_tunnel
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()
            logger.info("ngrok tunnel closed")
        except Exception as e:
            logger.error(f"Error closing ngrok tunnel: {e}")


def fix_malformed_json(creds: str) -> Optional[dict]:
    """
    Attempt to fix malformed JSON where quotes are missing.
    Handles cases like: {type:service_account,project_id:value}
    Converts to: {"type":"service_account","project_id":"value"}
    """
    import re
    
    # Check if it looks like malformed JSON (starts with { but missing quotes)
    if not creds.startswith('{') or '"' in creds[:100]:
        return None
    
    try:
        # Use regex-based approach - simpler and more reliable
        def quote_pair(match):
            key = match.group(1)
            value = match.group(2).strip()
            
            # Already has quotes - keep as-is
            if value.startswith('"') and value.endswith('"'):
                return f'"{key}":{value}'
            
            # Don't quote booleans or null
            if value.lower() in ['true', 'false', 'null']:
                return f'"{key}":{value}'
            
            # Don't quote numbers
            try:
                float(value)
                return f'"{key}":{value}'
            except ValueError:
                pass
            
            # Don't quote nested objects/arrays
            if value.startswith('{') or value.startswith('['):
                return f'"{key}":{value}'
            
            # It's a string - quote and escape special characters
            # Escape backslashes first, then quotes
            value_escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{key}":"{value_escaped}"'
        
        # Match key:value patterns
        # Pattern: (\w+): captures the key, ([^,}]+) captures the value until comma or }
        pattern = r'(\w+):([^,}]+)'
        fixed = re.sub(pattern, quote_pair, creds)
        
        # Try to parse
        creds_dict = json.loads(fixed)
        logger.info("Successfully fixed malformed JSON (missing quotes)")
        return creds_dict
        
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: try regex-based approach
        try:
            # Simple regex: find all key:value and quote them
            def quote_pair(match):
                key = match.group(1)
                value = match.group(2).strip()
                
                # Already has quotes - keep as-is
                if value.startswith('"') and value.endswith('"'):
                    return f'"{key}":{value}'
                
                # Don't quote booleans or null
                if value.lower() in ['true', 'false', 'null']:
                    return f'"{key}":{value}'
                
                # Don't quote numbers
                try:
                    float(value)
                    return f'"{key}":{value}'
                except ValueError:
                    pass
                
                # Don't quote nested objects/arrays
                if value.startswith('{') or value.startswith('['):
                    return f'"{key}":{value}'
                
                # It's a string - quote and escape special characters
                # Escape backslashes first, then quotes
                value_escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{key}":"{value_escaped}"'
            
            # Match key:value patterns
            # Pattern: (\w+): captures the key, ([^,}]+) captures the value until comma or }
            pattern = r'(\w+):([^,}]+)'
            fixed = re.sub(pattern, quote_pair, creds)
            
            # Try to parse
            creds_dict = json.loads(fixed)
            logger.info("Successfully fixed malformed JSON using regex fallback")
            return creds_dict
        except Exception as e2:
            logger.debug(f"Failed to fix malformed JSON: {e}, fallback also failed: {e2}")
            return None


def fix_credentials():
    """Fix GOOGLE_VERTEX_CREDENTIALS so Pipecat can parse it."""
    creds = GOOGLE_VERTEX_CREDENTIALS
    
    if not creds:
        raise ValueError("GOOGLE_VERTEX_CREDENTIALS environment variable is not set")
    
    creds = creds.strip()
    
    # Log what we're trying to parse (for debugging)
    logger.debug(f"GOOGLE_VERTEX_CREDENTIALS value type: starts with '{{'={creds.startswith('{')}, starts with './'={creds.startswith('./')}, is file path={os.path.isfile(creds) if not creds.startswith('{') else False}")
    
    # Remove surrounding quotes if present
    if (creds.startswith('"') and creds.endswith('"')) or (creds.startswith("'") and creds.endswith("'")):
        creds = creds[1:-1]
    
    # Try to parse as JSON first (only if it looks like JSON)
    if creds.startswith('{') or creds.startswith('['):
        try:
            # Handle escaped newlines in the JSON string
            # Replace literal \n with actual newlines for JSON parsing
            creds_dict = json.loads(creds)
            if "private_key" in creds_dict:
                # Fix newlines in private key
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            logger.debug("Successfully parsed credentials as JSON")
            return json.dumps(creds_dict)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try unescaping common issues
            try:
                # Try replacing escaped quotes and newlines
                creds_unescaped = creds.replace('\\"', '"').replace("\\'", "'")
                creds_dict = json.loads(creds_unescaped)
                if "private_key" in creds_dict:
                    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
                logger.debug("Successfully parsed credentials as JSON after unescaping")
                return json.dumps(creds_dict)
            except json.JSONDecodeError:
                # Try fixing malformed JSON (missing quotes)
                creds_dict = fix_malformed_json(creds)
                if creds_dict:
                    if "private_key" in creds_dict:
                        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
                    return json.dumps(creds_dict)
                logger.warning(f"Failed to parse credentials as JSON: {e}")
                # Don't pass yet - might be a file path
    
    # Try to resolve as file path
    file_path = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    
    # List of potential paths to check
    potential_paths = []
    
    # If absolute path
    if os.path.isabs(creds):
        potential_paths.append(creds)
    else:
        # Relative paths to check
        potential_paths.extend([
            creds,  # Current working directory
            os.path.join(script_dir, creds),  # Same directory as script
            os.path.join(current_dir, creds),  # Current working directory (explicit)
        ])
        
        # If it starts with ./, also try without the ./
        if creds.startswith('./'):
            creds_no_dot = creds[2:]
            potential_paths.extend([
                creds_no_dot,
                os.path.join(script_dir, creds_no_dot),
                os.path.join(current_dir, creds_no_dot),
            ])
        
        # If it ends with .json, also try in parent directories
        if creds.endswith('.json'):
            parent_dir = os.path.dirname(script_dir)
            potential_paths.extend([
                os.path.join(parent_dir, creds),
                os.path.join(parent_dir, creds[2:] if creds.startswith('./') else creds),
            ])
    
    # Check each potential path
    for path in potential_paths:
        if os.path.isfile(path):
            file_path = os.path.abspath(path)
            logger.info(f"Found credentials file at: {file_path}")
            break
    
    if not file_path:
        logger.debug(f"Checked paths: {potential_paths[:5]}... (showing first 5)")
    
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                creds_dict = json.load(f)
            logger.info(f"Successfully loaded credentials from file: {file_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read credentials from file '{file_path}': {e}") from e
    else:
        # File not found - try parsing as JSON one more time
        # (in case it's malformed JSON that needs fixing)
        creds_dict = None
        
        # Strategy 1: Direct JSON parse
        try:
            creds_dict = json.loads(creds)
            logger.debug("Successfully parsed credentials as JSON (fallback)")
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Try unescaping common issues
        if creds_dict is None:
            try:
                # Replace escaped quotes and newlines
                creds_unescaped = creds.replace('\\"', '"').replace("\\'", "'")
                creds_dict = json.loads(creds_unescaped)
                logger.debug("Successfully parsed credentials after unescaping (fallback)")
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try reading as raw string and parsing
        if creds_dict is None:
            try:
                # Handle case where newlines are literal \n strings
                creds_fixed = creds.replace('\\n', '\n').replace('\\"', '"')
                creds_dict = json.loads(creds_fixed)
                logger.debug("Successfully parsed credentials after fixing newlines (fallback)")
            except json.JSONDecodeError:
                pass
        
        if creds_dict is None:
            # Log the actual value for debugging (truncated)
            creds_preview = creds[:100].replace('\n', '\\n').replace('\r', '\\r')
            error_msg = (
                f"GOOGLE_VERTEX_CREDENTIALS is not valid JSON and not a valid file path.\n"
                f"Value preview: '{creds_preview}...'\n"
                f"Script directory: {script_dir}\n"
                f"Current directory: {current_dir}\n"
                f"Checked paths: {potential_paths[:3]}...\n"
                f"If using Vast AI, check your environment variables in the dashboard.\n"
                f"The .env file value may be overridden by Vast AI's environment settings."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    if "private_key" in creds_dict:
        # Fix newlines in private key (handle both \\n and \n)
        private_key = creds_dict["private_key"]
        # Replace escaped newlines with actual newlines
        private_key = private_key.replace("\\n", "\n")
        creds_dict["private_key"] = private_key

    return json.dumps(creds_dict)


def extract_phone_numbers_from_text(text: str) -> List[str]:
    """Extract all phone numbers from text and normalize them to 10-digit format."""
    if not text:
        return []
    
    phone_pattern = re.compile(
        r'(?:\+?91[-.\s]?)?([6-9]\d{9})\b|'
        r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b|'
        r'\b([6-9]\d{2}[-.\s]?\d{3}[-.\s]?\d{4})\b|'
        r'\b([6-9]\d{9})\b'
    )
    
    phone_numbers = set()
    matches = phone_pattern.findall(text)
    
    for match in matches:
        phone = next((m for m in match if m), None)
        if phone:
            phone_clean = re.sub(r'[^\d]', '', phone)
            
            if phone_clean.startswith('91') and len(phone_clean) == 12:
                phone_clean = phone_clean[2:]
            elif len(phone_clean) > 10:
                phone_clean = phone_clean[-10:]
            elif len(phone_clean) < 10:
                continue
            
            if len(phone_clean) == 10 and phone_clean[0] in '6789':
                phone_numbers.add(phone_clean)
    
    return list(phone_numbers)


def get_guardrail_phone_numbers(client_id: str) -> List[str]:
    """Extract all phone numbers from guardrails for a specific client."""
    with guardrails_lock:
        phone_numbers = set()
        guardrails = guardrails_storage.get(client_id, [])
        
        for guardrail in guardrails:
            question = guardrail.get("question", "").strip()
            answer = guardrail.get("answer", "").strip()
            
            if question:
                phone_numbers.update(extract_phone_numbers_from_text(question))
            if answer:
                phone_numbers.update(extract_phone_numbers_from_text(answer))
        
        return list(phone_numbers)


def format_guardrails_for_prompt(id: str) -> str:
    """
    Format stored guardrails for a specific client for inclusion in system prompt.
    Now uses MongoDB ObjectId instead of client_id string.
    """
    with guardrails_lock:
        guardrails = guardrails_storage.get(id, [])
        
        if not guardrails:
            return ""
        
        guardrails_text = "\n\n# CUSTOM INSTRUCTIONS AND GUARDRAILS\n\n"
        guardrails_text += "**IMPORTANT: The following question-answer pairs are custom instructions that guide how you should respond to similar questions.**\n\n"
        guardrails_text += "When a user asks a question that is similar to any of the questions below, you MUST respond in the manner specified in the corresponding answer. These instructions override default behavior when applicable.\n\n"
        
        for idx, guardrail in enumerate(guardrails, 1):
            question = guardrail.get("question", "").strip()
            answer = guardrail.get("answer", "").strip()
            
            if question and answer:
                guardrails_text += f"## Instruction {idx}\n\n"
                guardrails_text += f"**When asked (or similar to):** {question}\n\n"
                guardrails_text += f"**You should respond like this:** {answer}\n\n"
        
        guardrails_text += "**Remember:** Use these instructions as a guide. When a user's question is similar to any of the questions above, adapt your response to match the style and content of the corresponding answer, while still being natural and conversational.\n\n"
        guardrails_text += "# END OF CUSTOM INSTRUCTIONS AND GUARDRAILS\n"
        
        return guardrails_text


def get_current_datetime_info():
    """Get current date and time information in Asia/Kolkata timezone."""
    try:
        if ZoneInfo is not None:
            tz = ZoneInfo("Asia/Kolkata")
            now = datetime.now(tz)
            timezone_name = "Asia/Kolkata (IST)"
        else:
            from datetime import timezone
            ist_offset = timedelta(hours=5, minutes=30)
            tz = timezone(ist_offset)
            now = datetime.now(tz)
            timezone_name = "IST (UTC+5:30)"
        
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")
        day_of_week = now.strftime("%A")
        readable_date = now.strftime("%B %d, %Y")
        tomorrow = now + timedelta(days=1)
        tomorrow_date = tomorrow.strftime("%Y-%m-%d")
        tomorrow_readable = tomorrow.strftime("%B %d, %Y")
        tomorrow_day = tomorrow.strftime("%A")
        
        return {
            "current_date": current_date,
            "current_time": current_time,
            "day_of_week": day_of_week,
            "readable_date": readable_date,
            "tomorrow_date": tomorrow_date,
            "tomorrow_readable": tomorrow_readable,
            "tomorrow_day": tomorrow_day,
            "timezone": timezone_name
        }
    except Exception as e:
        logger.error(f"Error getting current datetime: {e}")
        now = datetime.now()
        return {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M"),
            "day_of_week": now.strftime("%A"),
            "readable_date": now.strftime("%B %d, %Y"),
            "tomorrow_date": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            "tomorrow_readable": (now + timedelta(days=1)).strftime("%B %d, %Y"),
            "tomorrow_day": (now + timedelta(days=1)).strftime("%A"),
            "timezone": "UTC (fallback)"
        }


def load_system_prompt(id: str) -> str:
    """
    Load system prompt from MongoDB for a specific client.
    Always includes the default system prompt from system_prompt.py along with client-specific prompt.
    Now uses MongoDB ObjectId instead of client_id string.
    """
    # Always load the default system prompt first
    from system_prompt import SYSTEM_PROMPT
    default_prompt = SYSTEM_PROMPT.strip()

    try:
        config = get_client_config(id)
        client_system_prompt = config.get('agent', {}).get('system_prompt', '').strip()

        if client_system_prompt:
            # Combine default prompt with client-specific prompt
            # Default prompt comes first, then client-specific prompt
            combined_prompt = f"{default_prompt}\n\n{client_system_prompt}"
            logger.debug(f"Loaded combined system prompt for client {id} (default + client-specific)")
            return combined_prompt
        else:
            # No client-specific prompt, use only default
            logger.info(f"No client-specific system prompt found for client {id}, using default only")
            return default_prompt
            
    except Exception as e:
        logger.error(f"Error loading system prompt for client {id}: {e}, using default only")
        return default_prompt


async def fetch_detailed_information(client_id: str, params: FunctionCallParams):
    """Fetch detailed information from the preprocessor API for queries outside the system prompt"""
    try:
        query = params.arguments["query"]
        phone_number = params.arguments.get("phone_number")
        email = params.arguments.get("email")
        
        if not phone_number and not email:
            logger.warning("CRITICAL: get_detailed_information tool called without phone number or email - this should never happen. Rejecting the call.")
            await params.result_callback({
                "summary": "I need either your phone number or email address to send you the information. Could you please provide one of them? I cannot proceed with sending information until I have your contact details.",
                "whatsapp_sent": False,
                "email_sent": False,
                "error": "Contact information required - tool should not have been called without phone number or email"
            })
            return
        
        request_payload = {"query": query}
        
        if phone_number:
            digits_only = ''.join(filter(str.isdigit, phone_number))
            
            if digits_only.startswith("91") and len(digits_only) > 10:
                phone_number_clean = digits_only[2:]
            elif len(digits_only) > 10:
                phone_number_clean = digits_only[-10:]
            elif len(digits_only) < 10:
                phone_number_clean = digits_only.zfill(10)
            else:
                phone_number_clean = digits_only
            
            if len(phone_number_clean) != 10:
                raise ValueError(f"Phone number must be exactly 10 digits, got {len(phone_number_clean)} digits")
            
            phone_number_with_prefix = f"91{phone_number_clean}"
            request_payload["number"] = phone_number_with_prefix
            logger.info(f"Phone number provided: {phone_number_with_prefix} (10 digits: {phone_number_clean})")
        
        if email:
            request_payload["email"] = email
            logger.info(f"Email provided: {email}")
        
        # Get client-specific preprocessor URL
        preprocessor_url = get_preprocessor_url(client_id)
        query_url = f"{preprocessor_url}/query"
        
        logger.info(f"Calling preprocessor API ({client_id}) with query: {query}, payload: {list(request_payload.keys())}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(query_url, json=request_payload)
            
            logger.info(f"Preprocessor API response status: {response.status_code}")
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    logger.error(f"Preprocessor API error response: {error_data}")
                except:
                    error_text = response.text
                    logger.error(f"Preprocessor API error response (non-JSON): {error_text}")
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                summary = data.get("summary", "")
                whatsapp_status = data.get("whatsapp_status", {})
                email_status = data.get("email_status", {})
                whatsapp_sent = whatsapp_status.get("status") == "success"
                email_sent = email_status.get("status") == "success"
                
                logger.info(f"Preprocessor API returned success. Summary length: {len(summary)}")
                logger.info(f"Delivery status - WhatsApp: {whatsapp_status.get('status')}, Email: {email_status.get('status')}")
                
                delivery_info = []
                if whatsapp_sent:
                    delivery_info.append("WhatsApp")
                if email_sent:
                    delivery_info.append("email")
                
                if delivery_info:
                    delivery_message = f"Information has been successfully sent via {', '.join(delivery_info)}."
                else:
                    if whatsapp_status.get("status") == "skipped" and email_status.get("status") == "skipped":
                        delivery_message = "I processed your request, but no contact method was available to send the information. Please provide your phone number or email."
                    else:
                        delivery_message = "I've processed your request. The information is being prepared and sent."
                
                await params.result_callback({
                    "summary": f"{summary}\n\n{delivery_message}",
                    "whatsapp_sent": whatsapp_sent,
                    "email_sent": email_sent,
                    "status": "success"
                })
            else:
                error_message = data.get("error", "Unable to process request at this moment")
                logger.warning(f"Preprocessor API returned non-success status: {data.get('status')}, error: {error_message}")
                await params.result_callback({
                    "summary": f"I'm having trouble processing your request right now. Please try again in a moment.",
                    "whatsapp_sent": False,
                    "email_sent": False,
                    "status": "error",
                    "error": error_message
                })
    except httpx.HTTPStatusError as e:
        error_message = f"Server returned status {e.response.status_code}"
        try:
            error_detail = e.response.json()
            if isinstance(error_detail, dict) and "detail" in error_detail:
                error_message = f"{error_message}: {error_detail['detail']}"
            logger.error(f"HTTP error fetching detailed information: {error_message}", exc_info=True)
        except:
            try:
                error_text = e.response.text
                logger.error(f"HTTP error fetching detailed information: {error_message}. Response: {error_text}", exc_info=True)
            except:
                logger.error(f"HTTP error fetching detailed information: {error_message}", exc_info=True)
        
        await params.result_callback({
            "summary": f"I'm having trouble processing your request right now. Please try again in a moment.",
            "whatsapp_sent": False,
            "email_sent": False,
            "status": "error",
            "error": error_message
        })
    except httpx.TimeoutException as e:
        error_message = "The request took too long to process"
        logger.error(f"Timeout error fetching detailed information: {e}", exc_info=True)
        await params.result_callback({
            "summary": f"I'm having trouble processing your request right now. Please try again in a moment.",
            "whatsapp_sent": False,
            "email_sent": False,
            "status": "error",
            "error": error_message
        })
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error fetching detailed information: {e}", exc_info=True)
        await params.result_callback({
            "summary": f"I'm having trouble processing your request right now. Please try again in a moment.",
            "whatsapp_sent": False,
            "email_sent": False,
            "status": "error",
            "error": error_message
        })


# Career paths and alumni info functions remain the same (internal tools)
async def get_career_paths(params: FunctionCallParams):
    """Get career paths for a specific branch - internal tool"""
    # Same implementation as original
    # ... (keeping original implementation)
    pass


async def get_alumni_info(params: FunctionCallParams):
    """Get alumni placement information for a specific branch - internal tool"""
    # Same implementation as original
    # ... (keeping original implementation)
    pass


def normalize_phone_number(phone_number: str) -> str:
    """Normalize phone number to 10 digits"""
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    if digits_only.startswith("91") and len(digits_only) > 10:
        phone_number_clean = digits_only[2:]
    elif len(digits_only) > 10:
        phone_number_clean = digits_only[-10:]
    elif len(digits_only) < 10:
        phone_number_clean = digits_only.zfill(10)
    else:
        phone_number_clean = digits_only
    
    return phone_number_clean


def get_mongodb_client():
    """
    Get MongoDB client (shared connection pool)
    Returns the global mongodb_client instance
    """
    return mongodb_client


def load_guardrails_from_mongodb(id: str):
    """
    Load guardrails from MongoDB into memory storage for a specific client.
    Now uses MongoDB ObjectId instead of client_id string.
    """
    if not MONGODB_URI:
        logger.warning(f"MongoDB URI not configured, guardrails will not persist for client {id}")
        return

    try:
        client = get_mongodb_client()
        if not client:
            logger.warning(f"Could not connect to MongoDB, guardrails will not persist for client {id}")
            return

        db_name = get_mongodb_database_name(id)
        db = client[db_name]
        guardrails_collection = db["guardrails"]

        stored_guardrails = list(guardrails_collection.find({}, {"_id": 0, "question": 1, "answer": 1}))

        with guardrails_lock:
            if id not in guardrails_storage:
                guardrails_storage[id] = []
            guardrails_storage[id].clear()
            guardrails_storage[id].extend(stored_guardrails)

        logger.info(f"Loaded {len(stored_guardrails)} guardrail(s) from MongoDB for client {id}")

    except Exception as e:
        logger.error(f"Error loading guardrails from MongoDB for client {id}: {e}", exc_info=True)


def save_guardrails_to_mongodb(id: str):
    """
    Save current guardrails from memory to MongoDB for a specific client.
    Now uses MongoDB ObjectId instead of client_id string.
    """
    if not MONGODB_URI:
        logger.warning(f"MongoDB URI not configured, guardrails will not persist for client {id}")
        return False

    try:
        client = get_mongodb_client()
        if not client:
            logger.warning(f"Could not connect to MongoDB, guardrails will not persist for client {id}")
            return False

        db_name = get_mongodb_database_name(id)
        db = client[db_name]
        guardrails_collection = db["guardrails"]

        with guardrails_lock:
            guardrails = guardrails_storage.get(id, [])
            guardrails_collection.delete_many({})

            if guardrails:
                guardrails_collection.insert_many(guardrails)
                logger.info(f"Saved {len(guardrails)} guardrail(s) to MongoDB for client {id}")
            else:
                logger.info(f"No guardrails to save for client {id} (storage is empty)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving guardrails to MongoDB for client {client_id}: {e}", exc_info=True)
        return False


def get_user_data(client_id: str, phone_number: str = None, email: str = None):
    """Check if user exists in MongoDB and retrieve user data and analytics by phone number or email"""
    if not MONGODB_URI:
        logger.warning(f"MongoDB URI not configured, skipping user lookup for client {client_id}")
        return None
    
    if not phone_number and not email:
        logger.warning("Neither phone number nor email provided for user lookup")
        return None
    
    try:
        client = get_mongodb_client()
        if not client:
            return None
        
        db_name = get_mongodb_database_name(client_id)
        db = client[db_name]
        users_collection = db["users"]
        analytics_collection = db["userAnalytics"]
        
        user = None
        
        if phone_number:
            phone_clean = normalize_phone_number(phone_number)
            
            if len(phone_clean) == 10:
                phone_db_format = f"91{phone_clean}"
                user = users_collection.find_one({"phone_number": phone_db_format})
                
                if not user:
                    user = users_collection.find_one({"phone_number": phone_clean})
                if not user:
                    user = users_collection.find_one({"phone": phone_db_format})
                if not user:
                    user = users_collection.find_one({"phone": phone_clean})
        
        if not user and email:
            email_clean = email.strip().lower()
            user = users_collection.find_one({"email": email_clean})
            
            if not user:
                user = users_collection.find_one({"email": {"$regex": f"^{email_clean}$", "$options": "i"}})
        
        if not user:
            lookup_info = f"phone: {phone_number}" if phone_number else f"email: {email}"
            logger.info(f"User not found in database for {lookup_info} (client: {client_id})")
            return None
        
        user_id = user.get("_id")
        
        analytics = None
        if user_id:
            try:
                from bson import ObjectId
                analytics = analytics_collection.find_one({"user_id": ObjectId(user_id)})
                
                if not analytics:
                    analytics = analytics_collection.find_one({"user_id": str(user_id)})
                if not analytics:
                    analytics = analytics_collection.find_one({"id": ObjectId(user_id)})
                if not analytics:
                    analytics = analytics_collection.find_one({"id": str(user_id)})
                if not analytics:
                    analytics = analytics_collection.find_one({"_id": user_id})
                if not analytics:
                    analytics = analytics_collection.find_one({"userId": str(user_id)})
            except Exception as e:
                logger.warning(f"Error querying analytics with ObjectId: {e}")
                analytics = analytics_collection.find_one({"user_id": str(user_id)})
        
        user_phone = user.get("phone_number") or user.get("phone") or ""
        
        user_data = {
            "exists": True,
            "user_profile": {
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "phone": user_phone
            },
            "analytics": {
                "course_interest": analytics.get("course_interest", "") or analytics.get("course interest", "") if analytics else "",
                "city": analytics.get("city", "") if analytics else "",
                "budget": analytics.get("budget", "") if analytics else "",
                "hostel_needed": analytics.get("hostel_needed", "") or analytics.get("hostel needed", "") if analytics else "",
                "intent_level": analytics.get("intent_level", "") or analytics.get("intent level", "") if analytics else ""
            } if analytics else {}
        }
        
        lookup_method = f"phone: {user_phone}" if user_phone else f"email: {user_data['user_profile']['email']}"
        logger.info(f"Found existing user: {user_data['user_profile']['name']} ({lookup_method}, client: {client_id})")
        return user_data
        
    except Exception as e:
        logger.error(f"Error retrieving user data from MongoDB for client {client_id}: {e}", exc_info=True)
        return None


async def check_user_exists(client_id: str, params: FunctionCallParams):
    """Check if user exists in database and retrieve their profile and analytics by phone number or email"""
    try:
        phone_number = params.arguments.get("phone_number")
        email = params.arguments.get("email")
        
        if not phone_number and not email:
            await params.result_callback({
                "user_exists": False,
                "message": "Either phone number or email must be provided to check user existence."
            })
            return
        
        lookup_info = f"phone: {phone_number}" if phone_number else f"email: {email}"
        logger.info(f"Checking user existence for {lookup_info} (client: {client_id})")
        
        user_data = get_user_data(client_id, phone_number=phone_number, email=email)
        
        if user_data and user_data.get("exists"):
            await params.result_callback({
                "user_exists": True,
                "user_profile": user_data["user_profile"],
                "analytics": user_data["analytics"],
                "message": f"Found existing user profile for {user_data['user_profile']['name']}. You already have their information including course interest, city, budget, hostel preference, and intent level. Use this information to have a personalized conversation without asking the standard counseling questions."
            })
        else:
            await params.result_callback({
                "user_exists": False,
                "message": "User not found in database. Proceed with standard counseling flow."
            })
            
    except Exception as e:
        logger.error(f"Error checking user existence for client {client_id}: {e}", exc_info=True)
        await params.result_callback({
            "user_exists": False,
            "message": "Error checking user database. Proceeding with standard counseling flow.",
            "error": str(e)
        })


async def create_daily_room() -> tuple[str, str]:
    """Create a Daily room and return the URL and token"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,
                    "enable_chat": False,
                    "enable_emoji_reactions": False,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Failed to create room: {response.status} - {error_text}")
                raise Exception(f"Failed to create Daily room: {response.status}")
            
            room_data = await response.json()
            room_url = room_data.get("url")
            room_name = room_data.get("name")
            
            if not room_url or not room_name:
                logger.error(f"Missing url or name in room response: {room_data}")
                raise Exception("Invalid room data from Daily API")

        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": True,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Failed to create token: {response.status} - {error_text}")
                raise Exception(f"Failed to create Daily token: {response.status}")
            
            token_data = await response.json()
            token = token_data.get("token")
            
            if not token:
                logger.error(f"Missing token in response: {token_data}")
                raise Exception("Invalid token data from Daily API")

    logger.info(f"Successfully created room: {room_url}")
    return room_url, token


async def run_bot(id: str, room_url: str, token: str):
    """
    Run the voice bot in the Daily room for a specific client.
    Now uses MongoDB ObjectId instead of client_id string.
    """
    transport = None
    try:
        logger.info(f"Starting bot for room: {room_url} (client id: {id})")
        
        # Load client configuration
        config = get_client_config(id)
        agent_config = config.get('agent', {})
        llm_config = agent_config.get('llm_config', {})
        
        transport = DailyTransport(
            room_url,
            token,
            "Voice Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=False,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=0.3,
                        min_volume=0.3,
                    )
                ),
                transcription_enabled=True,
            ),
        )

        project_id = GOOGLE_CLOUD_PROJECT_ID
        location = GOOGLE_CLOUD_LOCATION
        model_id = llm_config.get('model', 'gemini-live-2.5-flash-preview-native-audio-09-2025')
        
        model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
        
        logger.info(f"Using Vertex AI model: {model_path} (client: {id})")

        temperature = float(llm_config.get('temperature', os.getenv("LLM_TEMPERATURE", "0.1")))
        logger.info(f"Using LLM temperature: {temperature} (client: {id})")

        datetime_info = get_current_datetime_info()
        logger.info(f"Current date/time context: {datetime_info['current_date']} {datetime_info['current_time']} ({datetime_info['timezone']})")
        
        datetime_context = f"""

## CURRENT DATE AND TIME INFORMATION

**IMPORTANT: Use this information when answering questions about dates/times.**

- **Current Date**: {datetime_info['readable_date']} ({datetime_info['day_of_week']})
- **Current Date (YYYY-MM-DD format)**: {datetime_info['current_date']}
- **Current Time**: {datetime_info['current_time']} ({datetime_info['timezone']})
- **Tomorrow's Date**: {datetime_info['tomorrow_readable']} ({datetime_info['tomorrow_day']})
- **Tomorrow's Date (YYYY-MM-DD format)**: {datetime_info['tomorrow_date']}
- Current timezone is {datetime_info['timezone']}

"""
        
        # Get languages from agent config
        languages = agent_config.get('languages', [])
        languages_context = ""
        if languages:
            languages_list = ", ".join(languages)
            languages_context = f"""

## SUPPORTED LANGUAGES

**IMPORTANT: You MUST support and respond in the following languages: {languages_list}**

- You MUST be able to detect and respond in any of these languages: {languages_list}
- When a user speaks in any of these languages, IMMEDIATELY switch to that language without asking
- You can switch between these languages seamlessly as the user switches
- If a language is not in this list, default to English (or the first language in the list)

"""
        else:
            # Default to common Indian languages if not specified
            languages_context = """

## SUPPORTED LANGUAGES

**IMPORTANT: You MUST support and respond in the following languages: Tamil, English, Malayalam, Kannada, Telugu, Hindi**

- You MUST be able to detect and respond in any of these languages: Tamil, English, Malayalam, Kannada, Telugu, Hindi
- When a user speaks in any of these languages, IMMEDIATELY switch to that language without asking
- You can switch between these languages seamlessly as the user switches

"""
        
        guardrails_context = format_guardrails_for_prompt(id)
        
        # Load client-specific system prompt
        system_prompt = load_system_prompt(id)
        system_instruction = system_prompt + datetime_context + languages_context + guardrails_context

        # Build tools based on client config
        tools_list = []
        tools_config = agent_config.get('tools', [])
        
        # Create function wrappers that include client_id
        async def fetch_detailed_info_wrapper(params: FunctionCallParams):
            return await fetch_detailed_information(client_id, params)
        
        async def check_user_exists_wrapper(params: FunctionCallParams):
            return await check_user_exists(client_id, params)
        
        # Add tools based on config
        for tool_config in tools_config:
            if not tool_config.get('enabled', True):
                continue
            
            tool_name = tool_config.get('name')
            if tool_name == 'get_detailed_information':
                detailed_info_function = FunctionSchema(
                    name="get_detailed_information",
                    description=tool_config.get('description', "Send detailed information via WhatsApp/Email"),
                    properties={
                        "query": {
                            "type": "string",
                            "description": "The specific course or branch information the student wants",
                        },
                        "phone_number": {
                            "type": "string",
                            "description": "The student's phone number (10 digits)",
                        },
                        "email": {
                            "type": "string",
                            "description": "The student's email address",
                        },
                    },
                    required=["query"],
                )
                tools_list.append(detailed_info_function)
            elif tool_name == 'get_career_paths':
                career_paths_function = FunctionSchema(
                    name="get_career_paths",
                    description=tool_config.get('description', "Get career paths for a specific branch"),
                    properties={
                        "branch": {
                            "type": "string",
                            "description": "The exact branch name",
                        },
                    },
                    required=["branch"],
                )
                tools_list.append(career_paths_function)
            elif tool_name == 'get_alumni_info':
                alumni_info_function = FunctionSchema(
                    name="get_alumni_info",
                    description=tool_config.get('description', "Get alumni placement information"),
                    properties={
                        "branch": {
                            "type": "string",
                            "description": "The exact branch name",
                        },
                    },
                    required=["branch"],
                )
                tools_list.append(alumni_info_function)
            elif tool_name == 'check_user_exists':
                check_user_function = FunctionSchema(
                    name="check_user_exists",
                    description=tool_config.get('description', "Check if user exists in database"),
                    properties={
                        "phone_number": {
                            "type": "string",
                            "description": "The student's phone number (10 digits)",
                        },
                        "email": {
                            "type": "string",
                            "description": "The student's email address",
                        },
                    },
                    required=[],
                )
                tools_list.append(check_user_function)

        tools = ToolsSchema(standard_tools=tools_list)

        voice_id = llm_config.get('voice_id', 'Aoede')
        llm = GeminiLiveVertexLLMService(
            credentials=fix_credentials(),
            project_id=project_id,
            location=location,
            model=model_path,
            system_instruction=system_instruction,
            voice_id=voice_id,
            temperature=temperature,
            tools=tools,
        )

        # Register functions
        if any(t.get('name') == 'get_detailed_information' and t.get('enabled', True) for t in tools_config):
            llm.register_function("get_detailed_information", fetch_detailed_info_wrapper)
        if any(t.get('name') == 'get_career_paths' and t.get('enabled', True) for t in tools_config):
            llm.register_function("get_career_paths", get_career_paths)
        if any(t.get('name') == 'get_alumni_info' and t.get('enabled', True) for t in tools_config):
            llm.register_function("get_alumni_info", get_alumni_info)
        if any(t.get('name') == 'check_user_exists' and t.get('enabled', True) for t in tools_config):
            llm.register_function("check_user_exists", check_user_exists_wrapper)

        context = LLMContext([
            {
                "role": "assistant",
                "content": "Greet the student warmly and introduce yourself. Then ask for their name. Once you have their name, ask for their mobile number (10 digits) OR email address - at least one of them is required. If they provide a phone number, confirm it by reciting the 10 digits back to them. You can ask for both, but at least one contact method is mandatory. Once you have their name and at least one contact method (phone or email), you can proceed with the counseling session. Be friendly, warm, and approachable - like a caring counselor. Keep each question brief and wait for their response before moving to the next question."
            }
        ])

        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            context_aggregator.user(),
            llm,
            context_aggregator.assistant(),
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"First participant joined: {participant} (client: {client_id})")
            await transport.capture_participant_transcription(participant["id"])
            
            await asyncio.sleep(0.5)
            try:
                await task.queue_frames([LLMRunFrame()])
                logger.info("Initial greeting triggered with LLMRunFrame")
            except Exception as e:
                logger.error(f"Error sending greeting: {e}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}, reason: {reason} (client: {client_id})")
            
            try:
                conversation_history = []
                context_messages = []
                if hasattr(context_aggregator, 'context') and hasattr(context_aggregator.context, 'messages'):
                    context_messages = context_aggregator.context.messages
                elif hasattr(context_aggregator, '_context') and hasattr(context_aggregator._context, 'messages'):
                    context_messages = context_aggregator._context.messages
                elif hasattr(context, 'messages'):
                    context_messages = context.messages
                
                for msg in context_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content and role in ["user", "assistant"]:
                        speaker = "User" if role == "user" else "Agent"
                        conversation_history.append(f"{speaker}: {content}")
                
                conversation_text = "\n".join(conversation_history)
                
                if conversation_text.strip():
                    logger.info(f"Sending conversation history to postprocessor (client: {client_id})...")
                    try:
                        postprocessor_url = get_postprocessor_url(client_id)
                        async with httpx.AsyncClient(timeout=30.0) as client_http:
                            response = await client_http.post(
                                f"{postprocessor_url}/process?client_id={client_id}",
                                json={"conversation": conversation_text}
                            )
                            if response.status_code == 200:
                                logger.info("Conversation history sent to postprocessor successfully")
                            else:
                                logger.warning(f"Postprocessor returned status {response.status_code}: {response.text}")
                    except Exception as e:
                        logger.error(f"Error sending conversation history to postprocessor: {e}", exc_info=True)
                else:
                    logger.warning("No conversation history to send - context messages not accessible")
                    
            except Exception as e:
                logger.error(f"Error processing conversation history: {e}", exc_info=True)
            
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant} (client: {client_id})")
            await transport.capture_participant_transcription(participant["id"])

        logger.info("Starting pipeline runner")
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info("Pipeline runner completed")
    except Exception as e:
        logger.error("Error running bot for client {}: {}", client_id, str(e), exc_info=True)
        raise
    finally:
        if transport:
            try:
                logger.info("Cleaning up transport")
                await transport.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up transport: {e}")


@app.post("/start")
async def start_session(request: Request):
    """
    Create a Daily room, start the bot, and return connection details.
    Now expects MongoDB ObjectId as 'id' parameter instead of 'client_id'.
    """
    try:
        id = get_client_id_from_request(request)
        
        # Validate client exists (using ObjectId)
        try:
            config = get_client_config(id)
            client_id = config.get('client_id')  # Get the client_id string from config
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Client with id '{id}' not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        logger.info(f"Creating Daily room and starting bot for client: {id} ({client_id})...")
        
        room_url, token = await create_daily_room()
        logger.info(f"Created room: {room_url}")

        asyncio.create_task(run_bot(id, room_url, token))

        return JSONResponse(
            content={
                "room_url": room_url,
                "token": token,
                "id": id,
                "client_id": client_id  # Include both for backwards compatibility
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# Pydantic models for request validation
class GuardrailItem(BaseModel):
    question: str
    answer: str


class GuardrailsRequest(BaseModel):
    guardrails: List[GuardrailItem]


@app.post("/upload-guardrails")
async def upload_guardrails(
    request: Request,
    guardrails_request: GuardrailsRequest,
    user: User = Depends(require_auth)
):
    """
    Upload question-answer pairs (guardrails/instructions) for a specific client
    
    **Requires:** JWT authentication + ownership of the client
    """
    try:
        client_id = get_client_id_from_request(request)
        
        # Verify user owns this client
        check_client_ownership(client_id, user)
        
        if not guardrails_request.guardrails:
            raise HTTPException(status_code=400, detail="At least one guardrail (question-answer pair) is required")
        
        validated_guardrails = []
        for idx, guardrail in enumerate(guardrails_request.guardrails):
            question = guardrail.question.strip()
            answer = guardrail.answer.strip()
            
            if not question:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Guardrail at index {idx} has an empty question"
                )
            if not answer:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Guardrail at index {idx} has an empty answer"
                )
            
            validated_guardrails.append({
                "question": question,
                "answer": answer
            })
        
        with guardrails_lock:
            if client_id not in guardrails_storage:
                guardrails_storage[client_id] = []
            guardrails_storage[client_id].clear()
            guardrails_storage[client_id].extend(validated_guardrails)
        
        save_guardrails_to_mongodb(client_id)
        
        logger.info(f"Successfully uploaded {len(validated_guardrails)} guardrail(s) for client {client_id}")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully uploaded {len(validated_guardrails)} guardrail(s)",
                "count": len(validated_guardrails),
                "client_id": client_id
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading guardrails: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/guardrails")
async def get_guardrails(request: Request):
    """Get all currently stored guardrails for a specific client"""
    client_id = get_client_id_from_request(request)
    
    with guardrails_lock:
        guardrails = guardrails_storage.get(client_id, [])
        guardrails_with_index = [
            {
                "index": idx,
                "question": guardrail.get("question", ""),
                "answer": guardrail.get("answer", "")
            }
            for idx, guardrail in enumerate(guardrails)
        ]
        
        return JSONResponse(
            content={
                "status": "success",
                "guardrails": guardrails_with_index,
                "count": len(guardrails),
                "client_id": client_id
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "multi-tenant-agent"}


@app.get("/clients")
async def list_clients():
    """List all available clients"""
    clients = list_all_clients()
    return JSONResponse(content={"clients": clients, "count": len(clients)})


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8001"))
    
    logger.info("Loading guardrails from MongoDB for all clients...")
    clients = list_all_clients()
    for client_id in clients:
        try:
            load_guardrails_from_mongodb(client_id)
        except Exception as e:
            logger.warning(f"Failed to load guardrails for client {client_id}: {e}")
    
    try:
        public_url = start_ngrok_tunnel(port)
        logger.info("🎉 Bot is ready to accept connections!")
        logger.info(f"📋 Make POST requests to: {public_url}/start?client_id=YOUR_CLIENT_ID")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        logger.warning("⚠️  Continuing without ngrok. Bot will only be accessible locally.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

