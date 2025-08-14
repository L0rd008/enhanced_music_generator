#!/usr/bin/env python3
"""
Web API for Enhanced Music Generator
FastAPI-based REST API with WebSocket support for real-time generation
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .enhanced_generator import (
    EnhancedMusicGenerator, GenerationRequest, GenerationResult
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class GenerationRequestAPI(BaseModel):
    num_samples: int = Field(default=5, ge=1, le=50, description="Number of samples to generate")
    style_preferences: Optional[Dict[str, float]] = Field(default=None, description="Style preferences with weights")
    output_formats: Optional[List[str]] = Field(default=['mp3'], description="Output audio formats")
    quality_level: str = Field(default='high', description="Audio quality level")
    enable_evolution: bool = Field(default=False, description="Enable evolutionary refinement")
    enable_musicvae: bool = Field(default=False, description="Enable MusicVAE generation")
    custom_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Custom generation parameters")

class FeedbackRequest(BaseModel):
    request_id: str = Field(description="Generation request ID")
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    comments: Optional[str] = Field(default=None, description="User comments")
    favorite_samples: Optional[List[str]] = Field(default=None, description="List of favorite sample filenames")
    style_feedback: Optional[Dict[str, float]] = Field(default=None, description="Style-specific feedback")

class StreamingParameterUpdate(BaseModel):
    tempo: Optional[int] = Field(default=None, ge=60, le=200)
    key: Optional[int] = Field(default=None, ge=0, le=11)
    scale: Optional[str] = Field(default=None)
    melodic_density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    harmonic_complexity: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class MusicGeneratorAPI:
    """FastAPI application for music generation"""
    
    def __init__(self, generator: EnhancedMusicGenerator):
        self.generator = generator
        self.app = FastAPI(
            title="AI Music Generation API",
            description="Advanced AI-powered music generation with real-time capabilities",
            version="2.0.0"
        )
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        self.streaming_sessions: Dict[str, Dict] = {}
        
        # Setup middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Mount static files
        static_path = Path(__file__).parent.parent / "web" / "static"
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main web interface"""
            html_path = Path(__file__).parent.parent / "web" / "index.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("""
            <html>
                <head><title>AI Music Generator</title></head>
                <body>
                    <h1>🎵 AI Music Generation Suite</h1>
                    <p>Advanced music generation system with ML capabilities</p>
                    <p>API Documentation: <a href="/docs">/docs</a></p>
                </body>
            </html>
            """)
        
        @self.app.post("/api/generate")
        async def generate_music(
            request: GenerationRequestAPI,
            background_tasks: BackgroundTasks,
            user_id: Optional[str] = None
        ):
            """Generate music samples"""
            try:
                # Create internal request
                session_id = str(uuid.uuid4())
                internal_request = GenerationRequest(
                    num_samples=request.num_samples,
                    style_preferences=request.style_preferences,
                    output_formats=request.output_formats,
                    quality_level=request.quality_level,
                    enable_evolution=request.enable_evolution,
                    enable_musicvae=request.enable_musicvae,
                    custom_parameters=request.custom_parameters,
                    user_id=user_id or "anonymous",
                    session_id=session_id
                )
                
                # Generate music
                result = await self.generator.generate_music_async(internal_request)
                
                # Notify WebSocket clients
                background_tasks.add_task(self._notify_generation_complete, result)
                
                # Return result
                return {
                    "request_id": result.request_id,
                    "session_id": session_id,
                    "samples": [
                        {
                            "filename": sample.filename,
                            "metadata": {
                                "tempo": sample.tempo,
                                "key": sample.key,
                                "scale": sample.scale,
                                "duration": sample.duration_seconds,
                                "quality_score": result.quality_metrics.get(sample.filename, {}).get('overall_quality', 0)
                            }
                        }
                        for sample in result.samples
                    ],
                    "generation_time": result.generation_time,
                    "diversity_analysis": result.diversity_analysis
                }
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/feedback")
        async def submit_feedback(feedback: FeedbackRequest):
            """Submit user feedback for a generation"""
            try:
                self.generator.add_user_feedback(
                    feedback.request_id,
                    {
                        "rating": feedback.rating,
                        "comments": feedback.comments,
                        "favorite_samples": feedback.favorite_samples,
                        "style_feedback": feedback.style_feedback
                    }
                )
                return {"status": "success", "message": "Feedback recorded"}
            except Exception as e:
                logger.error(f"Feedback submission failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/analytics")
        async def get_analytics():
            """Get analytics dashboard data"""
            try:
                data = self.generator.get_analytics_dashboard_data()
                return data
            except Exception as e:
                logger.error(f"Analytics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/download/{filename}")
        async def download_file(filename: str):
            """Download generated audio file"""
            try:
                # Security: validate filename and path
                if not filename.replace('.', '').replace('_', '').replace('-', '').isalnum():
                    raise HTTPException(status_code=400, detail="Invalid filename")
                
                file_path = Path(self.generator.config['output_dir']) / filename
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                return FileResponse(
                    path=str(file_path),
                    filename=filename,
                    media_type='audio/mpeg' if filename.endswith('.mp3') else 'audio/wav'
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"File download failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/styles")
        async def get_available_styles():
            """Get available music styles"""
            return {
                "styles": self.generator.style_learner.style_categories,
                "style_parameters": {
                    style: self.generator.style_learner._get_default_style_parameters(style)
                    for style in self.generator.style_learner.style_categories
                }
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
        
        @self.app.websocket("/ws/streaming/{session_id}")
        async def streaming_websocket(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for streaming generation"""
            await self._handle_streaming_websocket(websocket, session_id)
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle general WebSocket connections"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "get_status":
                    status = {
                        "type": "status",
                        "active_generations": len(self.streaming_sessions),
                        "total_generations": self.generator.performance_metrics["total_generations"]
                    }
                    await websocket.send_text(json.dumps(status))
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _handle_streaming_websocket(self, websocket: WebSocket, session_id: str):
        """Handle streaming generation WebSocket"""
        await websocket.accept()
        
        # Initialize streaming session
        self.streaming_sessions[session_id] = {
            "websocket": websocket,
            "active": True,
            "parameters": {}
        }
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "start_streaming":
                    await self._start_streaming_generation(session_id, message.get("config", {}))
                elif message.get("type") == "update_parameters":
                    await self._update_streaming_parameters(session_id, message.get("parameters", {}))
                elif message.get("type") == "stop_streaming":
                    await self._stop_streaming_generation(session_id)
                    break
                    
        except Exception as e:
            logger.error(f"Streaming WebSocket error: {e}")
        finally:
            if session_id in self.streaming_sessions:
                del self.streaming_sessions[session_id]
    
    async def _start_streaming_generation(self, session_id: str, config: Dict):
        """Start streaming music generation"""
        try:
            # This is a simplified implementation
            # In practice, you'd integrate with the streaming generator from the original system
            session = self.streaming_sessions.get(session_id)
            if not session:
                return
            
            websocket = session["websocket"]
            
            # Send confirmation
            await websocket.send_text(json.dumps({
                "type": "streaming_started",
                "session_id": session_id,
                "message": "Streaming generation started"
            }))
            
            # Simulate streaming updates (replace with actual streaming logic)
            for i in range(10):
                if session_id not in self.streaming_sessions:
                    break
                
                await asyncio.sleep(2)  # Simulate generation time
                
                # Send progress update
                await websocket.send_text(json.dumps({
                    "type": "generation_progress",
                    "progress": (i + 1) * 10,
                    "current_parameters": session.get("parameters", {})
                }))
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
    
    async def _update_streaming_parameters(self, session_id: str, parameters: Dict):
        """Update streaming generation parameters"""
        session = self.streaming_sessions.get(session_id)
        if session:
            session["parameters"].update(parameters)
            
            # Send confirmation
            await session["websocket"].send_text(json.dumps({
                "type": "parameters_updated",
                "parameters": session["parameters"]
            }))
    
    async def _stop_streaming_generation(self, session_id: str):
        """Stop streaming generation"""
        session = self.streaming_sessions.get(session_id)
        if session:
            session["active"] = False
            
            await session["websocket"].send_text(json.dumps({
                "type": "streaming_stopped",
                "session_id": session_id
            }))
    
    async def _notify_generation_complete(self, result: GenerationResult):
        """Notify WebSocket clients of completed generation"""
        notification = {
            "type": "generation_complete",
            "request_id": result.request_id,
            "samples_count": len(result.samples),
            "generation_time": result.generation_time
        }
        
        # Send to all connected clients
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(notification))
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections.remove(websocket)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)

def create_api_app(generator: EnhancedMusicGenerator) -> FastAPI:
    """Create and configure the FastAPI application"""
    api = MusicGeneratorAPI(generator)
    return api.app

if __name__ == "__main__":
    # For testing
    from enhanced_generator import EnhancedMusicGenerator
    
    generator = EnhancedMusicGenerator()
    api = MusicGeneratorAPI(generator)
    api.run(host="127.0.0.1", port=8000, reload=True)
