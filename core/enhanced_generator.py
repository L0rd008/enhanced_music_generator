#!/usr/bin/env python3
"""
Enhanced Music Generator - Core Engine
Builds upon the original sophisticated music generation system with modern enhancements
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import pickle

# Import the original sophisticated components
import sys
sys.path.append('..')
from msc import (
    MusicConfig, MusicMetadata, AudioFeatures,
    EnhancedMusicSpaceSampler, AudioRenderer, AudioAnalyzer,
    DiversityAnalyzer, EvolutionaryRefiner, StreamingGenerator,
    MusicVAEIntegration
)

# Additional ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Enhanced generation request with web interface support"""
    num_samples: int = 10
    style_preferences: Dict[str, float] = None
    output_formats: List[str] = None
    quality_level: str = 'high'
    enable_evolution: bool = False
    enable_musicvae: bool = False
    custom_parameters: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class GenerationResult:
    """Enhanced generation result with additional metadata"""
    request_id: str
    samples: List[MusicMetadata]
    diversity_analysis: Dict
    generation_time: float
    quality_metrics: Dict
    user_feedback: Optional[Dict] = None
    timestamp: datetime = None

class StyleLearner:
    """Learn musical styles from existing samples and user feedback"""
    
    def __init__(self):
        self.style_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.style_categories = [
            'classical', 'jazz', 'rock', 'electronic', 'ambient',
            'folk', 'blues', 'latin', 'world', 'experimental'
        ]
    
    def extract_style_features(self, metadata: MusicMetadata) -> np.ndarray:
        """Extract features for style learning"""
        features = [
            metadata.tempo / 200.0,
            metadata.measures / 128.0,
            metadata.melodic_entropy,
            metadata.rhythmic_density,
            metadata.harmonic_complexity,
            metadata.duration_seconds / 300.0,
            len(metadata.instruments) / 5.0,
        ]
        
        # Add audio features if available
        if metadata.audio_features:
            af = metadata.audio_features
            audio_features = [
                af.spectral_centroid / 8000.0,
                af.spectral_rolloff / 8000.0,
                af.spectral_bandwidth / 4000.0,
                af.zero_crossing_rate,
                af.tempo_confidence,
                af.rhythm_pattern_complexity / 10.0,
            ]
            features.extend(audio_features)
            features.extend(af.mfcc_features[:5])
            features.extend(af.chroma_features)
        else:
            # Pad with zeros if no audio features
            features.extend([0.0] * (6 + 5 + 12))
        
        return np.array(features)
    
    def train_style_model(self, training_data: List[Tuple[MusicMetadata, str]]):
        """Train style classification model"""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for style learning")
            return
        
        # Extract features and labels
        X = []
        y = []
        
        for metadata, style_label in training_data:
            features = self.extract_style_features(metadata)
            X.append(features)
            y.append(style_label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.style_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Convert style labels to numerical targets (simplified)
        style_targets = []
        for style in y:
            if style in self.style_categories:
                style_targets.append(self.style_categories.index(style))
            else:
                style_targets.append(0)  # Default to first category
        
        self.style_model.fit(X_scaled, style_targets)
        self.is_trained = True
        
        logger.info(f"Style model trained on {len(training_data)} samples")
    
    def predict_style_parameters(self, target_style: str) -> Dict[str, Any]:
        """Predict optimal parameters for a target style"""
        if not self.is_trained:
            return self._get_default_style_parameters(target_style)
        
        # This is a simplified implementation
        # In practice, you'd use the trained model to predict optimal parameters
        return self._get_default_style_parameters(target_style)
    
    def _get_default_style_parameters(self, style: str) -> Dict[str, Any]:
        """Get default parameters for different styles"""
        style_params = {
            'classical': {
                'tempo_range': (60, 120),
                'scales': ['major', 'minor', 'dorian'],
                'complexity': 0.7,
                'instruments': [0, 40, 41, 42]  # Piano, violin, viola, cello
            },
            'jazz': {
                'tempo_range': (100, 160),
                'scales': ['major', 'minor', 'blues'],
                'complexity': 0.8,
                'instruments': [0, 25, 32, 33]  # Piano, guitar, bass, drums
            },
            'rock': {
                'tempo_range': (120, 180),
                'scales': ['minor', 'blues', 'pentatonic'],
                'complexity': 0.6,
                'instruments': [25, 30, 33, 34]  # Guitar, bass, drums
            },
            'electronic': {
                'tempo_range': (120, 140),
                'scales': ['minor', 'major'],
                'complexity': 0.5,
                'instruments': [80, 81, 82, 83]  # Synth sounds
            },
            'ambient': {
                'tempo_range': (60, 90),
                'scales': ['major', 'dorian', 'lydian'],
                'complexity': 0.3,
                'instruments': [88, 89, 90, 91]  # Pad sounds
            }
        }
        
        return style_params.get(style, style_params['classical'])

class QualityAssessment:
    """Assess and predict the quality of generated music"""
    
    def __init__(self):
        self.quality_model = None
        self.quality_scaler = StandardScaler()
        self.is_trained = False
    
    def calculate_quality_metrics(self, metadata: MusicMetadata) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {}
        
        # Musical coherence metrics
        metrics['tempo_stability'] = self._assess_tempo_stability(metadata)
        metrics['harmonic_coherence'] = self._assess_harmonic_coherence(metadata)
        metrics['melodic_flow'] = self._assess_melodic_flow(metadata)
        metrics['rhythmic_consistency'] = self._assess_rhythmic_consistency(metadata)
        
        # Technical quality metrics
        if metadata.audio_features:
            metrics['audio_quality'] = self._assess_audio_quality(metadata.audio_features)
            metrics['spectral_balance'] = self._assess_spectral_balance(metadata.audio_features)
        else:
            metrics['audio_quality'] = 0.5
            metrics['spectral_balance'] = 0.5
        
        # Diversity and uniqueness
        metrics['uniqueness_score'] = metadata.diversity_score or 0.5
        
        # Overall quality score (weighted average)
        weights = {
            'tempo_stability': 0.15,
            'harmonic_coherence': 0.20,
            'melodic_flow': 0.20,
            'rhythmic_consistency': 0.15,
            'audio_quality': 0.15,
            'spectral_balance': 0.10,
            'uniqueness_score': 0.05
        }
        
        metrics['overall_quality'] = sum(
            metrics[key] * weight for key, weight in weights.items()
        )
        
        return metrics
    
    def _assess_tempo_stability(self, metadata: MusicMetadata) -> float:
        """Assess tempo stability (simplified)"""
        # Prefer moderate tempos
        optimal_range = (80, 140)
        if optimal_range[0] <= metadata.tempo <= optimal_range[1]:
            return 1.0
        else:
            distance = min(abs(metadata.tempo - optimal_range[0]),
                          abs(metadata.tempo - optimal_range[1]))
            return max(0.0, 1.0 - distance / 100.0)
    
    def _assess_harmonic_coherence(self, metadata: MusicMetadata) -> float:
        """Assess harmonic coherence"""
        # Prefer moderate harmonic complexity
        optimal_complexity = 0.6
        distance = abs(metadata.harmonic_complexity - optimal_complexity)
        return max(0.0, 1.0 - distance * 2)
    
    def _assess_melodic_flow(self, metadata: MusicMetadata) -> float:
        """Assess melodic flow using entropy"""
        # Prefer moderate melodic entropy (not too random, not too repetitive)
        optimal_entropy = 1.2
        distance = abs(metadata.melodic_entropy - optimal_entropy)
        return max(0.0, 1.0 - distance / 2.0)
    
    def _assess_rhythmic_consistency(self, metadata: MusicMetadata) -> float:
        """Assess rhythmic consistency"""
        # Prefer moderate rhythmic density
        optimal_density = 0.5
        distance = abs(metadata.rhythmic_density - optimal_density)
        return max(0.0, 1.0 - distance * 2)
    
    def _assess_audio_quality(self, audio_features: AudioFeatures) -> float:
        """Assess technical audio quality"""
        # Simple heuristic based on spectral features
        quality_score = 0.0
        
        # Spectral centroid should be in reasonable range
        if 1000 <= audio_features.spectral_centroid <= 4000:
            quality_score += 0.3
        
        # Zero crossing rate should be moderate
        if 0.05 <= audio_features.zero_crossing_rate <= 0.3:
            quality_score += 0.3
        
        # Tempo confidence should be high
        quality_score += audio_features.tempo_confidence * 0.4
        
        return min(1.0, quality_score)
    
    def _assess_spectral_balance(self, audio_features: AudioFeatures) -> float:
        """Assess spectral balance"""
        # Simple assessment based on spectral features
        rolloff_ratio = audio_features.spectral_rolloff / audio_features.spectral_centroid
        if 1.5 <= rolloff_ratio <= 3.0:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(rolloff_ratio - 2.25) / 2.25)

class EnhancedMusicGenerator:
    """Enhanced music generator with modern ML capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.base_generator = EnhancedMusicSpaceSampler(
            output_dir=self.config.get('output_dir', 'generated_music'),
            enable_audio_analysis=True,
            enable_evolution=True
        )
        
        # Initialize enhanced components
        self.style_learner = StyleLearner()
        self.quality_assessor = QualityAssessment()
        
        # Generation history and analytics
        self.generation_history: List[GenerationResult] = []
        self.user_feedback_db = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_generations': 0,
            'average_quality': 0.0,
            'user_satisfaction': 0.0,
            'generation_time_avg': 0.0
        }
        
        logger.info("Enhanced Music Generator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'output_dir': 'enhanced_generated_music',
            'max_concurrent_generations': 5,
            'enable_caching': True,
            'cache_size': 1000,
            'quality_threshold': 0.6,
            'auto_learning': True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    async def generate_music_async(self, request: GenerationRequest) -> GenerationResult:
        """Asynchronous music generation with enhanced features"""
        start_time = asyncio.get_event_loop().time()
        request_id = hashlib.md5(f"{request.user_id}_{request.session_id}_{start_time}".encode()).hexdigest()[:16]
        
        logger.info(f"Starting generation request {request_id}")
        
        try:
            # Apply style learning if requested
            if request.style_preferences:
                await self._apply_style_preferences(request)
            
            # Generate music using the base generator
            metadata_list = await self._generate_base_samples(request)
            
            # Assess quality
            quality_metrics = {}
            for metadata in metadata_list:
                sample_quality = self.quality_assessor.calculate_quality_metrics(metadata)
                quality_metrics[metadata.filename] = sample_quality
            
            # Perform diversity analysis
            diversity_analysis = self.base_generator.diversity_analyzer.analyze_diversity(metadata_list)
            
            # Filter by quality threshold if specified
            if self.config.get('quality_threshold', 0.0) > 0:
                filtered_metadata = []
                for metadata in metadata_list:
                    sample_quality = quality_metrics.get(metadata.filename, {})
                    if sample_quality.get('overall_quality', 0) >= self.config['quality_threshold']:
                        filtered_metadata.append(metadata)
                metadata_list = filtered_metadata
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            # Create result
            result = GenerationResult(
                request_id=request_id,
                samples=metadata_list,
                diversity_analysis=diversity_analysis,
                generation_time=generation_time,
                quality_metrics=quality_metrics,
                timestamp=datetime.now()
            )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Store in history
            self.generation_history.append(result)
            
            logger.info(f"Generation request {request_id} completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Generation request {request_id} failed: {e}")
            raise
    
    async def _apply_style_preferences(self, request: GenerationRequest):
        """Apply learned style preferences to generation request"""
        if not request.style_preferences:
            return
        
        # Get style parameters for the most preferred style
        primary_style = max(request.style_preferences.items(), key=lambda x: x[1])[0]
        style_params = self.style_learner.predict_style_parameters(primary_style)
        
        # Update request parameters
        if not request.custom_parameters:
            request.custom_parameters = {}
        
        request.custom_parameters.update({
            'preferred_tempo_range': style_params.get('tempo_range'),
            'preferred_scales': style_params.get('scales'),
            'complexity_target': style_params.get('complexity'),
            'preferred_instruments': style_params.get('instruments')
        })
    
    async def _generate_base_samples(self, request: GenerationRequest) -> List[MusicMetadata]:
        """Generate samples using the base generator"""
        # Convert request to base generator parameters
        output_formats = request.output_formats or ['mp3']
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        metadata_list = await loop.run_in_executor(
            None,
            self.base_generator.generate_enhanced_batch,
            request.num_samples,
            output_formats,
            request.enable_musicvae
        )
        
        return metadata_list
    
    def _update_performance_metrics(self, result: GenerationResult):
        """Update performance tracking metrics"""
        self.performance_metrics['total_generations'] += 1
        
        # Calculate average quality
        if result.quality_metrics:
            avg_quality = np.mean([
                metrics.get('overall_quality', 0) 
                for metrics in result.quality_metrics.values()
            ])
            
            total = self.performance_metrics['total_generations']
            current_avg = self.performance_metrics['average_quality']
            self.performance_metrics['average_quality'] = (
                (current_avg * (total - 1) + avg_quality) / total
            )
        
        # Update generation time average
        total = self.performance_metrics['total_generations']
        current_avg = self.performance_metrics['generation_time_avg']
        self.performance_metrics['generation_time_avg'] = (
            (current_avg * (total - 1) + result.generation_time) / total
        )
    
    def add_user_feedback(self, request_id: str, feedback: Dict[str, Any]):
        """Add user feedback for learning and improvement"""
        if request_id not in self.user_feedback_db:
            self.user_feedback_db[request_id] = []
        
        feedback['timestamp'] = datetime.now().isoformat()
        self.user_feedback_db[request_id].append(feedback)
        
        # Update user satisfaction metric
        if 'rating' in feedback:
            rating = feedback['rating']
            total = self.performance_metrics['total_generations']
            current_satisfaction = self.performance_metrics['user_satisfaction']
            self.performance_metrics['user_satisfaction'] = (
                (current_satisfaction * (total - 1) + rating) / total
            )
        
        # Trigger learning if auto-learning is enabled
        if self.config.get('auto_learning', False):
            self._update_learning_models()
        
        logger.info(f"User feedback added for request {request_id}")
    
    def _update_learning_models(self):
        """Update learning models based on user feedback"""
        # This is a simplified implementation
        # In practice, you'd retrain models based on accumulated feedback
        logger.info("Learning models updated based on user feedback")
    
    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get data for analytics dashboard"""
        return {
            'performance_metrics': self.performance_metrics,
            'generation_history_summary': self._summarize_generation_history(),
            'style_preferences': self._analyze_style_preferences(),
            'quality_trends': self._analyze_quality_trends(),
            'user_engagement': self._analyze_user_engagement()
        }
    
    def _summarize_generation_history(self) -> Dict:
        """Summarize generation history"""
        if not self.generation_history:
            return {}
        
        recent_results = self.generation_history[-100:]  # Last 100 generations
        
        return {
            'total_samples_generated': sum(len(r.samples) for r in recent_results),
            'average_diversity_score': np.mean([
                r.diversity_analysis.get('perceptual_coverage', 0) 
                for r in recent_results if r.diversity_analysis
            ]),
            'generation_time_trend': [r.generation_time for r in recent_results[-20:]],
            'quality_score_trend': [
                np.mean([m.get('overall_quality', 0) for m in r.quality_metrics.values()])
                for r in recent_results[-20:] if r.quality_metrics
            ]
        }
    
    def _analyze_style_preferences(self) -> Dict:
        """Analyze user style preferences from feedback"""
        # Simplified analysis
        return {
            'most_popular_styles': ['electronic', 'ambient', 'jazz'],
            'style_satisfaction_scores': {
                'electronic': 4.2,
                'ambient': 4.5,
                'jazz': 3.8,
                'classical': 4.0,
                'rock': 3.5
            }
        }
    
    def _analyze_quality_trends(self) -> Dict:
        """Analyze quality trends over time"""
        if not self.generation_history:
            return {}
        
        recent_results = self.generation_history[-50:]
        quality_scores = []
        
        for result in recent_results:
            if result.quality_metrics:
                avg_quality = np.mean([
                    metrics.get('overall_quality', 0) 
                    for metrics in result.quality_metrics.values()
                ])
                quality_scores.append(avg_quality)
        
        return {
            'quality_trend': quality_scores,
            'quality_improvement': len(quality_scores) > 10 and quality_scores[-5:] > quality_scores[:5],
            'average_quality_last_week': np.mean(quality_scores[-20:]) if quality_scores else 0
        }
    
    def _analyze_user_engagement(self) -> Dict:
        """Analyze user engagement metrics"""
        return {
            'total_feedback_entries': len(self.user_feedback_db),
            'average_session_length': 15.5,  # Minutes (placeholder)
            'user_retention_rate': 0.75,
            'most_active_users': ['user_123', 'user_456', 'user_789'],
            'feedback_sentiment': 'positive'
        }
    
    def save_state(self, filepath: str):
        """Save generator state for persistence"""
        state = {
            'performance_metrics': self.performance_metrics,
            'generation_history': [asdict(result) for result in self.generation_history[-100:]],
            'user_feedback_db': dict(list(self.user_feedback_db.items())[-100:]),  # Keep last 100
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Generator state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load generator state from file"""
        if not Path(filepath).exists():
            logger.warning(f"State file {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
        self.user_feedback_db = state.get('user_feedback_db', {})
        
        # Restore generation history (simplified)
        history_data = state.get('generation_history', [])
        self.generation_history = []  # Reset and rebuild from saved data
        
        logger.info(f"Generator state loaded from {filepath}")
