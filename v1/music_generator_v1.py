#!/usr/bin/env python3
"""
Enhanced Intelligent Music Space Sampler
Generates perceptually unique, rhythmic, and musically valid audio files
with advanced diversity metrics, audio rendering, and real-time capabilities.
"""
import subprocess
import math
import numpy as np
import mido
import random
import json
import os
import shlex
import threading
import queue
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
import hashlib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import librosa
import soundfile as sf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# Try to import optional dependencies
try:
    import tensorflow as tf
    from magenta.models.music_vae import configs
    from magenta.models.music_vae.trained_model import TrainedModel
    MAGENTA_AVAILABLE = True
except ImportError:
    MAGENTA_AVAILABLE = False
    print("Warning: Magenta not available. MusicVAE features disabled.")

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False
    print("Warning: FluidSynth Python bindings not available. Using subprocess fallback.")

# Musical constants and scales
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'pentatonic': [0, 2, 4, 7, 9],
    'blues': [0, 3, 5, 6, 7, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11]
}

CHORD_PROGRESSIONS = {
    'major': [
        [0, 3, 5, 0],  # I-vi-IV-I
        [0, 5, 3, 4],  # I-V-vi-IV
        [0, 4, 5, 0],  # I-IV-V-I
        [0, 2, 3, 0],  # I-ii-iii-I
        [0, 4, 0, 5],  # I-IV-I-V
        [3, 5, 0, 4],  # vi-V-I-IV
    ],
    'minor': [
        [0, 3, 4, 0],  # i-III-iv-i
        [0, 6, 3, 0],  # i-VII-III-i
        [0, 4, 5, 0],  # i-iv-V-i
        [0, 2, 5, 0],  # i-ii-V-i
        [4, 0, 5, 3],  # iv-i-V-III
    ]
}

# Enhanced drum patterns with variations
DRUM_PATTERNS = {
    'basic_rock': [
        ([36], [42], [36], [42]),
        ([36, 42], [], [36], [42, 46]),
        ([36], [42, 46], [36, 46], [42]),
    ],
    'simple_pop': [
        ([36], [42], [36], [42, 46]),
        ([36], [42], [36, 42], [42]),
        ([36, 46], [42, 46], [36, 46], [42, 46]),
    ],
    'latin': [
        ([36], [42, 46], [], [42]),
        ([36, 46], [42], [46], [42, 46]),
    ],
    'shuffle': [
        ([36], [42], [36, 46], [42]),
        ([36, 46], [], [36], [42, 46]),
    ],
    'complex': [
        ([36], [42], [36, 46], [42, 46]),
        ([36, 42], [46], [36], [42, 46]),
        ([36], [42, 46], [36, 46], [42]),
    ]
}

@dataclass
class AudioFeatures:
    """Audio features extracted from generated audio"""
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    mfcc_features: List[float]
    chroma_features: List[float]
    tonnetz_features: List[float]
    tempo_confidence: float
    rhythm_pattern_complexity: float

@dataclass
class MusicConfig:
    """Enhanced configuration for a single musical piece"""
    key: int  # 0-11 (C, C#, D, ...)
    scale: str
    tempo: int  # BPM
    time_signature: Tuple[int, int]
    measures: int
    instruments: List[int]  # MIDI program numbers
    chord_progression: List[int]
    drum_pattern: str
    melodic_density: float  # 0-1, how busy the melody is
    harmonic_complexity: float  # 0-1, chord richness
    swing_factor: float = 0.0  # 0-1, rhythmic swing
    velocity_variation: float = 0.3  # 0-1, dynamic range
    articulation_style: str = 'normal'  # normal, staccato, legato
    modulation_intensity: float = 0.0  # 0-1, pitch bend/vibrato
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.ticks_per_beat = 480
        self.beats_per_measure = self.time_signature[0]
        self.ticks_per_measure = self.ticks_per_beat * self.beats_per_measure

@dataclass
class MusicMetadata:
    """Enhanced metadata for generated music"""
    filename: str
    tempo: int
    key: str
    scale: str
    time_signature: str
    instruments: List[str]
    measures: int
    melodic_entropy: float
    rhythmic_density: float
    harmonic_complexity: float
    duration_seconds: float
    embedding_hash: str
    audio_features: Optional[AudioFeatures] = None
    perceptual_cluster: Optional[int] = None
    diversity_score: Optional[float] = None
    generation_method: str = 'rule_based'

class AudioRenderer:
    """Enhanced audio rendering with multiple backends"""
    
    def __init__(self, soundfont_path: str = "GeneralUser-GS.sf2", 
                 sample_rate: int = 44100, quality: str = 'high'):
        self.soundfont_path = Path(soundfont_path)
        self.sample_rate = sample_rate
        self.quality = quality
        self.use_fluidsynth_api = FLUIDSYNTH_AVAILABLE
        
        # Audio processing settings
        self.compression_ratio = 4.0
        self.normalization_target = -12.0  # dB
        
    def render_midi_to_audio(self, midi_path: Path, output_format: str = 'mp3') -> Path:
        """Render MIDI to audio with post-processing"""
        if self.use_fluidsynth_api:
            return self._render_with_fluidsynth_api(midi_path, output_format)
        else:
            return self._render_with_subprocess(midi_path, output_format)
    
    def _render_with_fluidsynth_api(self, midi_path: Path, output_format: str) -> Path:
        """Render using FluidSynth Python API for better control"""
        try:
            # Initialize FluidSynth
            fs = fluidsynth.Synth()
            fs.start(driver='file')
            
            # Load soundfont
            sfid = fs.sfload(str(self.soundfont_path))
            
            # Load and play MIDI
            mid = mido.MidiFile(str(midi_path))
            audio_data = []
            
            for msg in mid.play():
                if msg.type == 'note_on':
                    fs.noteon(msg.channel, msg.note, msg.velocity)
                elif msg.type == 'note_off':
                    fs.noteoff(msg.channel, msg.note)
                elif msg.type == 'program_change':
                    fs.program_select(msg.channel, sfid, 0, msg.program)
                
                # Generate audio samples
                samples = fs.get_samples(int(self.sample_rate * msg.time / 1000))
                audio_data.extend(samples)
            
            fs.delete()
            
            # Convert to numpy array and apply post-processing
            audio_array = np.array(audio_data, dtype=np.float32)
            audio_array = self._apply_post_processing(audio_array)
            
            # Save to file
            output_path = midi_path.with_suffix(f'.{output_format}')
            if output_format == 'wav':
                sf.write(str(output_path), audio_array, self.sample_rate)
            else:
                # Save as WAV first, then convert
                temp_wav = midi_path.with_suffix('.temp.wav')
                sf.write(str(temp_wav), audio_array, self.sample_rate)
                self._convert_audio_format(temp_wav, output_path, output_format)
                temp_wav.unlink()
            
            return output_path
            
        except Exception as e:
            print(f"FluidSynth API rendering failed: {e}. Falling back to subprocess.")
            return self._render_with_subprocess(midi_path, output_format)
    
    def _render_with_subprocess(self, midi_path: Path, output_format: str) -> Path:
        """Fallback rendering using subprocess (original method)"""
        output_dir = midi_path.parent
        soundfont_abs_path = Path(__file__).parent / self.soundfont_path
        
        if not soundfont_abs_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {soundfont_abs_path}")

        wav_path = midi_path.with_suffix(".wav")
        output_path = midi_path.with_suffix(f".{output_format}")

        # FluidSynth command with quality settings
        quality_args = []
        if self.quality == 'high':
            quality_args = ["-g", "1.5", "-R", str(self.sample_rate)]
        elif self.quality == 'low':
            quality_args = ["-g", "1.0", "-R", "22050"]
        else:
            quality_args = ["-g", "1.2", "-R", str(self.sample_rate)]

        command_fluidsynth = [
            "fluidsynth", "-ni", *quality_args,
            str(soundfont_abs_path), midi_path.name,
            "-F", wav_path.name, "-q"
        ]
        
        try:
            subprocess.run(command_fluidsynth, check=True, cwd=output_dir, 
                          capture_output=True, text=True)
            
            # Apply post-processing
            audio_data, sr = librosa.load(str(wav_path), sr=self.sample_rate)
            processed_audio = self._apply_post_processing(audio_data)
            
            if output_format == 'wav':
                sf.write(str(wav_path), processed_audio, self.sample_rate)
                return wav_path
            else:
                sf.write(str(wav_path), processed_audio, self.sample_rate)
                self._convert_audio_format(wav_path, output_path, output_format)
                wav_path.unlink()
                return output_path
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio rendering failed: {e}")
    
    def _apply_post_processing(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply normalization and compression"""
        # Normalization
        if np.max(np.abs(audio_data)) > 0:
            target_amplitude = 10**(self.normalization_target / 20)
            current_peak = np.max(np.abs(audio_data))
            audio_data = audio_data * (target_amplitude / current_peak)
        
        # Simple compression (limiting)
        threshold = 0.8
        audio_data = np.where(np.abs(audio_data) > threshold,
                             np.sign(audio_data) * threshold, audio_data)
        
        return audio_data
    
    def _convert_audio_format(self, input_path: Path, output_path: Path, format: str):
        """Convert audio using FFmpeg"""
        bitrate_map = {'high': '320k', 'medium': '192k', 'low': '128k'}
        bitrate = bitrate_map.get(self.quality, '192k')
        
        command = [
            "ffmpeg", "-i", str(input_path), "-acodec", "libmp3lame",
            "-b:a", bitrate, "-y", str(output_path),
            "-hide_banner", "-loglevel", "error"
        ]
        
        subprocess.run(command, check=True)

class AudioAnalyzer:
    """Advanced audio feature extraction and analysis"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_path: Path) -> AudioFeatures:
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Temporal features
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Timbral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Harmonic features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Harmonic content
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            
            # Tempo analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_confidence = len(beats) / (len(y) / sr) if len(y) > 0 else 0
            
            # Rhythm complexity (onset density)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            rhythm_complexity = len(onset_frames) / (len(y) / sr) if len(y) > 0 else 0
            
            return AudioFeatures(
                spectral_centroid=np.mean(spectral_centroids),
                spectral_rolloff=np.mean(spectral_rolloff),
                spectral_bandwidth=np.mean(spectral_bandwidth),
                zero_crossing_rate=np.mean(zcr),
                mfcc_features=mfcc_mean.tolist(),
                chroma_features=chroma_mean.tolist(),
                tonnetz_features=tonnetz_mean.tolist(),
                tempo_confidence=tempo_confidence,
                rhythm_pattern_complexity=rhythm_complexity
            )
            
        except Exception as e:
            print(f"Feature extraction failed for {audio_path}: {e}")
            return self._default_features()
    
    def _default_features(self) -> AudioFeatures:
        """Return default features if extraction fails"""
        return AudioFeatures(
            spectral_centroid=0.0, spectral_rolloff=0.0, spectral_bandwidth=0.0,
            zero_crossing_rate=0.0, mfcc_features=[0.0] * 13,
            chroma_features=[0.0] * 12, tonnetz_features=[0.0] * 6,
            tempo_confidence=0.0, rhythm_pattern_complexity=0.0
        )

class DiversityAnalyzer:
    """Advanced diversity analysis with clustering and perceptual metrics"""
    
    def __init__(self, n_clusters: int = 20):
        self.n_clusters = n_clusters
        self.feature_weights = {
            'spectral': 0.3, 'temporal': 0.2, 'harmonic': 0.3, 'rhythmic': 0.2
        }
    
    def analyze_diversity(self, metadata_list: List[MusicMetadata]) -> Dict:
        """Perform comprehensive diversity analysis"""
        if not metadata_list:
            return {}
        
        # Extract feature vectors
        feature_vectors = self._extract_feature_vectors(metadata_list)
        
        # Perform clustering
        clusters = self._cluster_samples(feature_vectors)
        
        # Calculate perceptual distances
        distances = self._calculate_perceptual_distances(feature_vectors)
        
        # Update metadata with cluster assignments and diversity scores
        for i, metadata in enumerate(metadata_list):
            metadata.perceptual_cluster = int(clusters[i])
            metadata.diversity_score = self._calculate_diversity_score(
                feature_vectors[i], feature_vectors, distances[i]
            )
        
        return {
            'cluster_distribution': self._analyze_cluster_distribution(clusters),
            'perceptual_coverage': self._calculate_perceptual_coverage(distances),
            'feature_diversity': self._analyze_feature_diversity(feature_vectors),
            'redundancy_score': self._calculate_redundancy_score(distances)
        }
    
    def _extract_feature_vectors(self, metadata_list: List[MusicMetadata]) -> np.ndarray:
        """Extract feature vectors for diversity analysis"""
        vectors = []
        
        for metadata in metadata_list:
            # Structural features
            structural = [
                metadata.tempo / 200.0,  # Normalized tempo
                metadata.measures / 128.0,  # Normalized measures
                metadata.melodic_entropy,
                metadata.rhythmic_density,
                metadata.harmonic_complexity,
                metadata.duration_seconds / 300.0,  # Normalized duration
                len(metadata.instruments) / 5.0,  # Normalized instrument count
            ]
            
            # Audio features (if available)
            audio_features = []
            if metadata.audio_features:
                af = metadata.audio_features
                audio_features = [
                    af.spectral_centroid / 8000.0,  # Normalized
                    af.spectral_rolloff / 8000.0,
                    af.spectral_bandwidth / 4000.0,
                    af.zero_crossing_rate,
                    af.tempo_confidence,
                    af.rhythm_pattern_complexity / 10.0,
                ]
                # Add MFCC and chroma features (first few coefficients)
                audio_features.extend(af.mfcc_features[:5])
                audio_features.extend(af.chroma_features)
            else:
                # Use default values if audio features not available
                audio_features = [0.0] * (6 + 5 + 12)
            
            # Combine all features
            vector = structural + audio_features
            vectors.append(vector)
        
        return np.array(vectors)
    
    def _cluster_samples(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Cluster samples based on perceptual similarity"""
        if len(feature_vectors) < self.n_clusters:
            return np.arange(len(feature_vectors))
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(feature_vectors)
        return clusters
    
    def _calculate_perceptual_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Calculate pairwise perceptual distances"""
        return pairwise_distances(feature_vectors, metric='euclidean')
    
    def _calculate_diversity_score(self, vector: np.ndarray, all_vectors: np.ndarray,
                                  distances: np.ndarray) -> float:
        """Calculate diversity score for a single sample"""
        # Average distance to all other samples
        avg_distance = np.mean(distances)
        
        # Minimum distance (uniqueness)
        min_distance = np.min(distances[distances > 0]) if len(distances) > 1 else 1.0
        
        # Combine metrics
        diversity_score = 0.7 * avg_distance + 0.3 * min_distance
        return float(diversity_score)
    
    def _analyze_cluster_distribution(self, clusters: np.ndarray) -> Dict:
        """Analyze cluster distribution"""
        unique, counts = np.unique(clusters, return_counts=True)
        return {
            'n_clusters': len(unique),
            'cluster_sizes': counts.tolist(),
            'cluster_entropy': entropy(counts, base=2)
        }
    
    def _calculate_perceptual_coverage(self, distances: np.ndarray) -> float:
        """Calculate how well the samples cover the perceptual space"""
        # Use the mean of minimum distances as coverage metric
        min_distances = np.min(distances + np.eye(len(distances)) * 1e6, axis=1)
        coverage = np.mean(min_distances)
        return float(coverage)
    
    def _analyze_feature_diversity(self, feature_vectors: np.ndarray) -> Dict:
        """Analyze diversity in individual feature dimensions"""
        feature_stats = {}
        for i in range(feature_vectors.shape[1]):
            values = feature_vectors[:, i]
            feature_stats[f'feature_{i}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': float(np.max(values) - np.min(values))
            }
        return feature_stats
    
    def _calculate_redundancy_score(self, distances: np.ndarray) -> float:
        """Calculate redundancy score (lower is more diverse)"""
        # Count pairs below a similarity threshold
        threshold = np.percentile(distances[distances > 0], 10)  # Bottom 10%
        similar_pairs = np.sum(distances < threshold) - len(distances)  # Exclude diagonal
        total_pairs = len(distances) * (len(distances) - 1)
        redundancy = similar_pairs / total_pairs if total_pairs > 0 else 0
        return float(redundancy)

class EvolutionaryRefiner:
    """Evolutionary algorithm for refining generated music"""
    
    def __init__(self, population_size: int = 50, generations: int = 20):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def refine_population(self, configs: List[MusicConfig], 
                         fitness_func: callable) -> List[MusicConfig]:
        """Refine a population of configurations using evolutionary algorithms"""
        population = configs[:self.population_size]
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [fitness_func(config) for config in population]
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 if random.random() < 0.5 else parent2
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            if generation % 5 == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"Generation {generation}: Average fitness = {avg_fitness:.3f}")
        
        return population
    
    def _tournament_selection(self, population: List[MusicConfig], 
                            fitness_scores: List[float], tournament_size: int = 3) -> MusicConfig:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: MusicConfig, parent2: MusicConfig) -> MusicConfig:
        """Crossover two configurations"""
        # Create a new config by randomly selecting features from parents
        return MusicConfig(
            key=random.choice([parent1.key, parent2.key]),
            scale=random.choice([parent1.scale, parent2.scale]),
            tempo=int(np.mean([parent1.tempo, parent2.tempo])),
            time_signature=random.choice([parent1.time_signature, parent2.time_signature]),
            measures=random.choice([parent1.measures, parent2.measures]),
            instruments=random.choice([parent1.instruments, parent2.instruments]),
            chord_progression=random.choice([parent1.chord_progression, parent2.chord_progression]),
            drum_pattern=random.choice([parent1.drum_pattern, parent2.drum_pattern]),
            melodic_density=np.mean([parent1.melodic_density, parent2.melodic_density]),
            harmonic_complexity=np.mean([parent1.harmonic_complexity, parent2.harmonic_complexity]),
            swing_factor=np.mean([parent1.swing_factor, parent2.swing_factor]),
            velocity_variation=np.mean([parent1.velocity_variation, parent2.velocity_variation]),
            articulation_style=random.choice([parent1.articulation_style, parent2.articulation_style]),
            modulation_intensity=np.mean([parent1.modulation_intensity, parent2.modulation_intensity])
        )
    
    def _mutate(self, config: MusicConfig) -> MusicConfig:
        """Mutate a configuration"""
        mutation_strength = 0.1
        
        # Randomly mutate one or more parameters
        if random.random() < 0.3:
            config.key = (config.key + random.randint(-2, 2)) % 12
        
        if random.random() < 0.2:
            config.scale = random.choice(list(SCALES.keys()))
        
        if random.random() < 0.3:
            config.tempo += random.randint(-20, 20)
            config.tempo = max(60, min(200, config.tempo))
        
        if random.random() < 0.4:
            config.melodic_density += random.gauss(0, mutation_strength)
            config.melodic_density = max(0, min(1, config.melodic_density))
        
        if random.random() < 0.4:
            config.harmonic_complexity += random.gauss(0, mutation_strength)
            config.harmonic_complexity = max(0, min(1, config.harmonic_complexity))
        
        return config

class StreamingGenerator:
    """Real-time streaming music generation"""
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.is_streaming = False
        self.parameter_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.current_config = None
        
    def start_streaming(self, initial_config: MusicConfig):
        """Start streaming generation"""
        self.current_config = initial_config
        self.is_streaming = True
        
        # Start background generation thread
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        print("Streaming generation started!")
    
    def stop_streaming(self):
        """Stop streaming generation"""
        self.is_streaming = False
        if hasattr(self, 'generation_thread'):
            self.generation_thread.join(timeout=1.0)
        print("Streaming generation stopped!")
    
    def update_parameters(self, **kwargs):
        """Update generation parameters in real-time"""
        self.parameter_queue.put(kwargs)
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the next audio chunk"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _generation_loop(self):
        """Background generation loop"""
        sampler = MusicSpaceSampler()
        
        while self.is_streaming:
            # Check for parameter updates
            try:
                updates = self.parameter_queue.get_nowait()
                self._update_config(updates)
            except queue.Empty:
                pass
            
            # Generate a short musical segment
            if self.current_config:
                segment_config = self._create_segment_config()
                midi_file = sampler.generate_midi_from_config(segment_config)
                
                # Convert to audio (simplified for streaming)
                # In practice, you'd want a more efficient real-time synthesis
                audio_chunk = self._midi_to_audio_quick(midi_file)
                
                try:
                    self.audio_queue.put_nowait(audio_chunk)
                except queue.Full:
                    # Drop old chunks if queue is full
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio_chunk)
                    except queue.Empty:
                        pass
            
            time.sleep(0.1)  # Small delay to prevent CPU overload
    
    def _update_config(self, updates: Dict):
        """Update current configuration with new parameters"""
        if self.current_config:
            for key, value in updates.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
    
    def _create_segment_config(self) -> MusicConfig:
        """Create a short segment configuration for streaming"""
        segment_config = MusicConfig(
            key=self.current_config.key,
            scale=self.current_config.scale,
            tempo=self.current_config.tempo,
            time_signature=self.current_config.time_signature,
            measures=2,  # Short segments for streaming
            instruments=self.current_config.instruments,
            chord_progression=self.current_config.chord_progression,
            drum_pattern=self.current_config.drum_pattern,
            melodic_density=self.current_config.melodic_density,
            harmonic_complexity=self.current_config.harmonic_complexity,
            swing_factor=self.current_config.swing_factor,
            velocity_variation=self.current_config.velocity_variation,
            articulation_style=self.current_config.articulation_style,
            modulation_intensity=self.current_config.modulation_intensity
        )
        return segment_config
    
    def _midi_to_audio_quick(self, midi_file: mido.MidiFile) -> np.ndarray:
        """Quick MIDI to audio conversion for streaming (simplified)"""
        # This is a placeholder - in practice you'd want real-time synthesis
        # For now, return a simple sine wave based on the first note
        duration = midi_file.length
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Extract first note for frequency (very simplified)
        frequency = 440.0  # Default A4
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    frequency = 440.0 * (2 ** ((msg.note - 69) / 12))
                    break
        
        # Generate simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)

class MusicVAEIntegration:
    """Integration with MusicVAE for latent space sampling"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.available = MAGENTA_AVAILABLE
        
        if self.available:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize MusicVAE model"""
        try:
            # Use a pre-trained MusicVAE model
            self.config = configs.CONFIG_MAP['hierdec-trio_16bar']
            checkpoint_dir = 'models/music_vae/hierdec-trio_16bar'  # You need to download this
            self.model = TrainedModel(self.config, batch_size=1, checkpoint_dir_or_path=checkpoint_dir)
            print("MusicVAE model initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize MusicVAE: {e}")
            self.available = False
    
    def sample_from_latent_space(self, num_samples: int = 10, 
                                temperature: float = 1.0) -> List[mido.MidiFile]:
        """Sample from MusicVAE latent space"""
        if not self.available or not self.model:
            print("MusicVAE not available")
            return []
        
        try:
            # Sample from the latent space
            samples = self.model.sample(n=num_samples, length=32, temperature=temperature)
            
            # Convert to MIDI files
            midi_files = []
            for sample in samples:
                midi_file = self._sequence_to_midi(sample)
                midi_files.append(midi_file)
            
            return midi_files
            
        except Exception as e:
            print(f"MusicVAE sampling failed: {e}")
            return []
    
    def interpolate_sequences(self, start_sequence, end_sequence, 
                            num_steps: int = 10) -> List[mido.MidiFile]:
        """Interpolate between two musical sequences"""
        if not self.available or not self.model:
            return []
        
        try:
            # Encode sequences to latent space
            start_z = self.model.encode([start_sequence])
            end_z = self.model.encode([end_sequence])
            
            # Interpolate in latent space
            interpolated_sequences = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interp_z = (1 - alpha) * start_z + alpha * end_z
                interp_sequence = self.model.decode(interp_z, length=32)[0]
                midi_file = self._sequence_to_midi(interp_sequence)
                interpolated_sequences.append(midi_file)
            
            return interpolated_sequences
            
        except Exception as e:
            print(f"MusicVAE interpolation failed: {e}")
            return []
    
    def _sequence_to_midi(self, sequence) -> mido.MidiFile:
        """Convert MusicVAE sequence to MIDI file"""
        # This is a simplified conversion - you'd need proper sequence parsing
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
        
        # Convert sequence to MIDI messages (simplified)
        # In practice, you'd parse the actual sequence structure
        for i, note in enumerate(sequence[:64]):  # Simplified
            if hasattr(note, 'pitch') and hasattr(note, 'velocity'):
                track.append(mido.Message('note_on', note=int(note.pitch), 
                                        velocity=int(note.velocity), time=0))
                track.append(mido.Message('note_off', note=int(note.pitch), 
                                        velocity=0, time=480))
        
        return mid

class EnhancedMusicSpaceSampler(MusicSpaceSampler):
    """Enhanced version of the original MusicSpaceSampler with new capabilities"""
    
    def __init__(self, output_dir: str = "generated_music", 
                 soundfont_path: str = "GeneralUser-GS.sf2",
                 enable_audio_analysis: bool = True,
                 enable_evolution: bool = False):
        super().__init__(output_dir)
        
        # Initialize enhanced components
        self.audio_renderer = AudioRenderer(soundfont_path)
        self.audio_analyzer = AudioAnalyzer() if enable_audio_analysis else None
        self.diversity_analyzer = DiversityAnalyzer()
        self.evolutionary_refiner = EvolutionaryRefiner() if enable_evolution else None
        self.streaming_generator = StreamingGenerator()
        self.musicvae = MusicVAEIntegration()
        
        # Enhanced generation settings
        self.enable_audio_analysis = enable_audio_analysis
        self.enable_evolution = enable_evolution
        self.output_formats = ['mp3', 'wav']
        
    def generate_diverse_configs(self, num_samples: int = 1000, 
                               use_evolution: bool = False) -> List[MusicConfig]:
        """Enhanced config generation with optional evolutionary refinement"""
        configs = super().generate_diverse_configs(num_samples)
        
        if use_evolution and self.evolutionary_refiner:
            print("Applying evolutionary refinement...")
            # Define fitness function (example: prefer diverse and musically coherent pieces)
            def fitness_func(config: MusicConfig) -> float:
                score = 0.0
                # Reward musical coherence
                if config.scale in ['major', 'minor']:
                    score += 0.3
                # Reward moderate complexity
                score += 0.5 * (1 - abs(config.melodic_density - 0.5))
                score += 0.3 * config.harmonic_complexity
                # Reward reasonable tempo
                if 80 <= config.tempo <= 140:
                    score += 0.2
                return score
            
            configs = self.evolutionary_refiner.refine_population(configs, fitness_func)
        
        return configs
    
    def generate_enhanced_batch(self, num_samples: int = 100, 
                              output_formats: List[str] = None,
                              use_musicvae: bool = False) -> List[MusicMetadata]:
        """Enhanced batch generation with advanced features"""
        if output_formats is None:
            output_formats = ['mp3']
        
        print(f"Generating {num_samples} enhanced musical samples...")
        
        # Generate base configurations
        configs = self.generate_diverse_configs(num_samples, use_evolution=self.enable_evolution)
        
        # Add MusicVAE samples if available
        if use_musicvae and self.musicvae.available:
            print("Adding MusicVAE samples...")
            vae_midis = self.musicvae.sample_from_latent_space(min(num_samples // 4, 25))
            # Convert MusicVAE samples to configs (simplified)
            for i, midi in enumerate(vae_midis[:10]):
                config = self._midi_to_config(midi, f"musicvae_{i}")
                configs.append(config)
        
        print(f"Processing {len(configs)} configurations...")
        metadata_list = []
        
        for i, config in enumerate(configs):
            if i % 10 == 0:
                print(f"Generated {i}/{len(configs)} files...")
            
            try:
                # Generate MIDI
                midi_file = self.generate_midi_from_config(config)
                
                # Calculate basic metadata
                metadata = self.calculate_metadata(config, midi_file)
                metadata.generation_method = 'musicvae' if hasattr(config, '_is_musicvae') else 'rule_based'
                
                # Save and render audio
                for format in output_formats:
                    temp_midi_path = self.output_dir / f"temp_{metadata.embedding_hash}.mid"
                    midi_file.save(str(temp_midi_path))
                    
                    try:
                        # Render to audio
                        audio_path = self.audio_renderer.render_midi_to_audio(temp_midi_path, format)
                        
                        # Extract audio features if enabled
                        if self.enable_audio_analysis and self.audio_analyzer:
                            metadata.audio_features = self.audio_analyzer.extract_features(audio_path)
                        
                        # Update filename to audio format
                        metadata.filename = audio_path.name
                        
                        # Clean up temporary MIDI
                        if temp_midi_path.exists():
                            temp_midi_path.unlink()
                        
                    except Exception as e:
                        print(f"Audio rendering failed for sample {i}: {e}")
                        # Keep MIDI file if audio rendering fails
                        metadata.filename = temp_midi_path.name
                
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"Failed to generate sample {i}: {e}")
                continue
        
        # Perform diversity analysis
        print("Analyzing diversity...")
        diversity_stats = self.diversity_analyzer.analyze_diversity(metadata_list)
        
        # Save enhanced metadata
        metadata_path = self.output_dir / "enhanced_metadata.json"
        metadata_dict = {
            'samples': [asdict(m) for m in metadata_list],
            'diversity_analysis': diversity_stats,
            'generation_settings': {
                'num_samples': num_samples,
                'output_formats': output_formats,
                'audio_analysis_enabled': self.enable_audio_analysis,
                'evolution_enabled': self.enable_evolution,
                'musicvae_enabled': use_musicvae
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"\nGenerated {len(metadata_list)} unique musical pieces!")
        print(f"Files saved to: {self.output_dir}")
        self._print_enhanced_analysis(diversity_stats, metadata_list)
        
        return metadata_list
    
    def _midi_to_config(self, midi_file: mido.MidiFile, source_id: str) -> MusicConfig:
        """Convert MIDI file back to configuration (simplified)"""
        # This is a simplified conversion - in practice you'd analyze the MIDI more thoroughly
        config = MusicConfig(
            key=random.randint(0, 11),
            scale=random.choice(list(SCALES.keys())),
            tempo=120,  # Default tempo
            time_signature=(4, 4),
            measures=16,
            instruments=[0],  # Piano
            chord_progression=[0, 3, 5, 0],
            drum_pattern='basic_rock',
            melodic_density=0.5,
            harmonic_complexity=0.5
        )
        config._is_musicvae = True  # Mark as MusicVAE generated
        return config
    
    def _print_enhanced_analysis(self, diversity_stats: Dict, metadata_list: List[MusicMetadata]):
        """Print enhanced diversity analysis"""
        print("\n=== ENHANCED DIVERSITY ANALYSIS ===")
        
        if 'cluster_distribution' in diversity_stats:
            cluster_info = diversity_stats['cluster_distribution']
            print(f"Perceptual clusters: {cluster_info['n_clusters']}")
            print(f"Cluster entropy: {cluster_info['cluster_entropy']:.2f}")
        
        if 'perceptual_coverage' in diversity_stats:
            print(f"Perceptual coverage: {diversity_stats['perceptual_coverage']:.3f}")
        
        if 'redundancy_score' in diversity_stats:
            print(f"Redundancy score: {diversity_stats['redundancy_score']:.3f}")
        
        # Audio feature analysis
        if any(m.audio_features for m in metadata_list):
            audio_samples = [m for m in metadata_list if m.audio_features]
            spectral_centroids = [m.audio_features.spectral_centroid for m in audio_samples]
            rhythm_complexities = [m.audio_features.rhythm_pattern_complexity for m in audio_samples]
            
            print(f"\nAudio Features Analysis:")
            print(f"Spectral centroid range: {min(spectral_centroids):.0f} - {max(spectral_centroids):.0f} Hz")
            print(f"Rhythm complexity range: {min(rhythm_complexities):.2f} - {max(rhythm_complexities):.2f}")
        
        # Diversity scores
        diversity_scores = [m.diversity_score for m in metadata_list if m.diversity_score]
        if diversity_scores:
            print(f"Average diversity score: {np.mean(diversity_scores):.3f}")
            print(f"Diversity score range: {min(diversity_scores):.3f} - {max(diversity_scores):.3f}")
    
    def start_interactive_generation(self):
        """Start interactive real-time generation"""
        print("\n=== INTERACTIVE MUSIC GENERATION ===")
        print("Commands:")
        print("  'start' - Start streaming generation")
        print("  'stop' - Stop streaming generation") 
        print("  'tempo <bpm>' - Change tempo")
        print("  'key <0-11>' - Change key")
        print("  'density <0-1>' - Change melodic density")
        print("  'complexity <0-1>' - Change harmonic complexity")
        print("  'quit' - Exit interactive mode")
        
        # Create initial configuration
        initial_config = MusicConfig(
            key=0, scale='major', tempo=120, time_signature=(4, 4),
            measures=4, instruments=[0], chord_progression=[0, 3, 5, 0],
            drum_pattern='basic_rock', melodic_density=0.5, harmonic_complexity=0.5
        )
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    if self.streaming_generator.is_streaming:
                        self.streaming_generator.stop_streaming()
                    break
                elif command == 'start':
                    self.streaming_generator.start_streaming(initial_config)
                elif command == 'stop':
                    self.streaming_generator.stop_streaming()
                elif command.startswith('tempo '):
                    try:
                        tempo = int(command.split()[1])
                        self.streaming_generator.update_parameters(tempo=tempo)
                        print(f"Tempo updated to {tempo} BPM")
                    except (ValueError, IndexError):
                        print("Invalid tempo. Use: tempo <number>")
                elif command.startswith('key '):
                    try:
                        key = int(command.split()[1])
                        if 0 <= key <= 11:
                            self.streaming_generator.update_parameters(key=key)
                            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                            print(f"Key updated to {key_names[key]}")
                        else:
                            print("Key must be 0-11")
                    except (ValueError, IndexError):
                        print("Invalid key. Use: key <0-11>")
                elif command.startswith('density '):
                    try:
                        density = float(command.split()[1])
                        if 0 <= density <= 1:
                            self.streaming_generator.update_parameters(melodic_density=density)
                            print(f"Melodic density updated to {density:.2f}")
                        else:
                            print("Density must be 0-1")
                    except (ValueError, IndexError):
                        print("Invalid density. Use: density <0-1>")
                elif command.startswith('complexity '):
                    try:
                        complexity = float(command.split()[1])
                        if 0 <= complexity <= 1:
                            self.streaming_generator.update_parameters(harmonic_complexity=complexity)
                            print(f"Harmonic complexity updated to {complexity:.2f}")
                        else:
                            print("Complexity must be 0-1")
                    except (ValueError, IndexError):
                        print("Invalid complexity. Use: complexity <0-1>")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Stopping streaming...")
                if self.streaming_generator.is_streaming:
                    self.streaming_generator.stop_streaming()
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Enhanced main function with advanced features"""
    print("Enhanced Intelligent Music Space Sampler")
    print("=" * 50)
    
    # Create enhanced sampler
    sampler = EnhancedMusicSpaceSampler(
        output_dir="enhanced_music_samples",
        enable_audio_analysis=True,
        enable_evolution=False  # Set to True to enable evolutionary refinement
    )
    
    # Interactive mode selection
    print("\nSelect mode:")
    print("1. Batch generation")
    print("2. Interactive real-time generation")
    print("3. Both")
    
    try:
        mode = input("Enter choice (1-3): ").strip()
        
        if mode in ['1', '3']:
            # Batch generation
            print("\n=== BATCH GENERATION ===")
            num_samples = int(input("Number of samples to generate (default 20): ") or "20")
            use_musicvae = input("Use MusicVAE sampling? (y/n, default n): ").strip().lower() == 'y'
            
            # Generate enhanced batch
            metadata_list = sampler.generate_enhanced_batch(
                num_samples=num_samples,
                output_formats=['mp3'],
                use_musicvae=use_musicvae
            )
            
            print(f"\n=== GENERATION COMPLETE ===")
            print(f"Generated {len(metadata_list)} unique pieces")
            
            # Show sample metadata
            print("\n=== SAMPLE METADATA ===")
            for i, metadata in enumerate(metadata_list[:3]):
                print(f"\nSample {i+1}: {metadata.filename}")
                print(f"  Key: {metadata.key} {metadata.scale}")
                print(f"  Tempo: {metadata.tempo} BPM")
                print(f"  Duration: {metadata.duration_seconds:.1f}s")
                print(f"  Diversity Score: {metadata.diversity_score:.3f}" if metadata.diversity_score else "")
                print(f"  Cluster: {metadata.perceptual_cluster}" if metadata.perceptual_cluster is not None else "")
        
        if mode in ['2', '3']:
            # Interactive mode
            sampler.start_interactive_generation()
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()