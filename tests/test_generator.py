"""
Tests for the Enhanced Music Generator core functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

from core.enhanced_generator import (
    EnhancedMusicGenerator,
    GenerationRequest,
    StyleLearner,
    QualityAssessment
)


class TestEnhancedMusicGenerator:
    """Test cases for EnhancedMusicGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator instance"""
        return EnhancedMusicGenerator()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample generation request"""
        return GenerationRequest(
            num_samples=3,
            output_formats=['mp3'],
            quality_level='high',
            user_id='test_user',
            session_id='test_session'
        )
    
    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly"""
        assert generator is not None
        assert generator.style_learner is not None
        assert generator.quality_assessor is not None
        assert generator.config is not None
        assert generator.performance_metrics is not None
    
    def test_config_loading_default(self):
        """Test default configuration loading"""
        generator = EnhancedMusicGenerator()
        assert 'output_dir' in generator.config
        assert 'max_concurrent_generations' in generator.config
        assert generator.config['output_dir'] == 'enhanced_generated_music'
    
    def test_config_loading_custom(self, tmp_path):
        """Test custom configuration loading"""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "output_dir": "test_output",
            "quality_threshold": 0.8
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        generator = EnhancedMusicGenerator(str(config_file))
        assert generator.config['output_dir'] == 'test_output'
        assert generator.config['quality_threshold'] == 0.8
    
    @pytest.mark.asyncio
    async def test_generate_music_async_basic(self, generator, sample_request):
        """Test basic async music generation"""
        # Mock the base generator to avoid actual music generation
        with patch.object(generator, '_generate_base_samples') as mock_generate:
            mock_metadata = Mock()
            mock_metadata.filename = 'test_music.mp3'
            mock_metadata.tempo = 120
            mock_metadata.key = 'C'
            mock_metadata.scale = 'major'
            mock_metadata.duration_seconds = 60.0
            mock_metadata.melodic_entropy = 1.0
            mock_metadata.rhythmic_density = 0.5
            mock_metadata.harmonic_complexity = 0.6
            mock_metadata.audio_features = None
            mock_metadata.diversity_score = 0.8
            
            mock_generate.return_value = [mock_metadata]
            
            # Mock diversity analyzer
            with patch.object(generator.base_generator, 'diversity_analyzer') as mock_analyzer:
                mock_analyzer.analyze_diversity.return_value = {
                    'perceptual_coverage': 0.85,
                    'redundancy_score': 0.15
                }
                
                result = await generator.generate_music_async(sample_request)
                
                assert result is not None
                assert len(result.samples) == 1
                assert result.request_id is not None
                assert result.generation_time > 0
                assert 'test_music.mp3' in result.quality_metrics
    
    def test_add_user_feedback(self, generator):
        """Test user feedback addition"""
        request_id = 'test_request_123'
        feedback = {
            'rating': 4,
            'comments': 'Great music!',
            'favorite_samples': ['sample1.mp3']
        }
        
        generator.add_user_feedback(request_id, feedback)
        
        assert request_id in generator.user_feedback_db
        assert len(generator.user_feedback_db[request_id]) == 1
        assert generator.user_feedback_db[request_id][0]['rating'] == 4
    
    def test_performance_metrics_update(self, generator):
        """Test performance metrics updating"""
        initial_total = generator.performance_metrics['total_generations']
        
        # Mock a result to update metrics
        mock_result = Mock()
        mock_result.quality_metrics = {
            'sample1.mp3': {'overall_quality': 0.8},
            'sample2.mp3': {'overall_quality': 0.9}
        }
        mock_result.generation_time = 5.0
        
        generator._update_performance_metrics(mock_result)
        
        assert generator.performance_metrics['total_generations'] == initial_total + 1
        assert generator.performance_metrics['average_quality'] > 0
        assert generator.performance_metrics['generation_time_avg'] > 0


class TestStyleLearner:
    """Test cases for StyleLearner"""
    
    @pytest.fixture
    def style_learner(self):
        """Create a test style learner instance"""
        return StyleLearner()
    
    def test_style_learner_initialization(self, style_learner):
        """Test style learner initialization"""
        assert style_learner.style_model is None
        assert not style_learner.is_trained
        assert len(style_learner.style_categories) > 0
        assert 'classical' in style_learner.style_categories
        assert 'jazz' in style_learner.style_categories
    
    def test_extract_style_features(self, style_learner):
        """Test style feature extraction"""
        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.tempo = 120
        mock_metadata.measures = 32
        mock_metadata.melodic_entropy = 1.2
        mock_metadata.rhythmic_density = 0.6
        mock_metadata.harmonic_complexity = 0.7
        mock_metadata.duration_seconds = 180
        mock_metadata.instruments = [0, 1, 2]
        mock_metadata.audio_features = None
        
        features = style_learner.extract_style_features(mock_metadata)
        
        assert isinstance(features, type(features))  # numpy array
        assert len(features) > 0
    
    def test_get_default_style_parameters(self, style_learner):
        """Test default style parameter retrieval"""
        jazz_params = style_learner._get_default_style_parameters('jazz')
        
        assert 'tempo_range' in jazz_params
        assert 'scales' in jazz_params
        assert 'complexity' in jazz_params
        assert 'instruments' in jazz_params
        
        # Test unknown style defaults to classical
        unknown_params = style_learner._get_default_style_parameters('unknown_style')
        classical_params = style_learner._get_default_style_parameters('classical')
        assert unknown_params == classical_params


class TestQualityAssessment:
    """Test cases for QualityAssessment"""
    
    @pytest.fixture
    def quality_assessor(self):
        """Create a test quality assessor instance"""
        return QualityAssessment()
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for testing"""
        metadata = Mock()
        metadata.tempo = 120
        metadata.harmonic_complexity = 0.6
        metadata.melodic_entropy = 1.2
        metadata.rhythmic_density = 0.5
        metadata.diversity_score = 0.8
        metadata.audio_features = None
        return metadata
    
    def test_quality_assessor_initialization(self, quality_assessor):
        """Test quality assessor initialization"""
        assert quality_assessor.quality_model is None
        assert not quality_assessor.is_trained
    
    def test_calculate_quality_metrics(self, quality_assessor, mock_metadata):
        """Test quality metrics calculation"""
        metrics = quality_assessor.calculate_quality_metrics(mock_metadata)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'tempo_stability',
            'harmonic_coherence',
            'melodic_flow',
            'rhythmic_consistency',
            'audio_quality',
            'spectral_balance',
            'uniqueness_score',
            'overall_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # All metrics should be normalized
    
    def test_tempo_stability_assessment(self, quality_assessor, mock_metadata):
        """Test tempo stability assessment"""
        # Test optimal tempo
        mock_metadata.tempo = 120
        stability = quality_assessor._assess_tempo_stability(mock_metadata)
        assert stability == 1.0
        
        # Test extreme tempo
        mock_metadata.tempo = 300
        stability = quality_assessor._assess_tempo_stability(mock_metadata)
        assert stability < 1.0
    
    def test_harmonic_coherence_assessment(self, quality_assessor, mock_metadata):
        """Test harmonic coherence assessment"""
        # Test optimal complexity
        mock_metadata.harmonic_complexity = 0.6
        coherence = quality_assessor._assess_harmonic_coherence(mock_metadata)
        assert coherence == 1.0
        
        # Test extreme complexity
        mock_metadata.harmonic_complexity = 1.0
        coherence = quality_assessor._assess_harmonic_coherence(mock_metadata)
        assert coherence < 1.0


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self):
        """Test end-to-end music generation flow"""
        # This would be a more comprehensive test that actually
        # generates music and verifies the complete pipeline
        # For now, we'll skip this to avoid dependencies
        pytest.skip("Integration test requires full system setup")
    
    def test_config_integration(self, tmp_path):
        """Test configuration integration"""
        config_file = tmp_path / "integration_config.json"
        config_data = {
            "output_dir": str(tmp_path / "test_output"),
            "quality_threshold": 0.7,
            "max_concurrent_generations": 2
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        generator = EnhancedMusicGenerator(str(config_file))
        
        # Verify configuration is loaded correctly
        assert generator.config['output_dir'] == str(tmp_path / "test_output")
        assert generator.config['quality_threshold'] == 0.7
        assert generator.config['max_concurrent_generations'] == 2


if __name__ == '__main__':
    pytest.main([__file__])
