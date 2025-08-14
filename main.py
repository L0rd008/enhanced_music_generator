#!/usr/bin/env python3
"""
Enhanced AI-Powered Music Generation Suite
Main application entry point with multiple modes of operation
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to path to import the original msc module
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_generator import EnhancedMusicGenerator, GenerationRequest
from core.web_api import MusicGeneratorAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('music_generator.log')
    ]
)

logger = logging.getLogger(__name__)

class MusicGenerationApp:
    """Main application class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.generator = EnhancedMusicGenerator(config_path)
        self.api = None
        
    async def run_batch_generation(self, args):
        """Run batch music generation"""
        print("🎵 Starting Batch Music Generation")
        print("=" * 50)
        
        # Create generation request
        request = GenerationRequest(
            num_samples=args.samples,
            output_formats=args.formats,
            quality_level=args.quality,
            enable_evolution=args.evolution,
            enable_musicvae=args.musicvae,
            style_preferences=self._parse_style_preferences(args.styles),
            user_id="batch_user",
            session_id="batch_session"
        )
        
        try:
            # Generate music
            result = await self.generator.generate_music_async(request)
            
            # Display results
            print(f"\n✅ Generation Complete!")
            print(f"Request ID: {result.request_id}")
            print(f"Generated {len(result.samples)} samples in {result.generation_time:.2f}s")
            print(f"Output directory: {self.generator.config['output_dir']}")
            
            # Show sample details
            print("\n📊 Generated Samples:")
            for i, sample in enumerate(result.samples[:5], 1):  # Show first 5
                quality = result.quality_metrics.get(sample.filename, {}).get('overall_quality', 0)
                print(f"  {i}. {sample.filename}")
                print(f"     Key: {sample.key} {sample.scale}, Tempo: {sample.tempo} BPM")
                print(f"     Duration: {sample.duration_seconds:.1f}s, Quality: {quality:.2f}")
            
            if len(result.samples) > 5:
                print(f"     ... and {len(result.samples) - 5} more samples")
            
            # Show diversity analysis
            if result.diversity_analysis:
                print(f"\n🎯 Diversity Analysis:")
                coverage = result.diversity_analysis.get('perceptual_coverage', 0)
                redundancy = result.diversity_analysis.get('redundancy_score', 0)
                print(f"  Perceptual Coverage: {coverage:.3f}")
                print(f"  Redundancy Score: {redundancy:.3f}")
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            print(f"❌ Generation failed: {e}")
            return 1
        
        return 0
    
    def run_web_interface(self, args):
        """Run web interface"""
        print("🌐 Starting Web Interface")
        print("=" * 50)
        print(f"Server will start at: http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            self.api = MusicGeneratorAPI(self.generator)
            self.api.run(
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            logger.error(f"Web server failed: {e}")
            print(f"❌ Server failed: {e}")
            return 1
        
        return 0
    
    def run_interactive_cli(self, args):
        """Run interactive CLI mode"""
        print("🎹 Interactive Music Generation")
        print("=" * 50)
        print("Commands:")
        print("  generate <num> - Generate music samples")
        print("  styles - Show available styles")
        print("  analytics - Show analytics")
        print("  help - Show this help")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n🎵 > ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    self._show_cli_help()
                elif command == 'styles':
                    self._show_available_styles()
                elif command == 'analytics':
                    self._show_analytics()
                elif command.startswith('generate'):
                    parts = command.split()
                    num_samples = int(parts[1]) if len(parts) > 1 else 5
                    asyncio.run(self._interactive_generate(num_samples))
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        return 0
    
    async def _interactive_generate(self, num_samples: int):
        """Generate music interactively"""
        print(f"Generating {num_samples} samples...")
        
        request = GenerationRequest(
            num_samples=num_samples,
            output_formats=['mp3'],
            user_id="interactive_user",
            session_id="interactive_session"
        )
        
        try:
            result = await self.generator.generate_music_async(request)
            print(f"✅ Generated {len(result.samples)} samples in {result.generation_time:.2f}s")
            
            for sample in result.samples:
                print(f"  📄 {sample.filename} - {sample.key} {sample.scale}, {sample.tempo} BPM")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    def _show_cli_help(self):
        """Show CLI help"""
        print("\n📖 Available Commands:")
        print("  generate <num>  - Generate specified number of samples (default: 5)")
        print("  styles          - Show available music styles")
        print("  analytics       - Show generation analytics")
        print("  help            - Show this help message")
        print("  quit/exit       - Exit the application")
    
    def _show_available_styles(self):
        """Show available music styles"""
        print("\n🎼 Available Music Styles:")
        styles = self.generator.style_learner.style_categories
        for i, style in enumerate(styles, 1):
            print(f"  {i}. {style.title()}")
    
    def _show_analytics(self):
        """Show analytics dashboard data"""
        print("\n📊 Analytics Dashboard:")
        data = self.generator.get_analytics_dashboard_data()
        
        metrics = data.get('performance_metrics', {})
        print(f"  Total Generations: {metrics.get('total_generations', 0)}")
        print(f"  Average Quality: {metrics.get('average_quality', 0):.2f}")
        print(f"  User Satisfaction: {metrics.get('user_satisfaction', 0):.2f}")
        print(f"  Avg Generation Time: {metrics.get('generation_time_avg', 0):.2f}s")
        
        # Show popular styles
        style_data = data.get('style_preferences', {})
        popular_styles = style_data.get('most_popular_styles', [])
        if popular_styles:
            print(f"  Popular Styles: {', '.join(popular_styles)}")
    
    def _parse_style_preferences(self, styles_arg: Optional[str]) -> Optional[dict]:
        """Parse style preferences from command line argument"""
        if not styles_arg:
            return None
        
        preferences = {}
        for style_weight in styles_arg.split(','):
            if ':' in style_weight:
                style, weight = style_weight.split(':', 1)
                preferences[style.strip()] = float(weight.strip())
            else:
                preferences[style_weight.strip()] = 1.0
        
        return preferences if preferences else None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI-Powered Music Generation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Web interface
  python main.py web --port 8000
  
  # Batch generation
  python main.py batch --samples 20 --formats mp3 wav
  
  # Interactive CLI
  python main.py interactive
  
  # Batch with style preferences
  python main.py batch --samples 10 --styles "jazz:0.8,classical:0.5"
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Web interface mode
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host address')
    web_parser.add_argument('--port', type=int, default=8000, help='Port number')
    web_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Batch generation mode
    batch_parser = subparsers.add_parser('batch', help='Batch generation mode')
    batch_parser.add_argument('--samples', type=int, default=10, help='Number of samples')
    batch_parser.add_argument('--formats', nargs='+', default=['mp3'], 
                             choices=['mp3', 'wav'], help='Output formats')
    batch_parser.add_argument('--quality', choices=['low', 'medium', 'high'], 
                             default='high', help='Audio quality')
    batch_parser.add_argument('--evolution', action='store_true', 
                             help='Enable evolutionary refinement')
    batch_parser.add_argument('--musicvae', action='store_true', 
                             help='Enable MusicVAE generation')
    batch_parser.add_argument('--styles', type=str, 
                             help='Style preferences (e.g., "jazz:0.8,rock:0.5")')
    
    # Interactive CLI mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive CLI mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Default to interactive mode if no mode specified
    if not args.mode:
        args.mode = 'interactive'
    
    # Create and run application
    try:
        app = MusicGenerationApp(args.config)
        
        if args.mode == 'web':
            return app.run_web_interface(args)
        elif args.mode == 'batch':
            return asyncio.run(app.run_batch_generation(args))
        elif args.mode == 'interactive':
            return app.run_interactive_cli(args)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Application failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
