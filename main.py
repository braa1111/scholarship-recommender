#!/usr/bin/env python3
"""
Scholarship Recommender System - Main Entry Point
BSc Thesis Project - University of Khartoum
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.config import get_config, update_config
from data_preprocessing import DataPreprocessor
from collaborative_filtering import CollaborativeFiltering
from nlp_matching import NLPMatching
from hybrid_engine import HybridRecommender

def setup_environment():
    """Setup project environment and directories"""
    print("Setting up project environment...")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed', 
        'models/trained_models',
        'tests/test_data',
        'web_app/static/css',
        'web_app/static/js',
        'web_app/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("Environment setup completed!")

def generate_sample_data():
    """Generate sample data for testing"""
    print("\nGenerating sample data...")
    
    try:
        from data.sample_data_generator import generate_sample_data as gen_data
        students, scholarships, ratings = gen_data()
        
        print(f"✓ Generated {len(students)} students")
        print(f"✓ Generated {len(scholarships)} scholarships") 
        print(f"✓ Generated {len(ratings)} ratings")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating sample data: {e}")
        return False

def train_models():
    """Train all machine learning models"""
    print("\nTraining machine learning models...")
    
    config = get_config()
    
    try:
        # Step 1: Preprocess data
        print("Step 1: Preprocessing data...")
        preprocessor = DataPreprocessor()
        students, scholarships, ratings = preprocessor.run_full_preprocessing()
        
        # Step 2: Train Collaborative Filtering
        print("Step 2: Training Collaborative Filtering model...")
        cf = CollaborativeFiltering()
        cf.load_data()
        cf.train_model(algorithm=config.model.CF_ALGORITHM)
        cf.evaluate_model()
        cf.save_model()
        
        # Step 3: Setup NLP Matching
        print("Step 3: Setting up NLP matching...")
        nlp = NLPMatching(model_name=config.model.SENTENCE_MODEL)
        nlp.load_models()
        nlp.load_data()
        nlp.generate_embeddings()
        nlp.save_embeddings()
        
        # Step 4: Test Hybrid System
        print("Step 4: Testing hybrid system...")
        hybrid = HybridRecommender(
            cf_weight=config.model.CF_WEIGHT,
            nlp_weight=config.model.NLP_WEIGHT
        )
        hybrid.load_components()
        
        # Test with sample student
        sample_student = students.iloc[0]['student_id']
        recommendations = hybrid.get_hybrid_recommendations(sample_student, top_n=5)
        
        print(f"✓ Sample recommendations for {sample_student}:")
        for i, rec in recommendations.iterrows():
            print(f"  {i+1}. {rec['title']} (Score: {rec['hybrid_score']:.3f})")
        
        print("\n🎉 All models trained successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error training models: {e}")
        return False

def run_web_app():
    """Start the Flask web application"""
    print("\nStarting web application...")
    
    try:
        from web_app.app import app
        config = get_config()
        
        print(f"🌐 Starting Scholarship Recommender System...")
        print(f"   Access the application at: http://{config.web.HOST}:{config.web.PORT}")
        print(f"   Press Ctrl+C to stop the server")
        
        app.run(
            debug=config.web.DEBUG,
            host=config.web.HOST, 
            port=config.web.PORT
        )
        
    except Exception as e:
        print(f"✗ Error starting web application: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        from tests.test_system import run_performance_tests
        run_performance_tests()
        return True
        
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def show_system_info():
    """Display system information"""
    config = get_config()
    
    print("\n" + "="*60)
    print("SCHOLARSHIP RECOMMENDER SYSTEM")
    print("="*60)
    
    print(f"Application: {config.APP_NAME} v{config.VERSION}")
    print(f"Description: {config.DESCRIPTION}")
    print(f"University: {config.UNIVERSITY}")
    print(f"Department: {config.DEPARTMENT}")
    print(f"Authors: {', '.join(config.AUTHORS)}")
    
    print("\nConfiguration:")
    print(f"  - Collaborative Filtering: {config.model.CF_ALGORITHM}")
    print(f"  - Hybrid Weights: CF={config.model.CF_WEIGHT}, NLP={config.model.NLP_WEIGHT}")
    print(f"  - Web Server: http://{config.web.HOST}:{config.web.PORT}")
    
    print("\nAvailable Commands:")
    print("  python main.py setup     - Setup project environment")
    print("  python main.py train     - Train machine learning models") 
    print("  python main.py web       - Start web application")
    print("  python main.py test      - Run system tests")
    print("  python main.py all       - Run complete setup pipeline")
    print("  python main.py info      - Show system information")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Scholarship Recommender System')
    parser.add_argument('command', nargs='?', default='info',
                       choices=['setup', 'train', 'web', 'test', 'all', 'info'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
        generate_sample_data()
        
    elif args.command == 'train':
        train_models()
        
    elif args.command == 'web':
        run_web_app()
        
    elif args.command == 'test':
        run_tests()
        
    elif args.command == 'all':
        setup_environment()
        generate_sample_data()
        train_models()
        run_tests()
        run_web_app()
        
    elif args.command == 'info':
        show_system_info()
        
    else:
        show_system_info()

if __name__ == '__main__':
    main()