import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Scholarship Recommender System...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ Please activate your virtual environment first!")
        print("Run: scholarship_env\\Scripts\\activate")
        return
    
    # Install packages in batches
    packages_batches = [
        ["numpy", "pandas", "scikit-learn"],
        ["scikit-surprise", "nltk"],
        ["spacy", "flask"],
        ["transformers", "sentence-transformers", "torch"],
        ["sqlalchemy", "dataset", "tqdm"]
    ]
    
    for i, batch in enumerate(packages_batches, 1):
        if not run_command(f"pip install {' '.join(batch)}", f"Installing batch {i}/5"):
            print(f"Failed to install batch: {batch}")
            return
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("Trying alternative download method...")
        run_command(
            "pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
            "Alternative spaCy model download"
        )
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ NLTK download failed: {e}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import numpy, pandas, sklearn, surprise, nltk, spacy, flask, transformers
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python main.py setup")
    print("2. Run: python main.py train") 
    print("3. Run: python main.py web")

if __name__ == "__main__":
    main()