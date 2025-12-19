import pandas as pd
import os

def create_processed_data():
    print("🔄 Creating processed data files...")
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Read raw data
        students_df = pd.read_csv('data/raw/students.csv')
        scholarships_df = pd.read_csv('data/raw/scholarships.csv')
        ratings_df = pd.read_csv('data/raw/ratings.csv')
        
        print(f"📊 Raw data loaded: {len(students_df)} students, {len(scholarships_df)} scholarships, {len(ratings_df)} ratings")
        
        # Simple preprocessing - just copy for now
        students_df.to_csv('data/processed/students_processed.csv', index=False)
        scholarships_df.to_csv('data/processed/scholarships_processed.csv', index=False)
        ratings_df.to_csv('data/processed/ratings_processed.csv', index=False)
        
        print("✅ Processed data created successfully!")
        print("📁 Files created in data/processed/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating processed data: {e}")
        return False

if __name__ == "__main__":
    create_processed_data()