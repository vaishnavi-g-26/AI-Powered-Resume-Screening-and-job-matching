# data_preprocessor.py
import os
import pandas as pd
from resume_parser import read_pdf, read_docx, clean_text
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

class ResumeDataCollector:
    def __init__(self, dataset_path="dataset"):
        """
        Initialize the data collector
        dataset_path: folder containing resumes and categories.csv
        """
        self.dataset_path = dataset_path
        self.resumes_path = os.path.join(dataset_path, "resumes")
        self.categories_file = os.path.join(dataset_path, "categories.csv")
        
    def load_and_process_resumes(self):
        """
        Load all resumes, extract text, and match with categories
        """
        # Read categories mapping
        categories_df = pd.read_csv(self.categories_file)
        print(f"📊 Found {len(categories_df)} resume entries in categories.csv")
        
        processed_data = []
        errors = []
        
        for index, row in categories_df.iterrows():
            filename = row['filename']
            category = row['category']
            
            file_path = os.path.join(self.resumes_path, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                errors.append(f"❌ File not found: {filename}")
                continue
            
            try:
                # Extract text based on file extension
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        text = read_pdf(f)
                elif filename.endswith('.docx'):
                    text = read_docx(file_path)
                else:
                    errors.append(f"❌ Unsupported format: {filename}")
                    continue
                
                # Clean the text using your existing function
                cleaned_text = clean_text(text)
                
                # Store processed data
                processed_data.append({
                    'filename': filename,
                    'raw_text': text,
                    'cleaned_text': cleaned_text,
                    'category': category,
                    'text_length': len(cleaned_text)
                })
                
                print(f"✅ Processed: {filename} → {category} ({len(cleaned_text)} chars)")
                
            except Exception as e:
                errors.append(f"❌ Error processing {filename}: {str(e)}")
        
        # Print summary
        print(f"\n📈 Processing Complete!")
        print(f"   Successfully processed: {len(processed_data)} resumes")
        print(f"   Errors: {len(errors)}")
        
        if errors:
            print("\n⚠️ Errors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   {error}")
        
        return pd.DataFrame(processed_data)
    
    def create_training_data(self, processed_df):
        """
        Prepare data for training ML models
        """
        # Create feature matrix (X) and labels (y)
        X = processed_df['cleaned_text'].values
        y = processed_df['category'].values
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data
        self.save_processed_data(processed_df, X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, X_train, X_test, y_train, y_test):
        """
        Save processed data for later use
        """
        # Save full dataframe
        df.to_csv(os.path.join(self.dataset_path, "processed_resumes.csv"), index=False)
        
        # Save training data as pickle
        training_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'categories': list(df['category'].unique())
        }
        
        with open(os.path.join(self.dataset_path, "training_data.pkl"), 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"\n💾 Saved processed data to {self.dataset_path}/")
        print(f"   Categories found: {training_data['categories']}")
    
    def analyze_dataset(self, df):
        """
        Analyze the dataset distribution
        """
        print("\n📊 Dataset Analysis:")
        print("-" * 50)
        
        # Category distribution
        category_counts = df['category'].value_counts()
        print("\n🏷️ Category Distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} resumes ({percentage:.1f}%)")
        
        # Text length statistics
        print(f"\n📏 Text Length Statistics:")
        print(f"   Average: {df['text_length'].mean():.0f} characters")
        print(f"   Min: {df['text_length'].min()} characters")
        print(f"   Max: {df['text_length'].max()} characters")
        
        return category_counts

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    # Initialize collector
    collector = ResumeDataCollector(dataset_path="dataset")
    
    # Load and process all resumes
    print("🚀 Starting Resume Data Collection...\n")
    processed_df = collector.load_and_process_resumes()
    
    if len(processed_df) > 0:
        # Analyze dataset
        collector.analyze_dataset(processed_df)
        
        # Create training data
        X_train, X_test, y_train, y_test = collector.create_training_data(processed_df)
        
        print(f"\n✅ Training data ready!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
    else:
        print("\n❌ No resumes were successfully processed. Please check your dataset.")
        