# prepare_dataset.py (CORRECTED VERSION)
import pandas as pd
import os

print("📂 Current directory:", os.getcwd())

# Check if resume_dataset.csv exists in current directory
if not os.path.exists('resume_dataset.csv'):
    print("❌ ERROR: resume_dataset.csv not found in current directory!")
    print("\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    print("\n💡 Hint: Make sure you're in the right directory!")
    print("   Current directory:", os.getcwd())
    print("   Should be: D:\\Visual Studio\\.vscode\\AI-Resume-Screening-System-main\\AI-Resume-Screening-System-main")
    exit()

print("✅ Found resume_dataset.csv")

# Create dataset folder
os.makedirs('dataset', exist_ok=True)
os.makedirs('dataset/resumes', exist_ok=True)
print("✅ Created dataset/resumes folders")

# Read the dataset
df = pd.read_csv('resume_dataset.csv')  # Fixed path
print(f"✅ Loaded {len(df)} resumes from CSV")

# Create categories.csv
categories = []
for i in range(min(50, len(df))):  # Use first 50 resumes
    # Get job category - using job_position_name column
    category = df.iloc[i].get('job_position_name', 'Unknown')
    if pd.isna(category):
        category = 'Unknown'
    
    categories.append({
        'filename': f'resume_{i}.pdf',  # Note: using .pdf extension
        'category': str(category).strip()
    })
    
    # Create a simple text file for each resume (since we don't have actual PDFs)
    resume_text = []
    
    # Add career objective
    if pd.notna(df.iloc[i].get('career_objective', '')):
        resume_text.append(str(df.iloc[i]['career_objective']))
    
    # Add skills
    if pd.notna(df.iloc[i].get('skills', '')):
        resume_text.append(f"Skills: {df.iloc[i]['skills']}")
    
    # Add experience
    if pd.notna(df.iloc[i].get('professional_company_names', '')):
        resume_text.append(f"Experience at: {df.iloc[i]['professional_company_names']}")
    
    # Save as .txt file (easier for testing)
    with open(f'dataset/resumes/resume_{i}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(resume_text))
    
    # Also create empty .pdf files (to satisfy the filename in categories.csv)
    with open(f'dataset/resumes/resume_{i}.pdf', 'w') as f:
        f.write('This is a placeholder PDF file')

# Save categories.csv
categories_df = pd.DataFrame(categories)
categories_df.to_csv('dataset/categories.csv', index=False)

print(f"✅ Created dataset/categories.csv with {len(categories_df)} entries")
print("\n📊 First 5 categories:")
print(categories_df.head())

print("\n📁 Dataset folder contents:")
if os.path.exists('dataset'):
    print(os.listdir('dataset'))
else:
    print("Dataset folder not found")

print("\n📁 Resumes folder contents (first 5 files):")
if os.path.exists('dataset/resumes'):
    resumes = os.listdir('dataset/resumes')
    print(resumes[:5] if resumes else "No files found")
else:
    print("Resumes folder not found")

print("\n✅ Dataset preparation complete!")
print("\nNow run: python data_preprocessor.py")