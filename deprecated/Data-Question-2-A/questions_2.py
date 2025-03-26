import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

def analyze_csv_file(file_path):
    """Analyze a CSV file and return information about its contents."""
    try:
        # Get file size in MB
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        file_name = os.path.basename(file_path)
        
        print(f"\n{'='*50}")
        print(f"ANALYZING: {file_name} (Size: {file_size:.2f} MB)")
        print(f"{'='*50}")
        
        # For very large files, only read a sample
        if file_size > 100:
            print(f"File is large ({file_size:.2f} MB). Reading only first 1000 rows as sample.")
            df = pd.read_csv(file_path, nrows=1000)
            is_sample = True
        else:
            df = pd.read_csv(file_path)
            is_sample = False
        
        # Basic information
        print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
        if is_sample:
            print("(Sample data - actual file may be much larger)")
        
        # Column information - just count by type
        dtypes_count = df.dtypes.value_counts()
        print(f"Column types: {dict(dtypes_count)}")
        
        # List column names but don't print types
        print(f"Columns: {', '.join(df.columns[:10])}" + ("..." if len(df.columns) > 10 else ""))
        
        # Infer what the dataset is about
        infer_dataset_theme(df, file_name)
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")

def infer_dataset_theme(df, file_name):
    """Try to infer what the dataset is about based on column names and file name."""
    themes_found = []
    
    # Check file name for clues
    if "spotify" in file_name.lower():
        if "revenue" in file_name.lower():
            themes_found.append("Spotify revenue data")
        elif "users" in file_name.lower():
            themes_found.append("Spotify user statistics")
        else:
            themes_found.append("Spotify-related data")
    
    elif "imf" in file_name.lower() or "IMF" in file_name:
        themes_found.append("International Monetary Fund (IMF) data")
        
        if "labor" in file_name.lower() or "LS" in file_name:
            themes_found.append("Labor statistics")
        
        elif "CPI" in file_name:
            themes_found.append("Consumer Price Index data")
            
        elif "FAD" in file_name or "FM" in file_name:
            themes_found.append("Fiscal or Financial Market data")
    
    # Check column names for clues
    columns_lower = [col.lower() for col in df.columns]
    column_text = ' '.join(columns_lower)
    
    themes = {
        'economic': ['gdp', 'growth', 'inflation', 'economy', 'price', 'cpi', 'index'],
        'financial': ['revenue', 'profit', 'sales', 'income', 'cost', 'expense'],
        'demographic': ['population', 'gender', 'age', 'household', 'person', 'people'],
        'geographic': ['country', 'region', 'city', 'area', 'location', 'geographic'],
        'temporal': ['year', 'month', 'date', 'period', 'time', 'quarter']
    }
    
    for theme, keywords in themes.items():
        if any(keyword in column_text for keyword in keywords):
            themes_found.append(f"Contains {theme} data")
    
    # Look for country/region indicators
    if 'country' in columns_lower or 'countries' in columns_lower or 'region' in columns_lower:
        try:
            country_col = next(col for col in df.columns if 'country' in col.lower() or 'region' in col.lower())
            countries = df[country_col].dropna().unique()
            themes_found.append(f"Contains data for {len(countries)} countries/regions")
            if len(countries) <= 5:
                themes_found.append(f"Countries/regions: {', '.join(str(c) for c in countries)}")
            else:
                themes_found.append(f"Sample countries: {', '.join(str(c) for c in countries[:3])}, ...")
        except (StopIteration, TypeError):
            pass
    
    # Check for date columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
    if date_cols:
        date_ranges = []
        for col in date_cols[:2]:  # Limit to first 2 date columns
            if df[col].dtype == 'object':  # Strings that might be dates
                date_ranges.append(f"{col}: {len(df[col].dropna().unique())} unique values")
            else:
                date_ranges.append(col)
        if date_ranges:
            themes_found.append(f"Date/time columns: {', '.join(date_ranges)}")
    
    # Print findings
    print("SUMMARY:")
    for theme in themes_found:
        print(f"• {theme}")

def analyze_imf_labor_statistics():
    """Specific analysis for the IMF labor statistics file."""
    file_path = 'imf_labor_statistics_normal.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        # Try with full path
        directory = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2'
        file_path = os.path.join(directory, file_path)
        if not os.path.exists(file_path):
            print(f"Error: IMF labor statistics file not found")
            return
    
    print(f"\n{'#'*60}")
    print(f"DETAILED ANALYSIS OF IMF LABOR STATISTICS")
    print(f"{'#'*60}")
    
    # Read a sample of the data
    print("Reading sample of IMF labor statistics data...")
    df = pd.read_csv(file_path, nrows=10000)
    
    # Basic statistics
    print(f"\n1. BASIC STATISTICS:")
    print(f"   - Sample size: {df.shape[0]} rows")
    print(f"   - Features: {df.shape[1]} columns")
    
    # Analyze countries coverage
    country_col = None
    for col in df.columns:
        if 'country' in col.lower():
            country_col = col
            break
    
    if country_col:
        countries = df[country_col].nunique()
        print(f"\n2. GEOGRAPHIC COVERAGE:")
        print(f"   - Countries/regions: {countries}")
        # Top 5 countries by data points
        top_countries = df[country_col].value_counts().head(5)
        print(f"   - Top represented countries/regions:")
        for country, count in top_countries.items():
            print(f"     * {country}: {count} data points")
    
    # Analyze time span
    time_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower() or 'time' in col.lower()]
    if time_cols:
        print(f"\n3. TIME COVERAGE:")
        for col in time_cols[:2]:  # First 2 time columns
            try:
                if df[col].dtype == 'object':
                    unique_times = df[col].nunique()
                    print(f"   - {col}: {unique_times} unique time periods")
                else:
                    min_time = df[col].min()
                    max_time = df[col].max()
                    print(f"   - {col} range: {min_time} to {max_time}")
            except:
                print(f"   - Could not analyze time column: {col}")
    
    # Analyze key labor metrics - find numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter to likely labor stat columns
    labor_keywords = ['labor', 'employment', 'unemployment', 'workforce', 'wage', 'salary', 'job']
    labor_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in labor_keywords)]
    if not labor_cols:
        # If no obvious labor columns, take a sample of numeric columns
        labor_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    
    if labor_cols:
        print(f"\n4. KEY LABOR METRICS:")
        for col in labor_cols:
            try:
                mean_val = df[col].mean()
                median_val = df[col].median()
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"   - {col}:")
                print(f"     * Mean: {mean_val:.2f}")
                print(f"     * Median: {median_val:.2f}")
                print(f"     * Range: {min_val:.2f} to {max_val:.2f}")
            except:
                print(f"   - Could not analyze column: {col}")
    
    # Correlations between metrics
    if len(labor_cols) > 1:
        try:
            corr = df[labor_cols].corr()
            print(f"\n5. CORRELATIONS BETWEEN METRICS:")
            # Print only high correlations (absolute value > 0.5)
            high_corrs = []
            for i in range(len(labor_cols)):
                for j in range(i+1, len(labor_cols)):
                    corr_val = corr.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        high_corrs.append((labor_cols[i], labor_cols[j], corr_val))
            
            if high_corrs:
                for col1, col2, corr_val in high_corrs[:5]:  # Show top 5 correlations
                    print(f"   - {col1} and {col2}: {corr_val:.2f}")
            else:
                print("   - No strong correlations found between the analyzed metrics")
        except:
            print("   - Could not compute correlations")
    
    print("\nAnalysis of IMF labor statistics complete!")

def main():
    # Directory containing CSV files
    #directory = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2'
    
    # Find all CSV files
    #csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    #if not csv_files:
    #    print(f"No CSV files found in {directory}")
    #    return
    
    #print(f"Found {len(csv_files)} CSV files in the directory.")
    
    # Analyze each CSV file
    #for file_path in csv_files:
    #    analyze_csv_file(file_path)
    
    # Run specific analysis on IMF labor statistics
    #print("\nRunning detailed analysis on IMF labor statistics file...")
    #analyze_imf_labor_statistics()
    
    #print("\nAnalysis complete!")
    pass

if __name__ == "__main__":
    main()
