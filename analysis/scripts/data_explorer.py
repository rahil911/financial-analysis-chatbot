import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DataExplorer:
    def __init__(self, db_path: str, output_path: str):
        """
        Initialize the DataExplorer with paths for data and output
        
        Args:
            db_path: Path to the directory containing CSV files
            output_path: Path to save analysis results
        """
        self.db_path = Path(db_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store DataFrames
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # Dictionary to store analysis results
        self.analysis_results: Dict[str, dict] = {}

    def load_data(self, file_size_limit_mb: int = 500) -> None:
        """
        Load CSV files from the DB directory
        
        Args:
            file_size_limit_mb: Maximum file size to load in MB
        """
        print("Loading data files...")
        
        for file_path in self.db_path.glob("*.csv"):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > file_size_limit_mb:
                print(f"Skipping {file_path.name} (Size: {file_size_mb:.2f}MB > {file_size_limit_mb}MB)")
                continue
                
            try:
                print(f"Loading {file_path.name}...")
                df = pd.read_csv(file_path, low_memory=False)
                self.dataframes[file_path.stem] = df
                print(f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"Error loading {file_path.name}: {str(e)}")

    def analyze_data_quality(self) -> None:
        """Analyze data quality for each DataFrame"""
        print("\nAnalyzing data quality...")
        
        for name, df in self.dataframes.items():
            print(f"\nAnalyzing {name}...")
            
            analysis = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "columns": {}
            }
            
            for col in df.columns:
                col_analysis = {
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percentage": float(df[col].isna().sum() / max(len(df), 1)) * 100,
                    "unique_count": int(df[col].nunique()),
                    "unique_percentage": float(df[col].nunique() / max(len(df), 1)) * 100 if len(df) > 0 else 0
                }
                
                # Add basic statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]) and len(df) > 0:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        col_analysis.update({
                            "mean": float(non_null_values.mean()),
                            "std": float(non_null_values.std()),
                            "min": float(non_null_values.min()),
                            "max": float(non_null_values.max()),
                            "median": float(non_null_values.median())
                        })
                
                analysis["columns"][col] = col_analysis
            
            self.analysis_results[name] = analysis

    def analyze_table_sizes(self) -> None:
        """Analyze table sizes and data distributions"""
        print("\nAnalyzing table sizes...")
        
        table_sizes = {}
        for name, df in self.dataframes.items():
            size_info = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "non_null_counts": df.count().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            table_sizes[name] = size_info
        
        self.analysis_results["table_sizes"] = table_sizes

    def identify_relationships(self) -> None:
        """Identify potential relationships between datasets based on column names"""
        print("\nIdentifying relationships between datasets...")
        
        relationships = {}
        
        # Create a mapping of column names to datasets
        column_to_datasets = {}
        for dataset_name, df in self.dataframes.items():
            for column in df.columns:
                if column not in column_to_datasets:
                    column_to_datasets[column] = []
                column_to_datasets[column].append(dataset_name)
        
        # Find datasets with common columns
        for column, datasets in column_to_datasets.items():
            if len(datasets) > 1:
                for i in range(len(datasets)):
                    for j in range(i + 1, len(datasets)):
                        key = f"{datasets[i]}_{datasets[j]}"
                        if key not in relationships:
                            relationships[key] = []
                        relationships[key].append(column)
        
        self.analysis_results["relationships"] = relationships

    def generate_summary_report(self) -> None:
        """Generate a summary report of the analysis"""
        print("\nGenerating summary report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_path / f"analysis_report_{timestamp}.json"
        
        # Convert all numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        # Convert the analysis results
        serializable_results = json.loads(
            json.dumps(self.analysis_results, default=convert_to_serializable)
        )
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Report saved to {report_path}")

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline"""
        self.load_data()
        self.analyze_data_quality()
        self.analyze_table_sizes()
        self.identify_relationships()
        self.generate_summary_report()

def main():
    # Initialize paths
    db_path = Path("DB")
    output_path = Path("analysis/reports")
    
    # Create and run the explorer
    explorer = DataExplorer(db_path, output_path)
    explorer.run_analysis()

if __name__ == "__main__":
    main() 