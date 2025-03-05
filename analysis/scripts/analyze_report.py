import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def load_report(report_path: str) -> dict:
    """Load the analysis report"""
    with open(report_path, 'r') as f:
        return json.load(f)

def analyze_table_sizes(data: dict) -> pd.DataFrame:
    """Analyze and sort tables by size"""
    table_sizes = []
    for name, info in data['table_sizes'].items():
        table_sizes.append({
            'table_name': name,
            'rows': info['row_count'],
            'columns': info['column_count'],
            'size_mb': info['memory_usage_mb']
        })
    
    df = pd.DataFrame(table_sizes)
    return df.sort_values('rows', ascending=False)

def analyze_relationships(data: dict) -> List[Tuple[str, str, List[str]]]:
    """Analyze table relationships"""
    relationships = []
    for key, columns in data['relationships'].items():
        # Split on the last underscore to handle table names that contain underscores
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            table1, table2 = parts
            relationships.append((table1, table2, columns))
    
    return sorted(relationships, key=lambda x: len(x[2]), reverse=True)

def analyze_data_quality(data: dict) -> Dict[str, Dict[str, int]]:
    """Analyze data quality issues"""
    quality_issues = {}
    
    for table_name, table_info in data.items():
        if table_name not in ['table_sizes', 'relationships']:
            missing_columns = []
            data_type_issues = []
            
            for col_name, col_info in table_info['columns'].items():
                if col_info['missing_percentage'] > 0:
                    missing_columns.append({
                        'column': col_name,
                        'missing_percentage': col_info['missing_percentage']
                    })
            
            if missing_columns or data_type_issues:
                quality_issues[table_name] = {
                    'missing_data': missing_columns
                }
    
    return quality_issues

def identify_key_tables(size_df: pd.DataFrame) -> List[str]:
    """Identify key tables based on size and content"""
    # Focus on fact tables (prefix F_) and large dimension tables
    fact_tables = size_df[size_df['table_name'].str.contains('_F_', na=False)]
    large_dim_tables = size_df[
        (size_df['table_name'].str.contains('_D_', na=False)) & 
        (size_df['rows'] > 1000)
    ]
    
    return list(pd.concat([fact_tables, large_dim_tables])['table_name'])

def print_insights(report_path: str):
    """Print key insights from the analysis"""
    data = load_report(report_path)
    
    # Analyze table sizes
    size_df = analyze_table_sizes(data)
    print("\n=== Largest Tables ===")
    print(size_df.head(10).to_string(index=False))
    
    # Identify key tables
    key_tables = identify_key_tables(size_df)
    print("\n=== Key Tables for Analysis ===")
    for table in key_tables[:10]:
        print(f"- {table}")
    
    # Analyze relationships
    relationships = analyze_relationships(data)
    print("\n=== Strong Table Relationships ===")
    for t1, t2, cols in relationships[:10]:
        print(f"{t1} <-> {t2}: {len(cols)} shared columns")
        print(f"Shared columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
    
    # Analyze data quality
    quality_issues = analyze_data_quality(data)
    print("\n=== Data Quality Issues ===")
    for table, issues in list(quality_issues.items())[:10]:
        missing = issues.get('missing_data', [])
        if missing:
            print(f"\n{table}:")
            print("Missing data in columns:")
            for col in sorted(missing, key=lambda x: x['missing_percentage'], reverse=True)[:5]:
                print(f"  - {col['column']}: {col['missing_percentage']:.1f}% missing")

def main():
    # Find the most recent report
    reports_dir = Path("analysis/reports")
    reports = list(reports_dir.glob("analysis_report_*.json"))
    latest_report = max(reports, key=lambda x: x.stat().st_mtime)
    
    print(f"Analyzing report: {latest_report}")
    print_insights(latest_report)

if __name__ == "__main__":
    main() 