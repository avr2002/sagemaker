"""
Script to download Palmer Penguins dataset from GitHub.

This script downloads the Palmer Penguins dataset from the official GitHub repository
and renames the bill_* columns to culmen_* columns as requested.
"""

import sys
from pathlib import Path

import pandas as pd

from penguins.consts import DATA_DIR


def download_penguins_data() -> pd.DataFrame:
    """
    Download Palmer Penguins data from the official GitHub repository.
    
    :return: DataFrame with penguin data
    :raises: Exception if download fails
    """
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    
    print(f"Downloading Palmer Penguins data from: {url}")
    
    try:
        df = pd.read_csv(url)
        print(f"Successfully downloaded {len(df)} records")
        return df
    except Exception as e:
        raise Exception(f"Failed to download data: {e}")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename bill_* columns to culmen_* columns.
    
    :param df: Input DataFrame
    :return: DataFrame with renamed columns
    """
    column_mappings = {
        'bill_length_mm': 'culmen_length_mm',
        'bill_depth_mm': 'culmen_depth_mm'
    }
    
    df_renamed = df.rename(columns=column_mappings)
    print(f"Renamed columns: {list(column_mappings.keys())} -> {list(column_mappings.values())}")
    
    return df_renamed


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the DataFrame to a CSV file.
    
    :param df: DataFrame to save
    :param output_path: Path where to save the file
    :raises: Exception if saving fails
    """
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info about the dataset
        print("\nDataset overview:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        raise Exception(f"Failed to save data: {e}")


def main():
    """Main function to download Palmer Penguins dataset."""
    print("Palmer Penguins Dataset Downloader")
    print("=" * 40)
    
    # Define output path
    output_path = DATA_DIR / "penguins.csv"
    
    try:
        # Download the data
        df = download_penguins_data()
        
        # Rename columns
        df = rename_columns(df)
        
        # Save the data
        save_data(df, output_path)
        
        print(f"\nSUCCESS: Palmer Penguins dataset downloaded and saved to {output_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
