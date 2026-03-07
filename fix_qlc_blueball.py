import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_blue_ball(file_path):
    """
    Reads a lottery CSV file, populates the '蓝球' column
    with data from 'backWinningNum', and saves the file back.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    logging.info(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path)

        # Check for required columns
        if 'backWinningNum' not in df.columns:
            logging.error(f"'backWinningNum' column not found in {file_path}. Cannot proceed.")
            return

        logging.info("Copying 'backWinningNum' to '蓝球' column...")
        
        def parse_blue(val):
            try:
                if pd.isna(val): return 0
                s = str(val).strip()
                if not s: return 0
                # Handle cases where it might be space-separated, take the first one
                parts = s.split()
                return int(parts[0]) if parts else 0
            except: return 0

        df['蓝球'] = df['backWinningNum'].apply(parse_blue)

        # Save the DataFrame back to the original file
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Successfully updated '蓝球' column in {file_path}")
        
    except Exception as e:
        logging.error(f"An error occurred while processing {file_path}: {e}")

if __name__ == "__main__":
    # This script will specifically target the '七乐彩' data file as requested.
    qlc_file_path = os.path.join('data', '七乐彩_lottery_data.csv')
    fix_blue_ball(qlc_file_path)

    logging.info("Script finished.")
