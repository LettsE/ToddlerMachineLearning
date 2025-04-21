import os
import pandas as pd
import numpy as np
from itertools import groupby



def process_nonwear_times(data, nonwear_method, logbook_file, output_folder, studyid):
    """
    Processes non-wear detection based on the selected method.

    Parameters:
        - data: DataFrame containing raw accelerometer data.
        - selected_data: Dictionary with the selected non-wear method.
        - output_folder: Path to save the output CSVs.
        - studyid: Identifier for the participant.

    Returns:
        - daily_summary: DataFrame with daily wear and non-wear durations.
        - trimmed_data: DataFrame with only wear time data.
    """
    non_wear_method = nonwear_method
    logbook_file = logbook_file
    wear_time_csv_path = os.path.join(output_folder, "all_wear_times.csv")

    if os.path.exists(wear_time_csv_path):
        all_wear_times_df = pd.read_csv(wear_time_csv_path, parse_dates=["WearTimeStart", "WearTimeEnd"])
    else:
        all_wear_times_df = None

    
    if non_wear_method == "Logbook":
        if logbook_file and os.path.exists(logbook_file):
            logbook_df = pd.read_csv(logbook_file, parse_dates=["WearTimeStart", "WearTimeEnd"])
            wear_time_df = logbook_df[logbook_df["studyid"] == studyid].copy()
        else:
            return None, data  # No processing, return raw data

    else:
        return None, data  # No changes to raw data


    # Append to the all-study wear time dataset
    if all_wear_times_df is None:
        all_wear_times_df = wear_time_df
    else:
        all_wear_times_df = pd.concat([all_wear_times_df, wear_time_df], ignore_index=True)


    # Save the updated wear times for all participants
    all_wear_times_df.to_csv(wear_time_csv_path, index=False)

    # Compute daily wear and non-wear durations
    wear_time_summary = wear_time_df.copy()
    wear_time_summary["WearDuration"] = (wear_time_summary["WearTimeEnd"] - wear_time_summary["WearTimeStart"]).dt.total_seconds() / 60
    wear_time_summary["Date"] = wear_time_summary["WearTimeStart"].dt.date

    daily_summary = wear_time_summary.groupby(["studyid", "Date"]).agg({"WearDuration": "sum"}).reset_index()
    daily_summary["NonWearDuration"] = 1440 - daily_summary["WearDuration"]

    summary_csv_path = os.path.join(output_folder, "wear_daily_summary.csv")
    if os.path.exists(summary_csv_path):
        existing_summary = pd.read_csv(summary_csv_path)
        daily_summary = pd.concat([existing_summary, daily_summary], ignore_index=True)
    
    daily_summary.to_csv(summary_csv_path, index=False)

    # Trim raw data based on wear time
    trimmed_data = pd.DataFrame()
    
    for _, row in wear_time_df.iterrows():
        mask = (data["Datetime"] >= row["WearTimeStart"]) & (data["Datetime"] <= row["WearTimeEnd"])
        trimmed_data = pd.concat([trimmed_data, data[mask]])

    trimmed_data_csv_path = os.path.join(output_folder, f"{studyid}_trimmed_data.csv")
    trimmed_data.to_csv(trimmed_data_csv_path, index=False)

    return daily_summary, trimmed_data