import os
import pandas as pd

def summarize_predictions(prediction_data, epoch=5):
    prediction_data['Date'] = prediction_data['Time'].dt.date
    df = prediction_data.groupby(["Date", "Prediction"])['Prediction'].count().mul(epoch).div(60).reset_index(name='Min')

    return df.pivot(index='Date', columns='Prediction', values='Min')


def merge_summary_files(output_folder):
    # Define file paths
    wear_summary_path = os.path.join(output_folder, "wear_daily_summary.csv")
    by_day_summary_path = os.path.join(output_folder, "by_day_by_participants.csv")
    final_summary_path = os.path.join(output_folder, "FinalSummaryByParticipant.csv")
    
    if os.path.exists(wear_summary_path) and os.path.exists(by_day_summary_path):
        
        wear_summary = pd.read_csv(wear_summary_path)
        by_day_summary = pd.read_csv(by_day_summary_path)
        
        # Merge on 'studyid' and 'Date' (assuming these columns exist in both files)
        final_summary = pd.merge(wear_summary, by_day_summary, on=['studyid', 'Date'], how='outer')
        
        # Save the final merged summary
        final_summary.to_csv(final_summary_path, index=False)