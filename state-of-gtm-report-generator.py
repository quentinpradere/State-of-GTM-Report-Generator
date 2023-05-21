import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import datetime
from datetime import date, timedelta
from pandas import Timestamp
from builtins import KeyError
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import Tuple, List, Union
from pathlib import Path


today = datetime.date.today()


@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)

@st.cache_data
def enhance_salesforce_report(df: pd.DataFrame) -> pd.DataFrame:
    
    """Clean and enhance a dataframe.

    Args:
        df: Input dataframe to be cleaned.
        snowflake_dataset: Dataset used for merging.
        product_mapping: Dataset used for product mapping.

    Returns:
        A cleaned and enhanced DataFrame.
    """

    datetime_columns = ["Close Date", "Date: Moved to Develop", "Date: Moved to Discovery", 
                        "Date: Moved to Business Alignment", "Date: Moved to Validate", 
                        "Date: Moved to Propose", "Date: Moved to Negotiate", 
                        "Date: Moved to Won", "Date: Moved to Closed Won", 
                        "Date: Moved to Closed Lost", "Date: Moved to Dead"]

    # Convert columns to datetime
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])


    def update_dates(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
        """Update dates of specified columns.

        Args:
            df: Input dataframe.
            col_list: List of column names.

        Returns:
            Updated DataFrame.
        """
        for i in range(len(col_list) - 1):
            df[col_list[i + 1]] = df[col_list[i + 1]].where(df[col_list[i + 1]].notna(), df[col_list[i]])
        df.loc[df['Renewal?'] != 'No', 'Date: Moved to Discovery'] = df.loc[df['Renewal?'] != 'No', 'Date: Moved to Business Alignment']
        return df

    def get_furthest_stage(row: pd.Series) -> str:
        """Get the furthest stage for a given row.

        Args:
            row: Input series.

        Returns:
            Furthest stage.
        """
        stages = ['Date: Moved to Closed Won', 'Date: Moved to Won', 'Date: Moved to Negotiate', 'Date: Moved to Propose', 'Date: Moved to Validate', 'Date: Moved to Business Alignment', 'Date: Moved to Discovery']
        for stage in stages:
            if not pd.isna(row[stage]): 
                return stage.split(": ")[-1]
        return "Error"

    def fiscal_quarter(date: Union[pd.Timestamp, None]) -> str:
        """Get fiscal quarter of a given date.

        Args:
            date: Input date.

        Returns:
            Fiscal quarter.
        """
        if date is None or pd.isnull(date):
            return None
        quarter_start_months = {1: 'Q4', 2: 'Q1', 5: 'Q2', 8: 'Q3', 11: 'Q4'}
        quarter_start_month = max(m for m in quarter_start_months if m <= date.month)
        quarter = quarter_start_months[quarter_start_month]
        year = date.year if date.month == 1 else date.year + 1
        return f'FY{year}-{quarter}'

    def deal_duration(*dates: pd.Timestamp, original_date: pd.Timestamp) -> pd.Timedelta:
        """Compute the deal duration based on given dates.

        Args:
            dates: Various relevant dates.
            original_date: The original date to compare with.

        Returns:
            Deal duration.
        """
        if pd.isna(original_date):
            return pd.NaT
        else:
            relevant_dates = [d for d in dates if isinstance(d, pd.Timestamp)]
            if not relevant_dates:
                return pd.Timestamp.today() - original_date
            else:
                closest_date = min(relevant_dates, key=lambda d: abs(d - original_date))
                return closest_date - original_date

    # Continue applying the above functions
    date_cols = ['Date: Moved to Closed Won', 'Date: Moved to Won', 'Date: Moved to Negotiate', 'Date: Moved to Propose', 'Date: Moved to Validate', 'Date: Moved to Business Alignment', 'Date: Moved to Discovery', 'Date: Moved to Develop']
    df = update_dates(df, date_cols)

    df['Close Date - Month'] = df['Close Date'].dt.strftime('%B')
    df['Date: Moved to Discovery - Month'] = df['Date: Moved to Discovery'].dt.strftime('%B')
    df['Date: Moved to Validate - Month'] = df['Date: Moved to Validate'].dt.strftime('%B')
    df['Close Date - Fiscal Quarter'] = df['Close Date'].apply(fiscal_quarter)
    df['Date: Moved to Discovery - Fiscal Quarter'] = df['Date: Moved to Discovery'].apply(fiscal_quarter)
    df['Date: Moved to Validate - Fiscal Quarter'] = df['Date: Moved to Validate'].apply(fiscal_quarter)
    df['Close Date - Fiscal Year'] = df['Close Date - Fiscal Quarter'].str[:6]
    df['Date: Moved to Discovery - Fiscal Year'] = df['Date: Moved to Discovery - Fiscal Quarter'].str[:6]
    df['Date: Moved to Validate - Fiscal Year'] = df['Date: Moved to Validate - Fiscal Quarter'].str[:6]

    df['Total Opp Age'] = df.apply(lambda x: deal_duration(x['Date: Moved to Closed Won'], x['Date: Moved to Closed Lost'], x['Date: Moved to Dead'], original_date=x["Date: Moved to Discovery"]), axis=1)
    df['Total Opp Age'] = df['Total Opp Age'].dt.total_seconds() / (60 * 60 * 24)  # Convert to number of days

    df['Qualified Opp Age'] = df.apply(lambda x: deal_duration(x['Date: Moved to Closed Won'], x['Date: Moved to Closed Lost'], x['Date: Moved to Dead'], original_date=x["Date: Moved to Validate"]), axis=1)
    df['Qualified Opp Age'] = df['Qualified Opp Age'].dt.total_seconds() / (60 * 60 * 24)  # Convert to number of days

    df['Furthest Stage'] = df.apply(get_furthest_stage, axis=1)

    df['Channel'] = df['Channel'].replace({'EMEA': 'EMEA Reseller', 'EMEA Southern & Central Enterprise': 'EMEA Southern Enterprise'})   

    def update_channel_and_pod(row: pd.Series) -> Tuple[str, str]:
        """Update channel and pod information.

        Args:
            row: Input series.

        Returns:
            Updated channel and pod.
        """
        group_to_channel_dict = {'Southern Europe': 'EMEA Southern Enterprise', 'Central Europe': 'EMEA Central Enterprise', 'Northern Europe': 'EMEA Northern Mid-Market', 'NA Mid-Market': 'NA Mid-Market', 'NA SMB': 'NA SMB', 'GEOs': 'NA Enterprise', 'Verticals': 'NA Enterprise'}
        
        if pd.isnull(row['Channel']) or row['Channel'] in ['Do Not Use', 'EMEA']:
            if 'PAR LC' in row['Opportunity Record Type']:
                if 'International' in row['Opportunity Record Type']:
                    return ('INT Reseller', 'EMEA Reseller')
                else:
                    return ('NA Reseller', 'NA Enterprise')
            elif 'Partner' in row['Split Owner Role']:
                if 'International' in row['Opportunity Record Type']:
                    return ('INT Reseller', 'EMEA Reseller')
                else:
                    return ('NA Reseller', 'NA Enterprise')
            elif 'Mid-ENT' in row['Split Owner Role'] or 'Mid-Enterprise' in row['Split Owner Role']:
                return ('Unknown', 'NA Enterprise')
            elif pd.isna(row['Account Group']) or row['Account Group'] not in group_to_channel_dict:
                return ('Unknown', 'Unknown')
            else:
                return ('Unknown', group_to_channel_dict[row['Account Group']])
        else:
            return (row['Pod'], row['Channel'])
        
    df[['Pod', 'Channel']] = df.apply(update_channel_and_pod, axis=1, result_type='expand')
    
    def get_opp_status(stage):
        if stage == 'Closed Won':
            return 'Closed Won'
        elif stage in ['Closed Lost', 'Dead']:
            return 'Closed Lost/Dead'
        else:
            return 'Open'
    
    df['Opportunity Status'] = df['Stage'].apply(get_opp_status)
    
    # Define bin edges.
    bins = [0, 25000, 50000, 100000, 500000, 1000000, np.inf]

    # Define bin labels.
    labels = ['<$25k', '$25k-$50k', '$50k-$100k', '$100k-$500k', '$500k-$1M', '$1M+']

    # Create a new column in the DataFrame with the binned data.
    df['Deal Size Bucket'] = pd.cut(df['Amount (converted)'], bins=bins, labels=labels)

    df['New Logo?'] = df['New Logo?'].replace({1: 'New Logo', 0: 'Upsell'})
    
    return df

def process_data(data, values, index, column, aggfunc, start_date, end_date):

    # Extract the date column from the first element in columns
    DateColumn = column.split(' -')[0]

    filtered_data = data[(data[DateColumn] >= start_date) & (data[DateColumn] <= end_date)]

    if values == 'Opportunity Count':
        values = '18 Digit Opty ID'
    
    pivot_data = filtered_data.pivot_table(values=values, columns=column, fill_value = 0,
                                           index=index, aggfunc=aggfunc, margins=True, margins_name = 'Total')
    
    if values == '18 Digit Opty ID' or 'Opp Age' in values:
        pivot_data = pivot_data.applymap(lambda x: '{:,.0f}'.format(x) if not np.isnan(x) else np.nan)
    elif values == 'Amount (converted)':
        pivot_data = pivot_data.applymap(lambda x: '${:,.0f}'.format(x) if not np.isnan(x) else np.nan)


    if 'Channel' in index:
        ChannelOrder = ['NA Enterprise', 'NA Mid-Market', 'NA SMB', 'CPU', 'EMEA Northern Enterprise', 'EMEA Central Enterprise'
                        ,'EMEA Southern Enterprise', 'EMEA Northern Mid-Market', 'EMEA Reseller', 'APAC', 'Total']
        pivot_data = pivot_data.reindex(ChannelOrder, level='Channel')

    # Drop the rows where all values are null
    pivot_data = pivot_data.dropna(how='all')

    # If only one aggfunc or values is selected, rename the columns to hide aggfunc and values.
    if len(aggfunc) == 1 and len(values) == 1:
        pivot_data.columns = pivot_data.columns.droplevel([0, 1])

    return pivot_data



def fiscal_dates():
    today = pd.Timestamp.today()

    # fiscal year start months (Feb, May, Aug, Nov)
    fiscal_quarters_start = {1: 11, 2: 2, 3: 2, 4: 2, 5: 5, 6: 5, 7: 5, 8: 8, 9: 8, 10: 8, 11: 11, 12: 11}

    # get the start month of the current fiscal quarter
    start_month_this_year = fiscal_quarters_start[today.month]

    # The start date should be the start of the previous fiscal quarter of last year
    start_month_last_year = start_month_this_year - 3 if start_month_this_year > 3 else start_month_this_year + 9
    start_date = pd.Timestamp(today.year - 1, start_month_last_year, 1)

    # The end date should be the end of the previous fiscal quarter of this year
    end_month_this_year = start_month_this_year - 1 if start_month_this_year > 1 else start_month_this_year + 11
    end_year = today.year if start_month_this_year > 1 else today.year - 1
    end_day_this_year = pd.Timestamp(end_year, end_month_this_year, 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)

    return start_date, end_day_this_year


def append_df_to_gsheet(df, gsheet_url, tab_title, json_key='/Users/qpradere/Documents/Personal/Streamlit_Test/state-of-gtm_google-credentials.json'):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # Give the path to the Service Account Credential json file 
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_key, scope)
    
    # Authorise your Notebook
    gc = gspread.authorize(credentials)

    # Extract spreadsheet ID from the gsheet_url and open it
    spreadsheet_id = gsheet_url.split('/')[-2]
    spreadsheet = gc.open_by_key(spreadsheet_id)

    # get the first sheet of the Google Sheet
    worksheet = spreadsheet.add_worksheet(title=tab_title,  rows="1000", cols="100")

    # Get the data including index and columns
    index_names = list(df.index.names)  # get the index names
    column_names = df.columns.tolist()  # get the column names
    data_header = index_names + column_names  # combine index names and column names
    data_rows = df.reset_index().values.tolist()  # get the data rows
    data = [data_header] + data_rows  # combine the header and rows

    # Append the data to the worksheet
    for i, row in enumerate(data):
        worksheet.append_row(row)

    return "Data appended successfully"

def app():
    st.header('State of Go-To-Market Resource Builder')
    with st.expander("How to Use This"):
        st.write(Path("README.md").read_text())
    
    csv_file = st.file_uploader("Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False)

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'save_to_gsheet' not in st.session_state:  # Initialize save_to_gsheet
        st.session_state.save_to_gsheet = False

    if csv_file:
        if 'csv_file_name' not in st.session_state or st.session_state.csv_file_name != csv_file.name:
            data = load_data(csv_file)
            enhanced_data = enhance_salesforce_report(data)
            st.session_state.data = enhanced_data
            st.session_state.csv_file_name = csv_file.name


    if st.session_state.data is not None and not st.session_state.data.empty:
        csv_data = st.session_state.data.to_csv(index=False).encode('utf-8')

        # Get unique values for each filter column and create multiselect filters
        unique_channels = st.session_state.data['Channel'].unique().tolist()
        unique_pods = st.session_state.data['Pod'].unique().tolist()
        unique_account_types = st.session_state.data['Account Type'].unique().tolist()
        unique_opportunity_statuses = st.session_state.data['Opportunity Status'].unique().tolist()
        
        default_channels = ['NA Enterprise', 'NA Mid-Market', 'EMEA Northern Enterprise', 'EMEA Central Enterprise', 'EMEA Southern Enterprise', 'EMEA Northern Mid-Market', 'EMEA Reseller']
        selected_channels = st.sidebar.multiselect("Channel", unique_channels, default=default_channels)
        selected_pods = st.sidebar.multiselect("Pod", unique_pods)
        selected_account_types = st.sidebar.multiselect("Account Type", unique_account_types)
        selected_opportunity_statuses = st.sidebar.multiselect("Opportunity Status", unique_opportunity_statuses)

        # Create a copy of the original dataframe to keep the original intact
        filtered_data = st.session_state.data.copy()

        # Filter the dataframe based on the selected values
        if selected_channels:
            filtered_data = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        if selected_pods:
            filtered_data = filtered_data[filtered_data['Pod'].isin(selected_pods)]
        if selected_account_types:
            filtered_data = filtered_data[filtered_data['Account Type'].isin(selected_account_types)]
        if selected_opportunity_statuses:
            filtered_data = filtered_data[filtered_data['Opportunity Status'].isin(selected_opportunity_statuses)]


        st.download_button(
            label="Download Enhanced Dataset as CSV",
            data=csv_data,
            file_name="enhanced_data.csv",
            mime="text/csv",
        )


        # Create a placeholder for the button
        button_placeholder = st.empty()

        # Display pivot section and 'Close Pivot Section' button if pivot section is visible
        if st.session_state.get('pivot_section_visible', False):
            if button_placeholder.button('Close Pivot Section'):
                st.session_state.pivot_section_visible = False

        # Display 'Create Trends Pivot Table' button if pivot section is not visible
        if not st.session_state.get('pivot_section_visible', False):
            if button_placeholder.button('Create Trends Pivot Table'):
                st.session_state.pivot_section_visible = True
        
        if st.session_state.get('pivot_section_visible', False):

            col1, col2, col3 = st.columns(3)

            with col1:
                values = st.selectbox('Values',
                                        ['Opportunity Count', 'Amount (converted)', 'Total Opp Age', 'Qualified Opp Age'],
                                        index=0)

            with col2:
                index = st.multiselect('Rows',
                                        ['Channel', 'Account Group', 'Pod', 'User', 'Industry Alignment', 'New Logo?', 'Deal Size Bucket'],
                                        ['Channel'])

            with col3:
                if values == 'Opportunity Count':
                    aggfunc = st.selectbox('Function', ['count'])
                else:  # values == 'Amount (converted)'
                    aggfunc = st.selectbox('Function', ['sum', 'mean', 'median'], index=0)

            col4, col5 = st.columns([3, 2])

            with col4:
                start_date, end_date = st.date_input("Select a date range", fiscal_dates())

            with col5:
                column = st.selectbox('Column',
                                        ['Close Date - Month', 'Close Date - Fiscal Quarter', 'Close Date - Fiscal Year', 
                                         'Date: Moved to Discovery - Month', 'Date: Moved to Discovery - Fiscal Quarter', 'Date: Moved to Discovery - Fiscal Year',
                                         'Date: Moved to Validate - Month','Date: Moved to Validate - Fiscal Quarter', 'Date: Moved to Validate - Fiscal Year'], 
                                      index=1)
            
            if start_date and end_date:
                st.session_state.start_date = start_date
                st.session_state.end_date = end_date

                start_date_pd = Timestamp(st.session_state.start_date)
                end_date_pd = Timestamp(st.session_state.end_date)
                try:
                    pivot_data = process_data(filtered_data, values, index, column, aggfunc, start_date_pd, end_date_pd)
                    # Create two columns for the buttons
                    col1, col2 = st.columns(2)
                    
                    # Download button in the first column
                    with col1:
                        pivot_csv_data = pivot_data.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Pivot as CSV",
                            data=pivot_csv_data,
                            file_name="pivot_table.csv",
                            mime="text/csv",
                        )
                    
                    # Save to Google Sheets button in the second column
                    with col2:
                        if st.button('Upload Pivot to Google Sheets'):
                            st.session_state.save_to_gsheet = True

                        if st.session_state.save_to_gsheet:
                            tab_title = st.text_input('Enter the title for the new tab:', 'State of GTM Pivot')
                            gsheet_url = st.text_input('Enter the URL of the Google Sheets document:')

                            if gsheet_url and tab_title:
                                try:
                                    append_df_to_gsheet(pivot_data, gsheet_url, tab_title)
                                    st.success('Data successfully saved to Google Sheets')
                                    st.session_state.save_to_gsheet = False  # Reset the state after successful save
                                    st.session_state.go = False  # Reset the go state after successful save
                                except Exception as e:
                                    st.error(f"Error saving data to Google Sheets: {str(e)}")

                    # Display the dataframe after the buttons
                    st.dataframe(pivot_data)

                except KeyError as e:
                    st.error(f"Error processing data. It appears the column {str(e)} is not present in your data.")

if __name__ == "__main__":
    app()
