import streamlit as st
import pandas as pd
import redis
import plotly.express as px
from datetime import datetime, timedelta
import json
import threading
import time
import numpy as np # For handling inf/nan in calculations

# --- Configuration (should match backend scraper.py) ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 3
REDIS_UPDATE_CHANNEL = 'data_updated'
DEFAULT_DAYS_TO_LOAD_FOR_LINE_CHART = 90 # Load more days for the line chart initially

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="HKEX Shareholding Dashboard")

# --- Redis Connection ---
@st.cache_resource # Cache the Redis connection resource
def get_redis_connection():
    """Establishes and returns a Redis connection."""
    try:
        r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        r.ping()
        st.session_state.redis_connected = True
        return r
    except redis.exceptions.ConnectionError as e:
        st.session_state.redis_connected = False
        st.error(f"Failed to connect to Redis: {e}. Please ensure Redis is running and accessible.")
        return None

# --- Data Loading and Processing Functions ---
def deserialize_df_from_redis(json_bytes):
    """Deserializes a JSON string (from Redis bytes) into a Pandas DataFrame."""
    if json_bytes:
        try:
            # The JSON string is stored as bytes, so decode to utf-8 first
            json_string = json_bytes.decode('utf-8')
            df = pd.read_json(json_string, orient='split')
            # Ensure 'ScrapeDate' is datetime
            if 'ScrapeDate' in df.columns:
                df['ScrapeDate'] = pd.to_datetime(df['ScrapeDate'])
            # Ensure numeric columns are indeed numeric
            if 'Shareholding in CCASS' in df.columns:
                df['Shareholding in CCASS'] = pd.to_numeric(df['Shareholding in CCASS'], errors='coerce')
            if 'Stock Code' in df.columns:
                df['Stock Code'] = pd.to_numeric(df['Stock Code'], errors='coerce')
            if '% of the total number of Issued Shares/Units' in df.columns:
                df['% of the total number of Issued Shares/Units'] = pd.to_numeric(df['% of the total number of Issued Shares/Units'], errors='coerce')

            return df
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from Redis: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing DataFrame from Redis: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner="Fetching available dates from Redis...") # Cache for 5 minutes
def get_available_dates_sorted(_r_conn_placeholder): # Pass a placeholder to trigger rerun on conn change
    """
    Fetches all 'YYYY-MM-DD' keys from Redis, sorts them, and returns them.
    The placeholder helps in cache invalidation if Redis connection status changes.
    """
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return []
    try:
        date_keys_bytes = r.keys("????-??-??") # Pattern for YYYY-MM-DD
        date_keys_str = sorted([key.decode('utf-8') for key in date_keys_bytes], reverse=True)
        # Validate format strictly
        valid_date_keys = []
        for k in date_keys_str:
            try:
                datetime.strptime(k, "%Y-%m-%d")
                valid_date_keys.append(k)
            except ValueError:
                pass # ignore keys not matching the format
        return valid_date_keys
    except Exception as e:
        st.warning(f"Error fetching date keys from Redis: {e}")
        return []

@st.cache_data(ttl=60, show_spinner="Fetching data for date: {date_key_str}...") # Cache for 1 minute
def fetch_df_from_redis_cached(date_key_str, _r_conn_placeholder):
    """Cached function to fetch and deserialize a single DataFrame from Redis."""
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False) or not date_key_str:
        return pd.DataFrame()
    try:
        json_data = r.get(date_key_str)
        if json_data:
            return deserialize_df_from_redis(json_data)
        # Removed st.warning from here to avoid spamming if a date legitimately has no data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data for {date_key_str} from Redis: {e}")
        return pd.DataFrame()

def get_stock_options(df):
    """Generates a list of 'Stock Name (Stock Code)' for selectbox."""
    if df is not None and not df.empty and 'Name' in df.columns and 'Stock Code' in df.columns:
        # Ensure no NaN values in Name or Stock Code for options
        df_cleaned = df.dropna(subset=['Name', 'Stock Code'])
        # Ensure Stock Code is int for display
        df_cleaned['Stock Code'] = df_cleaned['Stock Code'].astype(int)
        options = sorted(list(set(df_cleaned['Name'] + " (" + df_cleaned['Stock Code'].astype(str) + ")")))
        return options
    return ["No stocks available"]

def get_historical_data_for_stock(selected_stock_code, available_dates, num_days):
    """
    Fetches historical data for a specific stock over a number of days.
    """
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return pd.DataFrame()

    stock_data_list = []
    # Ensure we don't try to fetch more dates than available
    dates_to_fetch = available_dates[:min(num_days, len(available_dates))] 

    for date_str in dates_to_fetch:
        daily_df = fetch_df_from_redis_cached(date_str, st.session_state.get('redis_connected'))
        if not daily_df.empty and 'Stock Code' in daily_df.columns:
            # Ensure selected_stock_code is of the same type as in DataFrame for comparison
            stock_specific_data = daily_df[daily_df['Stock Code'] == int(selected_stock_code)]
            if not stock_specific_data.empty:
                stock_data_list.append(stock_specific_data)
    
    if not stock_data_list:
        return pd.DataFrame()
    
    full_stock_history = pd.concat(stock_data_list, ignore_index=True)
    if 'ScrapeDate' not in full_stock_history.columns:
         # Attempt to infer ScrapeDate from the key if it's missing in the df (should ideally be there)
         # This is a fallback, the scraper should ensure ScrapeDate is in the stored DF.
         st.warning("ScrapeDate column missing in concatenated historical data. This might affect chart accuracy.")
         return pd.DataFrame() # Or handle by trying to add it based on keys
         
    return full_stock_history.sort_values(by='ScrapeDate')


def get_closest_available_df(target_date_dt, available_dates_sorted, max_lookback_days=7):
    """
    Finds the DataFrame for target_date_dt or the closest previous available date within max_lookback_days.
    Returns the DataFrame and the actual date string for which data was found.
    """
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return pd.DataFrame(), None # Return empty DF and None for date string

    for i in range(max_lookback_days + 1): # Check target_date_dt first, then go back
        current_check_date_dt = target_date_dt - timedelta(days=i)
        current_check_date_str = current_check_date_dt.strftime("%Y-%m-%d")
        if current_check_date_str in available_dates_sorted:
            df = fetch_df_from_redis_cached(current_check_date_str, st.session_state.get('redis_connected'))
            if df is not None and not df.empty:
                return df, current_check_date_str
    return pd.DataFrame(), None # Return empty DF and None if no suitable data found

# --- Pub/Sub Listener ---
def redis_pubsub_listener():
    """Listens to Redis Pub/Sub and sets a flag in session_state on new messages."""
    if not st.session_state.get('redis_connected', False):
        print("Pub/Sub: Redis not connected, listener not starting.")
        return

    r_pubsub = None
    pubsub = None
    try:
        r_pubsub = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        pubsub = r_pubsub.pubsub()
        pubsub.subscribe(REDIS_UPDATE_CHANNEL)
        st.session_state.pubsub_thread_running = True
        print(f"Pub/Sub: Subscribed to Redis channel: {REDIS_UPDATE_CHANNEL}")
        for message in pubsub.listen():
            if not st.session_state.get('pubsub_thread_should_run', True): # Check flag to stop thread
                print("Pub/Sub: Thread stop signal received.")
                break
            if message and message['type'] == 'message':
                print(f"Pub/Sub: Message received: {message['data']}")
                st.session_state.new_data_arrived_time = datetime.now()
    except redis.exceptions.ConnectionError:
        st.session_state.pubsub_thread_running = False
        print("Pub/Sub: Listener disconnected from Redis.")
    except Exception as e:
        st.session_state.pubsub_thread_running = False
        print(f"Pub/Sub: Error in listener: {e}")
    finally:
        if pubsub:
            try:
                pubsub.unsubscribe(REDIS_UPDATE_CHANNEL)
                pubsub.close()
            except Exception as e:
                print(f"Pub/Sub: Error during unsubscribe/close: {e}")
        if r_pubsub:
             r_pubsub.close()
        st.session_state.pubsub_thread_running = False
        print("Pub/Sub: Listener stopped.")


# --- Initialize Session State ---
if 'redis_connected' not in st.session_state:
    st.session_state.redis_connected = False
if 'new_data_arrived_time' not in st.session_state:
    st.session_state.new_data_arrived_time = None
if 'last_app_run_time_for_pubsub_check' not in st.session_state:
    st.session_state.last_app_run_time_for_pubsub_check = datetime.min
if 'pubsub_thread_started' not in st.session_state:
    st.session_state.pubsub_thread_started = False
if 'pubsub_thread_running' not in st.session_state:
    st.session_state.pubsub_thread_running = False
if 'pubsub_thread_should_run' not in st.session_state: # Flag to gracefully stop thread
    st.session_state.pubsub_thread_should_run = True
if 'latest_df_cache' not in st.session_state:
    st.session_state.latest_df_cache = None
    st.session_state.latest_df_date_str_cache = None


# --- New Function for Identifying Newly Added Stocks ---
def get_newly_added_stocks_df(current_df, historical_df, current_date_str, historical_date_str_actual):
    """
    Identifies stocks in current_df that are not in historical_df.
    
    Args:
        current_df (pd.DataFrame): DataFrame for the most recent date.
        historical_df (pd.DataFrame): DataFrame for the historical comparison date.
                                      Can be None or empty if no data for that date.
        current_date_str (str): The date string for current_df (e.g., "YYYY-MM-DD").
        historical_date_str_actual (str): The actual date string for historical_df.
                                          If historical_df is None/empty, this might indicate
                                          the target date for which data was not found.

    Returns:
        pd.DataFrame: DataFrame with 'Stock Code', 'Name', 'Appeared on/after', 'Compared against date'.
                      Returns an empty DataFrame if no new stocks or if current_df is empty.
    """
    if current_df.empty or 'Stock Code' not in current_df.columns:
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])

    current_codes = set(current_df['Stock Code'].dropna().unique())

    if historical_df is None or historical_df.empty or 'Stock Code' not in historical_df.columns:
        # If no valid historical data, all current stocks are "new" relative to that missing point
        # However, this might not be the desired behavior if we only want to show *truly* new stocks.
        # For now, let's assume if historical_df is missing, we can't determine "newness" reliably for this period.
        # Or, we could list all current_df stocks, but that might be misleading.
        # Let's return an empty DF in this case, indicating we can't make the comparison.
        # A message in the UI will explain this.
        # st.caption(f"Cannot determine new additions compared to {historical_date_str_actual or 'target historical date'} as historical data is missing.")
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])


    historical_codes = set(historical_df['Stock Code'].dropna().unique())
    new_stock_codes = current_codes - historical_codes

    if not new_stock_codes:
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])

    # Get the details of these new codes from the current_df
    # Ensure 'Name' column exists
    if 'Name' not in current_df.columns:
        newly_added_info = pd.DataFrame({'Stock Code': list(new_stock_codes)})
        newly_added_info['Name'] = "N/A" # Placeholder if Name is missing
    else:
        newly_added_info = current_df[current_df['Stock Code'].isin(new_stock_codes)][['Stock Code', 'Name']].drop_duplicates().copy()
    
    newly_added_info['Appeared on/after'] = pd.to_datetime(current_date_str).strftime('%Y-%m-%d')
    newly_added_info['Compared against date'] = pd.to_datetime(historical_date_str_actual).strftime('%Y-%m-%d') if historical_date_str_actual else "N/A"
    
    # Ensure Stock Code is int for display
    newly_added_info['Stock Code'] = newly_added_info['Stock Code'].astype(int)
    return newly_added_info.sort_values(by=['Name', 'Stock Code']).reset_index(drop=True)


# --- Main Application ---
def run_app():
    r = get_redis_connection()

    if r and st.session_state.redis_connected and \
       not st.session_state.get('pubsub_thread_started', False) and \
       st.session_state.get('pubsub_thread_should_run', True):
        listener_thread = threading.Thread(target=redis_pubsub_listener, daemon=True)
        listener_thread.start()
        st.session_state.pubsub_thread_started = True
        print("Main App: Pub/Sub listener thread started.")

    if st.session_state.new_data_arrived_time and \
       st.session_state.new_data_arrived_time > st.session_state.last_app_run_time_for_pubsub_check:
        st.toast(f"New data detected at {st.session_state.new_data_arrived_time.strftime('%H:%M:%S')}. Refreshing...", icon="🔄")
        get_available_dates_sorted.clear()
        fetch_df_from_redis_cached.clear()
        st.session_state.latest_df_cache = None
        st.session_state.last_app_run_time_for_pubsub_check = datetime.now()
        time.sleep(0.1) # Brief pause for toast
        st.rerun()

    st.session_state.last_app_run_time_for_pubsub_check = datetime.now()

    st.title("HKEX Shareholding Dashboard")
    st.markdown("Visualizing shareholding data scraped from HKEXnews.")

    if not st.session_state.get('redis_connected', False) or not r:
        st.warning("Dashboard is currently offline. Waiting for Redis connection...")
        if st.button("Retry Connection"):
            get_redis_connection.clear()
            st.rerun()
        return

    available_dates = get_available_dates_sorted(st.session_state.get('redis_connected'))

    if not available_dates:
        st.warning("No data found in Redis. The backend scraper might not have run yet or Redis is empty.")
        return

    latest_date_str = available_dates[0]
    
    if st.session_state.latest_df_cache is not None and st.session_state.latest_df_date_str_cache == latest_date_str:
        latest_df = st.session_state.latest_df_cache
    else:
        latest_df = fetch_df_from_redis_cached(latest_date_str, st.session_state.get('redis_connected'))
        if latest_df.empty:
             st.warning(f"No data loaded for the latest available date: {latest_date_str}. This might be a non-trading day or data is not yet available.")
        st.session_state.latest_df_cache = latest_df
        st.session_state.latest_df_date_str_cache = latest_date_str

    if latest_df.empty:
        # This check is important. If latest_df is empty, subsequent operations will fail or be meaningless.
        st.error(f"Critical: Could not load data for the latest date ({latest_date_str}). Dashboard cannot proceed with analysis for this date.")
        stock_options_list = ["No stocks available"]
    else:
        st.info(f"Displaying data as of: **{pd.to_datetime(latest_date_str).strftime('%A, %B %d, %Y')}**")
        stock_options_list = get_stock_options(latest_df)


    # --- Sidebar Controls ---
    st.sidebar.header("Chart Controls")
    st.sidebar.subheader("Individual Stock Trend")
    
    default_stock_index_line = 0
    if stock_options_list[0] != "No stocks available": # Check if list is not just the placeholder
        if "TENCENT (700)" in stock_options_list:
            default_stock_index_line = stock_options_list.index("TENCENT (700)")
        elif not latest_df.empty: # Ensure latest_df is not empty before trying to use it
            try:
                top_stock = latest_df.nlargest(1, 'Shareholding in CCASS')
                if not top_stock.empty:
                    top_stock_name = top_stock['Name'].iloc[0]
                    top_stock_code = int(top_stock['Stock Code'].iloc[0])
                    top_stock_option = f"{top_stock_name} ({top_stock_code})"
                    if top_stock_option in stock_options_list:
                        default_stock_index_line = stock_options_list.index(top_stock_option)
            except Exception as e:
                st.sidebar.warning(f"Could not determine default stock: {e}")


    selected_stock_option_line = st.sidebar.selectbox(
        "Select Stock:",
        stock_options_list,
        index=default_stock_index_line,
        key='selected_stock_line'
    )

    st.sidebar.subheader("Market Movers")
    bar_change_period_map = {"1-Day": 1, "5-Day": 5, "20-Day": 20}
    bar_change_period_label = st.sidebar.radio(
        "Change Period:",
        list(bar_change_period_map.keys()),
        key='bar_change_period_label',
        horizontal=True
    )
    bar_change_period_days = bar_change_period_map[bar_change_period_label]

    bar_change_metric = st.sidebar.radio(
        "Change Metric:",
        ["Percentage (%)", "Absolute (Shares)"],
        key='bar_change_metric',
        horizontal=True
    )

    bar_display_scope_map = {"Top 5": (5, True), "Top 10": (10, True), "Bottom 5": (5, False), "Bottom 10": (10, False)}
    bar_display_scope_label = st.sidebar.radio(
        "Display Scope:",
        list(bar_display_scope_map.keys()),
        key='bar_display_scope_label',
        horizontal=True
    )
    bar_n_movers, bar_top_movers = bar_display_scope_map[bar_display_scope_label]

    # --- Main Area: Charts ---
    st.subheader(f"Shareholding Trend")
    if selected_stock_option_line != "No stocks available" and selected_stock_option_line is not None:
        try:
            selected_stock_code_str = selected_stock_option_line.split('(')[-1][:-1]
            selected_stock_code = int(selected_stock_code_str)
            
            line_chart_data = get_historical_data_for_stock(selected_stock_code, available_dates, DEFAULT_DAYS_TO_LOAD_FOR_LINE_CHART)

            if not line_chart_data.empty and 'ScrapeDate' in line_chart_data.columns and 'Shareholding in CCASS' in line_chart_data.columns:
                fig_line = px.line(
                    line_chart_data,
                    x='ScrapeDate',
                    y='Shareholding in CCASS',
                    title=f"Shareholding for {selected_stock_option_line}",
                    labels={'ScrapeDate': 'Date', 'Shareholding in CCASS': 'Shares in CCASS'},
                    markers=True
                )
                fig_line.update_layout(hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info(f"No historical data found for {selected_stock_option_line} within the last {DEFAULT_DAYS_TO_LOAD_FOR_LINE_CHART} available data points, or data is incomplete.")
        except ValueError:
                st.error(f"Could not parse stock code from '{selected_stock_option_line}'. Please check data integrity.")
        except Exception as e:
            st.error(f"Error generating line chart: {e}")
    else:
        st.info("Select a stock from the sidebar to view its shareholding trend.")

    st.subheader(f"Shareholding Movers ({bar_change_period_label})")
    
    # current_day_df is latest_df
    if latest_df.empty:
        st.warning("Cannot calculate movers: Latest daily data is unavailable.")
    else:
        target_historical_date_dt = pd.to_datetime(latest_date_str) - timedelta(days=bar_change_period_days)
        # For movers, we need to look back further if the exact T-N day is not available.
        # Max lookback for movers can be larger, e.g., period_days + a buffer
        historical_df_movers, actual_historical_date_str_movers = get_closest_available_df(
            target_historical_date_dt, available_dates, max_lookback_days=bar_change_period_days + 10 
        )

        if historical_df_movers.empty:
            st.warning(f"Not enough historical data to calculate {bar_change_period_label} change. "
                        f"Required historical data around {target_historical_date_dt.strftime('%Y-%m-%d')} not found within reasonable lookback.")
        else:
            st.caption(f"Comparing data from {latest_date_str} with {actual_historical_date_str_movers} (closest to T-{bar_change_period_days} for movers calculation).")
            
            merged_df = pd.merge(
                latest_df[['Stock Code', 'Name', 'Shareholding in CCASS']],
                historical_df_movers[['Stock Code', 'Shareholding in CCASS']],
                on='Stock Code',
                suffixes=('_current', '_historical')
            )

            if merged_df.empty:
                st.info("No common stocks found between the current and historical periods for mover comparison.")
            else:
                if bar_change_metric == "Absolute (Shares)":
                    merged_df['Change'] = merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']
                    change_label = "Change in Shares"
                else: # Percentage (%)
                    merged_df['Change'] = ((merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']) / merged_df['Shareholding in CCASS_historical']) * 100
                    merged_df['Change'] = merged_df['Change'].replace([np.inf, -np.inf], np.nan)
                    change_label = "Change (%)"
                
                merged_df = merged_df.dropna(subset=['Change'])

                if not merged_df.empty:
                    merged_df = merged_df.sort_values(by='Change', ascending=(not bar_top_movers))
                    movers_df_display = merged_df.head(bar_n_movers)
                    
                    if not movers_df_display.empty:
                        movers_df_display['StockLabel'] = movers_df_display['Name'] + " (" + movers_df_display['Stock Code'].astype(int).astype(str) + ")"
                        fig_bar = px.bar(
                            movers_df_display,
                            x='StockLabel',
                            y='Change',
                            title=f"{bar_display_scope_label} - {bar_change_metric}",
                            labels={'StockLabel': 'Stock', 'Change': change_label},
                            color='Change',
                            color_continuous_scale=px.colors.diverging.RdYlGn if not bar_top_movers else px.colors.diverging.RdYlGn_r
                        )
                        fig_bar.update_layout(xaxis_title="Stock", yaxis_title=change_label)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info(f"No stocks found for {bar_display_scope_label} criteria after filtering.")
                else:
                    st.info("No stocks with calculable changes found for movers.")

    # --- New Component: Newly Added Stocks ---
    st.markdown("---") # Visual separator
    st.subheader("Newly Added Stocks Monitoring")

    if latest_df.empty:
        st.warning("Cannot determine newly added stocks: Latest daily data is unavailable.")
    else:
        periods_days = [1, 5, 20]
        column_names_display = {'Stock Code': 'Code', 'Name': 'Stock Name', 'Appeared on/after': 'Latest Data Date', 'Compared against date': 'Previous Data Date'}


        for days_back in periods_days:
            st.markdown(f"#### New Additions in the Past ~{days_back} Day(s)")
            
            target_historical_date_new_additions = pd.to_datetime(latest_date_str) - timedelta(days=days_back)
            
            # For new additions, we typically want to compare against a single point in the past (T-days_back).
            # max_lookback_days can be small, e.g., days_back + a small buffer, or even just 0 if we want strict T-days_back.
            # Using a small lookback (e.g., 7 days around the target) to find *some* data if exact T-days_back is missing.
            historical_df_new_additions, actual_historical_date_str_new = get_closest_available_df(
                target_historical_date_new_additions, 
                available_dates, 
                max_lookback_days=days_back + 7 # Look around the target date
            )

            if historical_df_new_additions.empty:
                st.caption(f"Could not find historical data around T-{days_back} ({target_historical_date_new_additions.strftime('%Y-%m-%d')}) to compare for new additions. "
                           f"The scraper might need to populate more historical data, or this period might predate available data.")
            else:
                newly_added_stocks = get_newly_added_stocks_df(
                    latest_df, 
                    historical_df_new_additions, 
                    latest_date_str, 
                    actual_historical_date_str_new
                )

                if not newly_added_stocks.empty:
                    st.caption(f"Showing stocks present in data for {pd.to_datetime(latest_date_str).strftime('%Y-%m-%d')} but not found in data for {pd.to_datetime(actual_historical_date_str_new).strftime('%Y-%m-%d')}.")
                    st.dataframe(newly_added_stocks.rename(columns=column_names_display), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No new stock code additions found when comparing {latest_date_str} with data from around T-{days_back} (specifically {actual_historical_date_str_new if actual_historical_date_str_new else 'target date'}).")
    
    # Add a small footer or status information
    st.sidebar.markdown("---")
    if st.session_state.get('redis_connected', False):
        status_color = "green"
        status_text = "Connected"
    else:
        status_color = "red"
        status_text = "Disconnected"
    st.sidebar.markdown(f"**Redis Status:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
    
    if st.session_state.get('pubsub_thread_running', False) :
         st.sidebar.markdown(f"**Live Updates:** <span style='color:green;'>Active</span>", unsafe_allow_html=True)
    elif st.session_state.get('pubsub_thread_started', False): # Started but not running (e.g. error)
         st.sidebar.markdown(f"**Live Updates:** <span style='color:orange;'>Attempted (check logs)</span>", unsafe_allow_html=True)
    else:
         st.sidebar.markdown(f"**Live Updates:** <span style='color:red;'>Inactive</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    # Ensure graceful shutdown of the pubsub listener thread if app is stopped
    try:
        run_app()
    except KeyboardInterrupt:
        st.session_state.pubsub_thread_should_run = False
        print("Application shutting down by KeyboardInterrupt...")
        time.sleep(0.5) # Give thread a moment to see the flag
    finally:
        st.session_state.pubsub_thread_should_run = False
        print("Application exited.")

