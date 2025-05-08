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
        st.warning(f"No data found in Redis for key: {date_key_str}")
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
    dates_to_fetch = available_dates[:num_days] # Get the most recent N dates

    for date_str in dates_to_fetch:
        daily_df = fetch_df_from_redis_cached(date_str, st.session_state.get('redis_connected'))
        if not daily_df.empty and 'Stock Code' in daily_df.columns:
            stock_specific_data = daily_df[daily_df['Stock Code'] == selected_stock_code]
            if not stock_specific_data.empty:
                stock_data_list.append(stock_specific_data)
    
    if not stock_data_list:
        return pd.DataFrame()
    
    # Concatenate all found data, ensuring 'ScrapeDate' is present
    full_stock_history = pd.concat(stock_data_list, ignore_index=True)
    if 'ScrapeDate' not in full_stock_history.columns:
         st.warning("ScrapeDate column missing in concatenated historical data.")
         return pd.DataFrame()
         
    return full_stock_history.sort_values(by='ScrapeDate')


def get_closest_available_df(target_date_dt, available_dates_sorted, max_lookback_days=7):
    """
    Finds the DataFrame for target_date_dt or the closest previous available date within max_lookback_days.
    """
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return None, None

    for i in range(max_lookback_days + 1):
        current_check_date_dt = target_date_dt - timedelta(days=i)
        current_check_date_str = current_check_date_dt.strftime("%Y-%m-%d")
        if current_check_date_str in available_dates_sorted:
            df = fetch_df_from_redis_cached(current_check_date_str, st.session_state.get('redis_connected'))
            if df is not None and not df.empty:
                return df, current_check_date_str
    return None, None


# --- Pub/Sub Listener (Simplified: Triggers rerun on next interaction if new data flag is set) ---
def redis_pubsub_listener():
    """Listens to Redis Pub/Sub and sets a flag in session_state on new messages."""
    if not st.session_state.get('redis_connected', False):
        return

    r_pubsub = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    pubsub = r_pubsub.pubsub()
    try:
        pubsub.subscribe(REDIS_UPDATE_CHANNEL)
        st.session_state.pubsub_thread_running = True
        print(f"Subscribed to Redis channel: {REDIS_UPDATE_CHANNEL}") # For console debugging
        for message in pubsub.listen():
            if message and message['type'] == 'message':
                print(f"Pub/Sub message received: {message['data']}") # For console debugging
                st.session_state.new_data_arrived_time = datetime.now()
                # Note: Direct st.rerun() from thread is problematic.
                # The main app will check 'new_data_arrived_time'.
    except redis.exceptions.ConnectionError:
        st.session_state.pubsub_thread_running = False
        print("Pub/Sub listener disconnected from Redis.") # For console debugging
    except Exception as e:
        st.session_state.pubsub_thread_running = False
        print(f"Error in Pub/Sub listener: {e}") # For console debugging
    finally:
        if pubsub:
            pubsub.unsubscribe(REDIS_UPDATE_CHANNEL)
            pubsub.close()
        st.session_state.pubsub_thread_running = False
        print("Pub/Sub listener stopped.") # For console debugging


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
if 'latest_df_cache' not in st.session_state: # Cache for the latest_df to avoid re-deriving stock options
    st.session_state.latest_df_cache = None
    st.session_state.latest_df_date_str_cache = None


# --- Main Application ---
def run_app():
    # Establish Redis connection (or get from cache)
    r = get_redis_connection()

    # Start Pub/Sub listener thread if not already started and Redis is connected
    if r and st.session_state.redis_connected and not st.session_state.pubsub_thread_started:
        listener = threading.Thread(target=redis_pubsub_listener, daemon=True)
        listener.start()
        st.session_state.pubsub_thread_started = True
        print("Pub/Sub listener thread started.") # For console debugging

    # Check for Pub/Sub updates
    if st.session_state.new_data_arrived_time and \
       st.session_state.new_data_arrived_time > st.session_state.last_app_run_time_for_pubsub_check:
        st.toast(f"New data detected at {st.session_state.new_data_arrived_time.strftime('%H:%M:%S')}. Refreshing...", icon="🔄")
        # Clear relevant caches
        get_available_dates_sorted.clear()
        fetch_df_from_redis_cached.clear()
        st.session_state.latest_df_cache = None # Clear specific cache
        st.session_state.last_app_run_time_for_pubsub_check = datetime.now()
        # Small delay to allow UI to show toast before rerun
        time.sleep(0.1)
        st.rerun()

    st.session_state.last_app_run_time_for_pubsub_check = datetime.now()


    st.title("HKEX Shareholding Dashboard")
    st.markdown("Visualizing shareholding data scraped from HKEXnews.")

    if not st.session_state.get('redis_connected', False) or not r:
        st.warning("Dashboard is currently offline. Waiting for Redis connection...")
        if st.button("Retry Connection"):
            get_redis_connection.clear() # Clear resource cache to force re-connect
            st.rerun()
        return

    # --- Load initial data for UI ---
    available_dates = get_available_dates_sorted(st.session_state.get('redis_connected'))

    if not available_dates:
        st.warning("No data found in Redis. The backend scraper might not have run yet or Redis is empty.")
        return

    latest_date_str = available_dates[0]
    
    # Use cached latest_df if available and for the same date
    if st.session_state.latest_df_cache is not None and st.session_state.latest_df_date_str_cache == latest_date_str:
        latest_df = st.session_state.latest_df_cache
    else:
        latest_df = fetch_df_from_redis_cached(latest_date_str, st.session_state.get('redis_connected'))
        st.session_state.latest_df_cache = latest_df
        st.session_state.latest_df_date_str_cache = latest_date_str


    if latest_df.empty:
        st.warning(f"Could not load data for the latest available date: {latest_date_str}. The data might be corrupted or missing.")
        stock_options_list = ["No stocks available"]
    else:
        st.info(f"Displaying data as of: **{pd.to_datetime(latest_date_str).strftime('%A, %B %d, %Y')}**")
        stock_options_list = get_stock_options(latest_df)

    # --- Sidebar Controls ---
    st.sidebar.header("Chart Controls")

    # Line Chart Controls
    st.sidebar.subheader("Individual Stock Trend")
    
    default_stock_index_line = 0
    if "TENCENT (700)" in stock_options_list: # Default to Tencent if available
        default_stock_index_line = stock_options_list.index("TENCENT (700)")
    elif stock_options_list[0] != "No stocks available" and not latest_df.empty:
        # Default to stock with highest shareholding if Tencent not found
        try:
            top_stock = latest_df.nlargest(1, 'Shareholding in CCASS')
            if not top_stock.empty:
                top_stock_name = top_stock['Name'].iloc[0]
                top_stock_code = int(top_stock['Stock Code'].iloc[0])
                top_stock_option = f"{top_stock_name} ({top_stock_code})"
                if top_stock_option in stock_options_list:
                    default_stock_index_line = stock_options_list.index(top_stock_option)
        except Exception:
            pass # Keep default index 0

    selected_stock_option_line = st.sidebar.selectbox(
        "Select Stock:",
        stock_options_list,
        index=default_stock_index_line,
        key='selected_stock_line'
    )

    # Bar Chart Controls
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
            # Extract stock code (it's an integer)
            selected_stock_code = int(selected_stock_option_line.split('(')[-1][:-1])
            
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
    
    current_day_df = latest_df # Already fetched
    
    if current_day_df.empty:
        st.warning("Cannot calculate movers: Latest daily data is unavailable.")
    else:
        target_historical_date_dt = pd.to_datetime(latest_date_str) - timedelta(days=bar_change_period_days)
        historical_df, actual_historical_date_str = get_closest_available_df(target_historical_date_dt, available_dates)

        if historical_df is None or historical_df.empty:
            st.warning(f"Not enough historical data to calculate {bar_change_period_label} change. "
                        f"Required historical data around {target_historical_date_dt.strftime('%Y-%m-%d')} not found.")
        else:
            st.caption(f"Comparing data from {latest_date_str} with {actual_historical_date_str} (closest to T-{bar_change_period_days}).")
            
            # Merge current and historical data
            merged_df = pd.merge(
                current_day_df[['Stock Code', 'Name', 'Shareholding in CCASS']],
                historical_df[['Stock Code', 'Shareholding in CCASS']],
                on='Stock Code',
                suffixes=('_current', '_historical')
            )

            if merged_df.empty:
                st.info("No common stocks found between the current and historical periods for comparison.")
            else:
                # Calculate change
                if bar_change_metric == "Absolute (Shares)":
                    merged_df['Change'] = merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']
                    change_label = "Change in Shares"
                else: # Percentage (%)
                    # Handle division by zero or NaN if historical is 0 or missing
                    merged_df['Change'] = ((merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']) / merged_df['Shareholding in CCASS_historical']) * 100
                    merged_df['Change'] = merged_df['Change'].replace([np.inf, -np.inf], np.nan) # Replace inf with NaN
                    change_label = "Change (%)"
                
                merged_df = merged_df.dropna(subset=['Change']) # Remove rows where change couldn't be calculated

                if not merged_df.empty:
                    # Sort and select top/bottom N
                    merged_df = merged_df.sort_values(by='Change', ascending=(not bar_top_movers))
                    movers_df = merged_df.head(bar_n_movers)
                    
                    if not movers_df.empty:
                        movers_df['StockLabel'] = movers_df['Name'] + " (" + movers_df['Stock Code'].astype(int).astype(str) + ")"
                        fig_bar = px.bar(
                            movers_df,
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
                    st.info("No stocks with calculable changes found.")
    st.dataframe(merged_df)

if __name__ == "__main__":
    run_app()
