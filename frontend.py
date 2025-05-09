import streamlit as st
import pandas as pd
import redis
import plotly.express as px
from datetime import datetime, timedelta
import json
import threading
import time
import numpy as np # For handling inf/nan in calculations
import warnings

warnings.filterwarnings("ignore")
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
    """Establishes and returns a Redis connection for the main app."""
    try:
        r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        r.ping() # Test connection
        st.session_state.redis_connected = True
        # print("Main App: Redis connection successful.")
        return r
    except redis.exceptions.ConnectionError as e:
        st.session_state.redis_connected = False
        st.error(f"Main App: Failed to connect to Redis: {e}. Please ensure Redis is running and accessible.")
        print(f"Main App: Redis connection failed: {e}")
        return None

# --- Data Loading and Processing Functions ---
def deserialize_df_from_redis(json_bytes):
    """Deserializes a JSON string (from Redis bytes) into a Pandas DataFrame."""
    if json_bytes:
        try:
            json_string = json_bytes.decode('utf-8')
            df = pd.read_json(json_string, orient='split')
            if 'ScrapeDate' in df.columns:
                df['ScrapeDate'] = pd.to_datetime(df['ScrapeDate'])
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

@st.cache_data(ttl=300, show_spinner="Fetching available dates from Redis...")
def get_available_dates_sorted(_r_conn_placeholder):
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return []
    try:
        date_keys_bytes = r.keys("????-??-??")
        date_keys_str = sorted([key.decode('utf-8') for key in date_keys_bytes], reverse=True)
        valid_date_keys = []
        for k in date_keys_str:
            try:
                datetime.strptime(k, "%Y-%m-%d")
                valid_date_keys.append(k)
            except ValueError:
                pass
        return valid_date_keys
    except Exception as e:
        st.warning(f"Error fetching date keys from Redis: {e}")
        return []

@st.cache_data(ttl=60, show_spinner="Fetching data for date: {date_key_str}...")
def fetch_df_from_redis_cached(date_key_str, _r_conn_placeholder):
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False) or not date_key_str:
        return pd.DataFrame()
    try:
        print(f"CACHE MISS/EXPIRED for fetch_df_from_redis_cached: Fetching '{date_key_str}' from Redis.")
        json_data = r.get(date_key_str)
        if json_data:
            return deserialize_df_from_redis(json_data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data for {date_key_str} from Redis: {e}")
        return pd.DataFrame()

def get_stock_options(df):
    if df is not None and not df.empty and 'Name' in df.columns and 'Stock Code' in df.columns:
        df_cleaned = df.dropna(subset=['Name', 'Stock Code'])
        df_cleaned['Stock Code'] = df_cleaned['Stock Code'].astype(int)
        options = sorted(list(set(df_cleaned['Name'] + " (" + df_cleaned['Stock Code'].astype(str) + ")")))
        return options
    return ["No stocks available"]

def get_historical_data_for_stock(selected_stock_code, available_dates, num_days):
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return pd.DataFrame()
    stock_data_list = []
    dates_to_fetch = available_dates[:min(num_days, len(available_dates))]
    for date_str in dates_to_fetch:
        daily_df = fetch_df_from_redis_cached(date_str, st.session_state.get('redis_connected'))
        if not daily_df.empty and 'Stock Code' in daily_df.columns:
            stock_specific_data = daily_df[daily_df['Stock Code'] == int(selected_stock_code)]
            if not stock_specific_data.empty:
                stock_data_list.append(stock_specific_data)
    if not stock_data_list:
        return pd.DataFrame()
    full_stock_history = pd.concat(stock_data_list, ignore_index=True)
    if 'ScrapeDate' not in full_stock_history.columns:
         st.warning("ScrapeDate column missing in concatenated historical data.")
         return pd.DataFrame()
    return full_stock_history.sort_values(by='ScrapeDate')

def get_closest_available_df(target_date_dt, available_dates_sorted, max_lookback_days=7):
    r = get_redis_connection()
    if not r or not st.session_state.get('redis_connected', False):
        return pd.DataFrame(), None
    for i in range(max_lookback_days + 1):
        current_check_date_dt = target_date_dt - timedelta(days=i)
        current_check_date_str = current_check_date_dt.strftime("%Y-%m-%d")
        if current_check_date_str in available_dates_sorted:
            df = fetch_df_from_redis_cached(current_check_date_str, st.session_state.get('redis_connected'))
            if df is not None and not df.empty:
                return df, current_check_date_str
    return pd.DataFrame(), None

# --- Pub/Sub Listener ---
def redis_pubsub_listener():
    r_pubsub = None
    pubsub = None
    while st.session_state.get('pubsub_thread_should_run', True):
        try:
            # print("Pub/Sub Listener: Attempting to connect to Redis...")
            r_pubsub = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            r_pubsub.ping()
            # print("Pub/Sub Listener: Connected to Redis successfully.")
            pubsub = r_pubsub.pubsub()
            pubsub.subscribe(REDIS_UPDATE_CHANNEL)
            if not st.session_state.get('pubsub_thread_running', False):
                 st.session_state.pubsub_thread_running = True
            # print(f"Pub/Sub Listener: Subscribed to Redis channel: {REDIS_UPDATE_CHANNEL}")
            for message in pubsub.listen():
                if not st.session_state.get('pubsub_thread_should_run', True):
                    print("Pub/Sub Listener: Thread stop signal received during listen.")
                    break
                if message and message['type'] == 'message':
                    msg_data = message['data'].decode('utf-8') if isinstance(message['data'], bytes) else message['data']
                    print(f"Pub/Sub Listener: Message received for date key: {msg_data}. Triggering app refresh.")
                    
                    # Clear caches to ensure fresh data and connection
                    get_redis_connection.clear()
                    get_available_dates_sorted.clear()
                    fetch_df_from_redis_cached.clear()

                    print("Attempt Rerun")
                    st.rerun() # This will stop the current script run and restart it from the top.
                    # Clear any manual session state caches if they exist (optional, depends on specific needs)
                    # For this simplification, we rely on clearing the @st.cache_data functions primarily.
                    # st.session_state.latest_df_cache = None # Example if still used
                    # st.session_state.latest_df_date_str_cache = None # Example if still used

                    
            
            if not st.session_state.get('pubsub_thread_should_run', True): # Check again after loop
                break
        except redis.exceptions.ConnectionError as conn_err:
            print(f"Pub/Sub Listener: Redis connection failed: {conn_err}. Retrying in 10s...")
            if st.session_state.get('pubsub_thread_running', False):
                 st.session_state.pubsub_thread_running = False
            for _ in range(10):
                if not st.session_state.get('pubsub_thread_should_run', True): break
                time.sleep(1)
            if not st.session_state.get('pubsub_thread_should_run', True): break
            return
        except Exception as e:
            print(f"Pub/Sub Listener: Error in listener: {e}. Retrying in 10s...")
            if st.session_state.get('pubsub_thread_running', False):
                 st.session_state.pubsub_thread_running = False
            for _ in range(10):
                if not st.session_state.get('pubsub_thread_should_run', True): break
                time.sleep(1)
            if not st.session_state.get('pubsub_thread_should_run', True): break
            return
        
    st.session_state.pubsub_thread_running = False
    print("Pub/Sub Listener: Thread fully stopped.")

# --- Initialize Session State ---
if 'redis_connected' not in st.session_state:
    st.session_state.redis_connected = False
# REMOVED: new_data_arrived_time
# REMOVED: last_app_run_time_for_pubsub_check
if 'pubsub_thread_started' not in st.session_state:
    st.session_state.pubsub_thread_started = False
if 'pubsub_thread_running' not in st.session_state:
    st.session_state.pubsub_thread_running = False
if 'pubsub_thread_should_run' not in st.session_state:
    st.session_state.pubsub_thread_should_run = True
# Keep these manual caches if they provide intra-run optimization,
# but they will be repopulated on each full run triggered by pub/sub.
if 'latest_df_cache' not in st.session_state:
    st.session_state.latest_df_cache = None
    st.session_state.latest_df_date_str_cache = None
# REMOVED: updated_date_key_from_pubsub


def get_newly_added_stocks_df(current_df, historical_df, current_date_str, historical_date_str_actual):
    if current_df.empty or 'Stock Code' not in current_df.columns:
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])
    current_codes = set(current_df['Stock Code'].dropna().unique())
    if historical_df is None or historical_df.empty or 'Stock Code' not in historical_df.columns:
        # If no historical data, all current stocks could be considered "new" against a void.
        # For this function's purpose (showing *changes*), an empty DataFrame is appropriate.
        # Or, one might return all current_df stocks with a note "no historical comparison point".
        # Current implementation returns empty, implying new *relative to available history*.
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])

    historical_codes = set(historical_df['Stock Code'].dropna().unique())
    new_stock_codes = current_codes - historical_codes

    if not new_stock_codes:
        return pd.DataFrame(columns=['Stock Code', 'Name', 'Appeared on/after', 'Compared against date'])

    if 'Name' not in current_df.columns:
        newly_added_info = pd.DataFrame({'Stock Code': list(new_stock_codes)})
        newly_added_info['Name'] = "N/A"
    else:
        newly_added_info = current_df[current_df['Stock Code'].isin(new_stock_codes)][['Stock Code', 'Name']].drop_duplicates().copy()

    newly_added_info['Appeared on/after'] = pd.to_datetime(current_date_str).strftime('%Y-%m-%d')
    newly_added_info['Compared against date'] = pd.to_datetime(historical_date_str_actual).strftime('%Y-%m-%d') if historical_date_str_actual else "N/A"
    newly_added_info['Stock Code'] = newly_added_info['Stock Code'].astype(int)
    return newly_added_info.sort_values(by=['Name', 'Stock Code']).reset_index(drop=True)

# --- Main Application ---
def run_app():
    r = get_redis_connection()
    if r and st.session_state.redis_connected and \
       not st.session_state.get('pubsub_thread_started', False):
        st.session_state.pubsub_thread_should_run = True
        listener_thread = threading.Thread(target=redis_pubsub_listener, daemon=True)
        listener_thread.start()
        st.session_state.pubsub_thread_started = True
        print("Main App: Pub/Sub listener thread start initiated.")
    elif not r and not st.session_state.get('pubsub_thread_started', False):
        print("Main App: Main Redis connection failed. Pub/Sub listener thread will not be started yet.")

    # REMOVED: The logic block that checked st.session_state.new_data_arrived_time
    # st.rerun() is now called directly from the pubsub listener.

    st.title("HKEX Shareholding Dashboard")
    st.markdown("Visualizing shareholding data scraped from HKEXnews.")

    if not st.session_state.get('redis_connected', False):
        st.warning("Dashboard is currently attempting to connect to Redis...")
        if st.button("Retry Main Redis Connection"):
            get_redis_connection.clear() # Clear resource cache for connection
            st.rerun()
    
    # Data loading will happen here on every run/rerun
    available_dates = get_available_dates_sorted(st.session_state.get('redis_connected'))
    print(f"DATA LOAD: Available dates (up to 5): {available_dates[:5] if available_dates else 'None'}")

    latest_df = pd.DataFrame()
    latest_date_str = None
    if available_dates:
        latest_date_str = available_dates[0]
        # The manual st.session_state.latest_df_cache is an intra-run optimization.
        # It's useful if latest_df is used multiple times *within the same script execution*.
        # Since fetch_df_from_redis_cached is cleared by pub/sub, this will fetch fresh data from Redis
        # at the start of a new run.
        if st.session_state.latest_df_cache is not None and \
           st.session_state.latest_df_date_str_cache == latest_date_str:
            print(f"DATA LOAD: Using st.session_state CACHED latest_df for {latest_date_str}.")
            latest_df = st.session_state.latest_df_cache
        else:
            print(f"DATA LOAD: FETCHING latest_df for {latest_date_str} via fetch_df_from_redis_cached.")
            latest_df = fetch_df_from_redis_cached(latest_date_str, st.session_state.get('redis_connected'))
            if not latest_df.empty:
                 print(f"DATA LOAD: FETCHED new latest_df for {latest_date_str}. Shape: {latest_df.shape}.")
            elif latest_date_str:
                 st.warning(f"No data loaded for the latest available date: {latest_date_str}.")
                 print(f"DATA LOAD: Fetched EMPTY latest_df for {latest_date_str}.")
            
            st.session_state.latest_df_cache = latest_df # Store for current run
            st.session_state.latest_df_date_str_cache = latest_date_str # Store for current run
    
    if not latest_df.empty:
        st.info(f"Displaying data as of: **{pd.to_datetime(latest_date_str).strftime('%A, %B %d, %Y')}** (Total Records: {latest_df.shape[0]})")
        stock_options_list = get_stock_options(latest_df)
    elif latest_date_str:
        st.error(f"Critical: Could not load data for the latest date ({latest_date_str}).")
        stock_options_list = ["No stocks available"]
    else:
        if st.session_state.get('redis_connected', False):
             st.warning("No data found in Redis. Backend scraper might not have run or Redis is empty.")
        stock_options_list = ["No stocks available"]

    # --- Sidebar Controls ---
    st.sidebar.header("Chart Controls")
    st.sidebar.subheader("Individual Stock Trend")
    default_stock_index_line = 0
    if stock_options_list[0] != "No stocks available":
        if "TENCENT (700)" in stock_options_list:
            default_stock_index_line = stock_options_list.index("TENCENT (700)")
        elif not latest_df.empty:
            try:
                top_stock = latest_df.nlargest(1, 'Shareholding in CCASS')
                if not top_stock.empty:
                    top_stock_name = top_stock['Name'].iloc[0]
                    top_stock_code = int(top_stock['Stock Code'].iloc[0])
                    top_stock_option = f"{top_stock_name} ({top_stock_code})"
                    if top_stock_option in stock_options_list:
                        default_stock_index_line = stock_options_list.index(top_stock_option)
            except Exception: pass
    selected_stock_option_line = st.sidebar.selectbox("Select Stock:", stock_options_list, index=default_stock_index_line, key='selected_stock_line')

    st.sidebar.subheader("Market Movers")
    bar_change_period_map = {"1-Day": 1, "5-Day": 5, "20-Day": 20}
    bar_change_period_label = st.sidebar.radio("Change Period:", list(bar_change_period_map.keys()), key='bar_change_period_label', horizontal=True)
    bar_change_period_days = bar_change_period_map[bar_change_period_label]
    bar_change_metric = st.sidebar.radio("Change Metric:", ["Percentage (%)", "Absolute (Shares)"], key='bar_change_metric', horizontal=True)
    bar_display_scope_map = {"Top 5": (5, True), "Top 10": (10, True), "Bottom 5": (5, False), "Bottom 10": (10, False)}
    bar_display_scope_label = st.sidebar.radio("Display Scope:", list(bar_display_scope_map.keys()), key='bar_display_scope_label', horizontal=True)
    bar_n_movers, bar_top_movers = bar_display_scope_map[bar_display_scope_label]

    # --- Main Area: Charts ---
    st.subheader(f"Shareholding Trend")
    if selected_stock_option_line != "No stocks available" and selected_stock_option_line is not None:
        try:
            selected_stock_code_str = selected_stock_option_line.split('(')[-1][:-1]
            selected_stock_code = int(selected_stock_code_str)
            line_chart_data = get_historical_data_for_stock(selected_stock_code, available_dates, DEFAULT_DAYS_TO_LOAD_FOR_LINE_CHART)
            if not line_chart_data.empty and 'ScrapeDate' in line_chart_data.columns and 'Shareholding in CCASS' in line_chart_data.columns:
                fig_line = px.line(line_chart_data, x='ScrapeDate', y='Shareholding in CCASS', title=f"Shareholding for {selected_stock_option_line}", labels={'ScrapeDate': 'Date', 'Shareholding in CCASS': 'Shares in CCASS'}, markers=True)
                fig_line.update_layout(hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
            else: st.info(f"No historical data for {selected_stock_option_line} (last {DEFAULT_DAYS_TO_LOAD_FOR_LINE_CHART} data points).")
        except ValueError: st.error(f"Could not parse stock code from '{selected_stock_option_line}'.")
        except Exception as e: st.error(f"Error generating line chart: {e}")
    else: st.info("Select a stock to view its trend.")

    st.subheader(f"Shareholding Movers ({bar_change_period_label})")
    if latest_df.empty: st.warning("Cannot calculate movers: Latest daily data unavailable.")
    else:
        target_historical_date_dt = pd.to_datetime(latest_date_str) - timedelta(days=bar_change_period_days)
        historical_df_movers, actual_historical_date_str_movers = get_closest_available_df(target_historical_date_dt, available_dates, max_lookback_days=bar_change_period_days + 10) # Look back a bit more to find data
        if historical_df_movers.empty: st.warning(f"Not enough historical data for {bar_change_period_label} change (around {target_historical_date_dt.strftime('%Y-%m-%d')}).")
        else:
            st.caption(f"Comparing {latest_date_str} with {actual_historical_date_str_movers} for movers.")
            merged_df = pd.merge(latest_df[['Stock Code', 'Name', 'Shareholding in CCASS']], historical_df_movers[['Stock Code', 'Shareholding in CCASS']], on='Stock Code', suffixes=('_current', '_historical'))
            if merged_df.empty: st.info("No common stocks for mover comparison.")
            else:
                if bar_change_metric == "Absolute (Shares)":
                    merged_df['Change'] = merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']
                    change_label = "Change in Shares"
                else: # Percentage
                    merged_df['Change'] = ((merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']) / merged_df['Shareholding in CCASS_historical']) * 100
                    merged_df['Change'] = merged_df['Change'].replace([np.inf, -np.inf], np.nan) # Handle division by zero if historical was 0
                    change_label = "Change (%)"
                merged_df = merged_df.dropna(subset=['Change'])
                if not merged_df.empty:
                    merged_df = merged_df.sort_values(by='Change', ascending=(not bar_top_movers))
                    movers_df_display = merged_df.head(bar_n_movers)
                    if not movers_df_display.empty:
                        movers_df_display['StockLabel'] = movers_df_display['Name'] + " (" + movers_df_display['Stock Code'].astype(int).astype(str) + ")"
                        fig_bar = px.bar(movers_df_display, x='StockLabel', y='Change', title=f"{bar_display_scope_label} - {bar_change_metric}", labels={'StockLabel': 'Stock', 'Change': change_label}, color='Change', color_continuous_scale=px.colors.diverging.RdYlGn if not bar_top_movers else px.colors.diverging.RdYlGn_r)
                        fig_bar.update_layout(xaxis_title="Stock", yaxis_title=change_label)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else: st.info(f"No stocks for {bar_display_scope_label} criteria after filtering.")
                else: st.info("No stocks with calculable changes for movers.")

    st.markdown("---")
    st.subheader("Newly Added Stocks Monitoring")
    if latest_df.empty: st.warning("Cannot determine newly added stocks: Latest daily data unavailable.")
    else:
        periods_days = [1, 5, 20]
        column_names_display = {'Stock Code': 'Code', 'Name': 'Stock Name', 'Appeared on/after': 'Latest Data Date', 'Compared against date': 'Previous Data Date'}
        for days_back in periods_days:
            st.markdown(f"#### New Additions in the Past ~{days_back} Day(s)")
            target_historical_date_new_additions = pd.to_datetime(latest_date_str) - timedelta(days=days_back)
            historical_df_new_additions, actual_historical_date_str_new = get_closest_available_df(target_historical_date_new_additions, available_dates, max_lookback_days=days_back + 7)
            if historical_df_new_additions.empty and actual_historical_date_str_new is None:
                 st.caption(f"Could not find any historical data around T-{days_back} ({target_historical_date_new_additions.strftime('%Y-%m-%d')}) to compare for new additions.")
            # If historical_df_new_additions is empty BUT actual_historical_date_str_new is NOT None, it means get_closest_available_df found a date but the data was empty.
            # The function get_newly_added_stocks_df handles historical_df.empty by returning an empty df or all current stocks.
            # For clarity, we ensure that newly_added_stocks is called regardless if latest_df exists.
            
            newly_added_stocks = get_newly_added_stocks_df(latest_df, historical_df_new_additions, latest_date_str, actual_historical_date_str_new)
            if not newly_added_stocks.empty:
                st.caption(f"Stocks in {pd.to_datetime(latest_date_str).strftime('%Y-%m-%d')} data but not in {pd.to_datetime(actual_historical_date_str_new).strftime('%Y-%m-%d')} data.")
                st.dataframe(newly_added_stocks.rename(columns=column_names_display), use_container_width=True, hide_index=True)
            else:
                comparison_date_info = f"data from ~T-{days_back}"
                if actual_historical_date_str_new:
                    comparison_date_info = f"data from {pd.to_datetime(actual_historical_date_str_new).strftime('%Y-%m-%d')}"
                elif historical_df_new_additions.empty and actual_historical_date_str_new is None: # Truly no historical point found
                    comparison_date_info = f"any comparable historical data around T-{days_back}"

                st.info(f"No new stock additions found comparing {latest_date_str} with {comparison_date_info}.")
    
    st.sidebar.markdown("---")
    status_color, status_text = ("green", "Connected") if st.session_state.get('redis_connected', False) else ("red", "Disconnected")
    st.sidebar.markdown(f"**Main Redis Status:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
    if st.session_state.get('pubsub_thread_running', False): st.sidebar.markdown(f"**Live Updates (Pub/Sub):** <span style='color:green;'>Active</span>", unsafe_allow_html=True)
    elif st.session_state.get('pubsub_thread_started', False): st.sidebar.markdown(f"**Live Updates (Pub/Sub):** <span style='color:orange;'>Attempting/Reconnecting</span>", unsafe_allow_html=True)
    else: st.sidebar.markdown(f"**Live Updates (Pub/Sub):** <span style='color:red;'>Inactive</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        run_app()
    except KeyboardInterrupt:
        print("Main App: KeyboardInterrupt. Shutting down...")
        st.session_state.pubsub_thread_should_run = False
        time.sleep(1) # Give thread a moment to see the flag
    finally:
        # Ensure the flag is set for graceful thread shutdown on any exit
        st.session_state.pubsub_thread_should_run = False
        print("Main App: Application exited.")