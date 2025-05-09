import redis
import pandas as pd
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
import time
import logging
import io
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv
# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 3 # Ensure this matches your backend and frontend
REDIS_UPDATE_CHANNEL = 'data_updated' # Ensure this matches your backend and frontend

load_dotenv()
# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = 587  # Or 465 for SSL, adjust as needed
SMTP_USERNAME = os.getenv("SMTP_USERNAME") 
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") 
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS")

# Analysis Configuration
MOVERS_COUNT = 5
DAYS_FOR_NEW_ADDITIONS = [1, 5] # For 1-day and 5-day new additions
MAX_HISTORICAL_LOOKBACK_DAYS = 15 # Max days to look back for D-1 or D-5 data if exact date is missing

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
        # logging.FileHandler('email_automation.log') # Optional: log to file
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def format_millions(x, pos):
    """Formats a number in millions (e.g., 1000000 -> 1M)."""
    if abs(x) < 1_000_000:
        return f'{x/1_000_000:.1f}M' # Show one decimal for values less than 1M if desired, or adjust
    return f'{int(x/1_000_000)}M'

def deserialize_df_from_redis(json_bytes):
    """Deserializes a JSON string (from Redis bytes) into a Pandas DataFrame."""
    if json_bytes:
        try:
            json_string = json_bytes.decode('utf-8')
            df = pd.read_json(json_string, orient='split')
            if 'ScrapeDate' in df.columns:
                df['ScrapeDate'] = pd.to_datetime(df['ScrapeDate'])
            # Ensure numeric types for relevant columns
            for col in ['Shareholding in CCASS', 'Stock Code', '% of the total number of Issued Shares/Units']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Stock Code']) # Critical for merging
            df['Stock Code'] = df['Stock Code'].astype(int)
            return df
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Redis: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing DataFrame from Redis: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_available_dates_sorted(r_conn):
    """Fetches and sorts all valid YYYY-MM-DD date keys from Redis."""
    if not r_conn:
        return []
    try:
        date_keys_bytes = r_conn.keys("????-??-??") # Pattern for YYYY-MM-DD
        date_keys_str = sorted([key.decode('utf-8') for key in date_keys_bytes], reverse=True)
        valid_date_keys = []
        for k in date_keys_str:
            try:
                datetime.strptime(k, "%Y-%m-%d")
                valid_date_keys.append(k)
            except ValueError:
                logger.warning(f"Invalid date key format found in Redis: {k}")
        return valid_date_keys
    except Exception as e:
        logger.error(f"Error fetching date keys from Redis: {e}")
        return []

def fetch_df_from_redis(r_conn, date_key_str):
    """Fetches and deserializes a DataFrame for a specific date key from Redis."""
    if not r_conn or not date_key_str:
        return pd.DataFrame()
    try:
        json_data = r_conn.get(date_key_str)
        if json_data:
            return deserialize_df_from_redis(json_data)
        logger.warning(f"No data found in Redis for key: {date_key_str}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data for {date_key_str} from Redis: {e}")
        return pd.DataFrame()

def get_closest_available_df(r_conn, target_date_dt, available_dates_sorted, max_lookback_days=MAX_HISTORICAL_LOOKBACK_DAYS):
    """
    Finds the DataFrame for the target_date_dt or the closest prior date within max_lookback_days.
    """
    if not r_conn:
        return pd.DataFrame(), None
    for i in range(max_lookback_days + 1):
        current_check_date_dt = target_date_dt - timedelta(days=i)
        current_check_date_str = current_check_date_dt.strftime("%Y-%m-%d")
        if current_check_date_str in available_dates_sorted:
            df = fetch_df_from_redis(r_conn, current_check_date_str)
            if df is not None and not df.empty:
                logger.info(f"Found data for {current_check_date_str} (target was {target_date_dt.strftime('%Y-%m-%d')}, lookback {i} days)")
                return df, current_check_date_str
    logger.warning(f"Could not find data for target {target_date_dt.strftime('%Y-%m-%d')} within {max_lookback_days} lookback days.")
    return pd.DataFrame(), None


class HKEXEmailAutomationTool:
    def __init__(self):
        self.redis_client = None
        try:
            self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.critical(f"Failed to connect to Redis: {e}. Email automation will not run.")
            raise  # Re-raise to stop initialization if Redis is unavailable

        self.available_dates_cache = [] # Cache for available dates to reduce Redis calls

    def _refresh_available_dates(self):
        """Refreshes the cache of available dates from Redis."""
        self.available_dates_cache = get_available_dates_sorted(self.redis_client)

    def _calculate_movers(self, current_df, historical_df):
        """Calculates top and bottom movers based on absolute share change."""
        if current_df.empty or historical_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Ensure 'Stock Code' and 'Shareholding in CCASS' are present
        required_cols = ['Stock Code', 'Name', 'Shareholding in CCASS']
        if not all(col in current_df.columns for col in required_cols) or \
           not all(col in historical_df.columns for col in ['Stock Code', 'Shareholding in CCASS']):
            logger.error("Required columns missing for mover calculation.")
            return pd.DataFrame(), pd.DataFrame()

        merged_df = pd.merge(
            current_df[['Stock Code', 'Name', 'Shareholding in CCASS']],
            historical_df[['Stock Code', 'Shareholding in CCASS']],
            on='Stock Code',
            suffixes=('_current', '_historical')
        )

        if merged_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        merged_df['ShareChange'] = merged_df['Shareholding in CCASS_current'] - merged_df['Shareholding in CCASS_historical']
        merged_df = merged_df.dropna(subset=['ShareChange'])

        top_movers = merged_df.nlargest(MOVERS_COUNT, 'ShareChange')
        bottom_movers = merged_df.nsmallest(MOVERS_COUNT, 'ShareChange')

        return top_movers, bottom_movers

    def _generate_movers_chart(self, movers_df, title, color='green'):
        """Generates a bar chart for movers and returns it as bytes."""
        if movers_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a 'StockLabel' for better display on chart
        movers_df['StockLabel'] = movers_df['Name'] + " (" + movers_df['Stock Code'].astype(str) + ")"

        bars = ax.barh(movers_df['StockLabel'], movers_df['ShareChange'], color=color)
        ax.set_xlabel('Change in Number of Shares (Millions)') # Update axis label
        ax.set_ylabel('Stock')
        ax.set_title(title)
        ax.invert_yaxis()  # Display top-to-bottom

        # --- MODIFICATION START ---
        # Helper function for formatting labels in millions
        def format_millions_for_chart(num):
            if abs(num) >= 1_000_000:
                return f'{num / 1_000_000:.1f}M'.replace('.0M','M') # e.g. 11.0M becomes 11M, 11.5M stays 11.5M
            elif abs(num) >= 1_000: # For values between 1K and 1M, show in K
                 return f'{num / 1_000:.0f}K'
            else: # For values less than 1K, show as is
                return f'{num:.0f}'

        # Format x-axis to show numbers in millions
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_millions_for_chart(x)))

        # Add labels to bars in millions
        for bar in bars:
            width = bar.get_width()
            formatted_width = format_millions_for_chart(width)
            label_x_pos = width if width > 0 else width - (0.02 * ax.get_xlim()[1]) # Adjust label position for negative bars slightly
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, formatted_width,
                    va='center', ha='left' if width > 0 else 'right', fontsize=9) # Adjusted fontsize
        # --- MODIFICATION END ---

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.getvalue()

    def _calculate_new_additions(self, current_df, historical_df):
        """Identifies new stock additions."""
        if current_df.empty or 'Stock Code' not in current_df.columns:
            return pd.DataFrame()
        
        current_codes = set(current_df['Stock Code'].unique())

        if historical_df is None or historical_df.empty or 'Stock Code' not in historical_df.columns:
            # If no historical data, all current stocks could be considered "new"
            # For this function, we'll assume "new" means not present in the *specific* historical_df provided
            logger.info("No valid historical data for new additions comparison; all current stocks might be listed as new if historical_df is empty.")
            # If historical_df is truly empty (not just lacking 'Stock Code'), then all current are new
            if historical_df is None or historical_df.empty:
                 newly_added_info = current_df[['Stock Code', 'Name']].drop_duplicates().copy()
                 newly_added_info['Stock Code'] = newly_added_info['Stock Code'].astype(int)
                 return newly_added_info.sort_values(by=['Name', 'Stock Code']).reset_index(drop=True)


        historical_codes = set(historical_df['Stock Code'].unique())
        new_stock_codes = current_codes - historical_codes

        if not new_stock_codes:
            return pd.DataFrame()

        newly_added_info = current_df[current_df['Stock Code'].isin(new_stock_codes)][['Stock Code', 'Name']].drop_duplicates().copy()
        newly_added_info['Stock Code'] = newly_added_info['Stock Code'].astype(int)
        return newly_added_info.sort_values(by=['Name', 'Stock Code']).reset_index(drop=True)

    def _format_new_additions_html(self, new_additions_df, period_desc):
        """Formats new additions DataFrame to an HTML table string."""
        if new_additions_df.empty:
            return f"<p>No new stock additions found compared to {period_desc}.</p>"
        
        html = f"<h4>New Stock Additions (Compared to {period_desc})</h4>"
        # Strip HTML tags from stock names if any, just in case
        new_additions_df_display = new_additions_df.copy()
        if 'Name' in new_additions_df_display.columns:
            new_additions_df_display['Name'] = new_additions_df_display['Name'].astype(str).str.replace('<[^<]+?>', '', regex=True)
        
        html += new_additions_df_display[['Stock Code', 'Name']].to_html(index=False, border=0, classes="dataframe")
        return html


    def _generate_email_content(self, current_date_str, current_df, analysis_data):
        """Generates the MIMEMultipart email message."""
        msg = MIMEMultipart('related')
        msg['Subject'] = f"HKEX Southbound Daily Update - {current_date_str}"
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENTS

        # --- HTML Body ---
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
                .container {{ padding: 20px; background-color: #ffffff; margin: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #555; margin-top: 30px; }}
                h4 {{ color: #777; margin-top: 20px; }}
                p {{ line-height: 1.6; color: #444; }}
                table.dataframe {{ border-collapse: collapse; width: auto; margin-bottom: 20px; font-size: 0.9em; }}
                table.dataframe th, table.dataframe td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                table.dataframe th {{ background-color: #f0f0f0; }}
                .chart-container {{ margin-top: 15px; margin-bottom: 30px; text-align: center; }}
                img.chart {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #888; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>HKEX Southbound Market Insights</h2>
                <p>Date of Report: <strong>{current_date_str}</strong></p>
                
                <h3>Executive Summary</h3>
                <p>
                    This report provides an automated analysis of the HKEX Southbound trading data.
                    Key highlights include significant stock movements and newly tracked entities.
                    <em>(This section can be expanded with more sophisticated automated analysis or manually curated insights.)</em>
                </p>
        """

        # Movers Analysis
        html_body += "<h3>Daily Shareholding Movers (1-Day Change)</h3>"
        if analysis_data['top_movers_chart_cid']:
            html_body += f"""
            <p>Comparison between {current_date_str} and {analysis_data['d_minus_1_actual_date']}.</p>
            <div class="chart-container">
                <h4>Top {MOVERS_COUNT} Gainers (by Number of Shares)</h4>
                <img src="cid:{analysis_data['top_movers_chart_cid']}" alt="Top Movers Chart" class="chart">
            </div>
            """
        else:
            html_body += "<p>Top movers data or chart is unavailable.</p>"

        if analysis_data['bottom_movers_chart_cid']:
            html_body += f"""
            <div class="chart-container">
                <h4>Top {MOVERS_COUNT} Decliners (by Number of Shares)</h4>
                <img src="cid:{analysis_data['bottom_movers_chart_cid']}" alt="Bottom Movers Chart" class="chart">
            </div>
            """
        else:
            html_body += "<p>Bottom movers data or chart is unavailable.</p>"

        # New Additions
        html_body += "<h3>Newly Tracked Stocks</h3>"
        html_body += analysis_data.get('new_additions_1_day_html', "<p>1-Day new additions data unavailable.</p>")
        html_body += analysis_data.get('new_additions_5_day_html', "<p>5-Day new additions data unavailable.</p>")
        
        html_body += """
                <div class="footer">
                    <p>This is an automated report. Data is sourced from HKEX.</p>
                </div>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))

        # Attach images
        if analysis_data['top_movers_chart_bytes'] and analysis_data['top_movers_chart_cid']:
            img_top = MIMEImage(analysis_data['top_movers_chart_bytes'])
            img_top.add_header('Content-ID', f"<{analysis_data['top_movers_chart_cid']}>")
            msg.attach(img_top)
        
        if analysis_data['bottom_movers_chart_bytes'] and analysis_data['bottom_movers_chart_cid']:
            img_bottom = MIMEImage(analysis_data['bottom_movers_chart_bytes'])
            img_bottom.add_header('Content-ID', f"<{analysis_data['bottom_movers_chart_cid']}>")
            msg.attach(img_bottom)
            
        return msg

    def _send_email(self, msg):
        """Sends the composed email message."""
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.ehlo('Gmail')
                server.starttls()  # Enable security
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
            logger.info(f"Email sent successfully to: {EMAIL_RECIPIENTS}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)

    def process_update(self, current_date_key_str):
        """
        Processes a new data update: fetches data, performs analysis, generates and sends email.
        """
        logger.info(f"Processing update for date: {current_date_key_str}")
        self._refresh_available_dates() # Ensure date cache is up-to-date

        current_df = fetch_df_from_redis(self.redis_client, current_date_key_str)
        if current_df.empty:
            logger.error(f"Could not fetch current data for {current_date_key_str}. Aborting email process.")
            return

        current_date_dt = datetime.strptime(current_date_key_str, "%Y-%m-%d")
        analysis_data = {}

        # --- 1-Day Movers ---
        d_minus_1_target_dt = current_date_dt - timedelta(days=1)
        historical_df_d1, d_minus_1_actual_date = get_closest_available_df(
            self.redis_client, d_minus_1_target_dt, self.available_dates_cache
        )
        analysis_data['d_minus_1_actual_date'] = d_minus_1_actual_date or "N/A"

        top_movers_chart_bytes, bottom_movers_chart_bytes = None, None
        top_movers_cid, bottom_movers_cid = None, None

        if not historical_df_d1.empty:
            top_movers_df, bottom_movers_df = self._calculate_movers(current_df, historical_df_d1)
            if not top_movers_df.empty:
                top_movers_chart_bytes = self._generate_movers_chart(top_movers_df, f"Top {MOVERS_COUNT} Gainers by Shares ({current_date_key_str} vs {d_minus_1_actual_date})", color='green')
                if top_movers_chart_bytes: top_movers_cid = 'top_movers_chart'
            if not bottom_movers_df.empty:
                # For bottom movers, ensure share change is negative. If positive, they are smallest gainers, not decliners.
                bottom_movers_df_filtered = bottom_movers_df[bottom_movers_df['ShareChange'] < 0]
                if not bottom_movers_df_filtered.empty:
                    bottom_movers_chart_bytes = self._generate_movers_chart(bottom_movers_df_filtered, f"Top {MOVERS_COUNT} Decliners by Shares ({current_date_key_str} vs {d_minus_1_actual_date})", color='red')
                    if bottom_movers_chart_bytes: bottom_movers_cid = 'bottom_movers_chart'
                else:
                    logger.info("No stocks with negative share change found for bottom movers chart.")
        else:
            logger.warning(f"No historical data found around D-1 ({d_minus_1_target_dt.strftime('%Y-%m-%d')}) for movers calculation.")

        analysis_data['top_movers_chart_bytes'] = top_movers_chart_bytes
        analysis_data['top_movers_chart_cid'] = top_movers_cid
        analysis_data['bottom_movers_chart_bytes'] = bottom_movers_chart_bytes
        analysis_data['bottom_movers_chart_cid'] = bottom_movers_cid
        
        # --- New Additions ---
        # 1-Day New Additions
        if not historical_df_d1.empty:
            new_additions_1_day_df = self._calculate_new_additions(current_df, historical_df_d1)
            analysis_data['new_additions_1_day_html'] = self._format_new_additions_html(
                new_additions_1_day_df, f"~1 trading day ago ({d_minus_1_actual_date or 'N/A'})"
            )
        else:
            # If no D-1, all current stocks are "new" compared to nothingness.
            all_current_as_new_1_day = self._calculate_new_additions(current_df, pd.DataFrame()) # Pass empty df
            analysis_data['new_additions_1_day_html'] = self._format_new_additions_html(
                 all_current_as_new_1_day, "any prior day (D-1 data missing)"
            )
            logger.warning(f"No D-1 data for 1-day new additions; listing all current stocks if any.")

        # 5-Day New Additions
        d_minus_5_target_dt = current_date_dt - timedelta(days=5)
        historical_df_d5, d_minus_5_actual_date = get_closest_available_df(
            self.redis_client, d_minus_5_target_dt, self.available_dates_cache
        )
        analysis_data['d_minus_5_actual_date'] = d_minus_5_actual_date or "N/A"

        if not historical_df_d5.empty:
            new_additions_5_day_df = self._calculate_new_additions(current_df, historical_df_d5)
            analysis_data['new_additions_5_day_html'] = self._format_new_additions_html(
                new_additions_5_day_df, f"~5 trading days ago ({d_minus_5_actual_date or 'N/A'})"
            )
        else:
            all_current_as_new_5_day = self._calculate_new_additions(current_df, pd.DataFrame()) # Pass empty df
            analysis_data['new_additions_5_day_html'] = self._format_new_additions_html(
                 all_current_as_new_5_day, "any prior data (D-5 data missing)"
            )
            logger.warning(f"No D-5 data for 5-day new additions; listing all current stocks if any.")

        # Generate and send email
        email_msg = self._generate_email_content(current_date_key_str, current_df, analysis_data)
        self._send_email(email_msg)

    def listen_for_updates(self):
        """Listens to Redis Pub/Sub for new data update notifications."""
        if not self.redis_client:
            logger.critical("Redis client not initialized. Cannot listen for updates.")
            return

        pubsub = self.redis_client.pubsub()
        try:
            pubsub.subscribe(REDIS_UPDATE_CHANNEL)
            logger.info(f"Subscribed to Redis channel: {REDIS_UPDATE_CHANNEL}")
            logger.info("Waiting for new data notifications...")

            for message in pubsub.listen():
                if message and message['type'] == 'message':
                    date_key = message['data'].decode('utf-8')
                    logger.info(f"Received update notification for date key: {date_key}")
                    try:
                        # Validate date_key format
                        datetime.strptime(date_key, "%Y-%m-%d")
                        self.process_update(date_key)
                    except ValueError:
                        logger.error(f"Received invalid date key format: {date_key}. Skipping.")
                    except Exception as e:
                        logger.error(f"Error processing update for {date_key}: {e}", exc_info=True)
                    logger.info("Waiting for next data notification...")

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis Pub/Sub connection error: {e}. Attempting to reconnect...")
            time.sleep(10) # Wait before retrying
            self.listen_for_updates() # Recursive call to reconnect
        except Exception as e:
            logger.critical(f"Critical error in Pub/Sub listener: {e}", exc_info=True)
        finally:
            if pubsub:
                pubsub.close()
            logger.info("Pub/Sub listener stopped.")


if __name__ == "__main__":
    # --- Pre-run check for Matplotlib backend ---
    # For environments without a display server (like some Docker containers or headless servers),
    # Matplotlib might default to a backend that requires one, causing errors.
    # Setting it to 'Agg' is a common fix for non-interactive plotting.
    try:
        import matplotlib
        matplotlib.use('Agg') # Use a non-interactive backend suitable for generating files
        logger.info(f"Matplotlib backend set to: {matplotlib.get_backend()}")
    except Exception as e:
        logger.warning(f"Could not set Matplotlib backend to 'Agg': {e}. Charts might fail if no display is available.")

    # --- Test Email Sending (Optional) ---
    # You can uncomment this section to test email sending functionality on script start.
    # Make sure to fill in your SMTP details and recipient list above.
    # logger.info("Attempting to send a test email...")
    # test_msg = MIMEMultipart()
    # test_msg['Subject'] = "Email Automation Tool - Test Email"
    # test_msg['From'] = EMAIL_SENDER
    # test_msg['To'] = EMAIL_RECIPIENTS
    # test_msg.attach(MIMEText("This is a test email from the HKEX Email Automation Tool.", 'plain'))
    # try:
    #     server = smtplib.SMTP('smtp.gmail.com:587')
    #     server.ehlo('Gmail')
    #     server.starttls()
    #     server.login(SMTP_USERNAME, SMTP_PASSWORD)
    #     server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, test_msg.as_string())
    #     server.quit()
    #     logger.info("Test email sent successfully.")
    # except Exception as e:
    #     logger.error(f"Failed to send test email: {e}. Please check your SMTP configuration.")


    # --- Initialize and run the tool ---
    try:
        email_tool = HKEXEmailAutomationTool()
        email_tool.listen_for_updates()
    except redis.exceptions.ConnectionError:
        logger.critical("Could not start email automation due to Redis connection failure.")
    except KeyboardInterrupt:
        logger.info("Email automation tool stopped by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
