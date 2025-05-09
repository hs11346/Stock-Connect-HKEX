import logging
import time
import json
from datetime import datetime, timedelta

import pandas as pd
import redis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import UnexpectedAlertPresentException, TimeoutException, NoSuchElementException
import warnings

warnings.filterwarnings('ignore')
# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 3
REDIS_UPDATE_CHANNEL = 'data_updated'
LOG_FILE = 'scraper.log'
DEFAULT_SCRAPING_INTERVAL_SECONDS = 60  # 1 minute
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 10
SCRAPE_CURRENT_DAY_CONTINUOUSLY = True # Toggle for continuous scraping
DAYS_FOR_INITIAL_POPULATION = 60

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class HKEXScraper:
    """
    A class to scrape shareholding data from the HKEXnews website.
    It handles WebDriver initialization, page interactions, data extraction,
    error handling, logging, and storing data in Redis.
    """
    BASE_URL = "https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk"

    def __init__(self, redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB):
        """
        Initializes the HKEXScraper.

        Args:
            redis_host (str): Hostname for the Redis server.
            redis_port (int): Port number for the Redis server.
            redis_db (int): Redis database number.
        """
        self.driver = None
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
        logger.info("HKEXScraper initialized. Redis connection configured for %s:%s DB %s", redis_host, redis_port, redis_db)

    def _initialize_driver(self):
        """Initializes a headless Chrome WebDriver."""
        if self.driver:
            self.close_driver()

        logger.info("Initializing WebDriver...")
        options = Options()
        options.add_argument('--headless=new')
        

        try:
            # Using WebDriverManager to automatically handle driver versions
            self.driver = webdriver.Chrome(options=options)
            logger.info("WebDriver initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize WebDriver: %s", e, exc_info=True)
            raise  # Re-raise the exception to be handled by the calling method

    def close_driver(self):
        """Closes the WebDriver if it's running."""
        if self.driver:
            logger.info("Closing WebDriver...")
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully.")
            except Exception as e:
                logger.error("Error closing WebDriver: %s", e, exc_info=True)
            finally:
                self.driver = None

    def _accept_cookies(self):
        """Accepts cookies on the page if the button is present."""
        try:
            logger.info("Attempting to accept cookies...")
            time.sleep(2) # Wait for the cookie banner to appear
            accept_button = self.driver.find_element(By.ID, "onetrust-accept-btn-handler")
            accept_button.click()
            logger.info("Cookies accepted.")
            time.sleep(1) # Wait for overlay to disappear
        except NoSuchElementException:
            logger.info("Cookie consent button not found or already accepted.")
        except Exception as e:
            logger.warning("Could not click cookie consent button: %s", e)


    def get_data_from_date(self, target_date: datetime) -> pd.DataFrame | None:
        """
        Fetches shareholding data for a specific date.

        Args:
            target_date (datetime): The date for which to fetch data.

        Returns:
            pd.DataFrame: A DataFrame containing the shareholding data, or None if no data is found
                          or an error occurs.
        """
        if not self.driver:
            self._initialize_driver()
            self.driver.get(self.BASE_URL)
            self._accept_cookies()


        date_str_input = target_date.strftime("%Y/%m/%d")
        logger.info("Fetching data for date: %s", date_str_input)

        try:
            # Ensure the date input field is present and interactable
            date_input_element = self.driver.find_element(By.ID, "txtShareholdingDate")
            
            # Set the date value using JavaScript
            script = f"arguments[0].value = '{date_str_input}';"
            self.driver.execute_script(script, date_input_element)
            logger.debug("Date %s set in input field.", date_str_input)

            # Click the search button
            search_button = self.driver.find_element(By.ID, "btnSearch")
            search_button.click()
            logger.debug("Search button clicked.")

            time.sleep(3) # Increased wait time for data to load

            # Verify the displayed date
            try:
                current_date_element = self.driver.find_element(By.CLASS_NAME, "ccass-heading")
                displayed_date_text = current_date_element.text
                # Example: "Shareholding Date: DD/MM/YYYY\nDetail of Shareholding:"
                # Need to parse this carefully
                actual_displayed_date_str = displayed_date_text.split('\n')[0].replace("Shareholding Date: ", "").strip()
                # Convert actual_displayed_date_str (DD/MM/YYYY) to YYYY/MM/DD for comparison
                if actual_displayed_date_str:
                    parsed_displayed_date = datetime.strptime(actual_displayed_date_str, "%Y/%m/%d")
                    formatted_displayed_date = parsed_displayed_date.strftime("%Y/%m/%d")
                    logger.info("Data displayed for date: %s", formatted_displayed_date)
                    if formatted_displayed_date != date_str_input:
                        logger.warning("Requested date %s but page shows %s. This might indicate no data or a redirect.",
                                       date_str_input, formatted_displayed_date)
                        # This could be a case where data for the exact date is missing, and HKEX shows the nearest available.
                        # Depending on requirements, one might choose to return None or proceed.
                        # For now, we proceed and let the table check handle it.
                else:
                    logger.warning("Could not extract displayed date from header: '%s'", displayed_date_text)
                    return None
            except NoSuchElementException:
                logger.warning("Could not find CCASS heading to verify date. Page might not have loaded correctly.")
                return None
            except Exception as e:
                logger.warning("Error verifying displayed date: %s", e)
                return None


            # Obtain table data
            table_elements = self.driver.find_elements(By.TAG_NAME, "table")
            if len(table_elements) < 2:
                logger.warning("Data table not found for %s. It's possible no data exists for this date.", date_str_input)
                return None

            table_html = table_elements[1].get_attribute('outerHTML')
            df = pd.read_html(table_html)[0]

            # Clean DataFrame
            df.columns = ['Stock Code', 'Name', 'Shareholding in CCASS', '% of the total number of Issued Shares/Units']
            df['Stock Code'] = df['Stock Code'].str.replace("Stock Code:", "", regex=False).str.strip()
            df['Name'] = df['Name'].str.replace("Name:", "", regex=False).str.strip()
            df['Shareholding in CCASS'] = df['Shareholding in CCASS'].str.replace("Shareholding in CCASS:", "", regex=False).str.strip()
            df[r'% of the total number of Issued Shares/Units'] = df[r'% of the total number of Issued Shares/Units'].str.replace(r'% of the total number of Issued Shares/Units:', "", regex=False).str.strip()

            df['Shareholding in CCASS'] = df['Shareholding in CCASS'].str.replace(",", "").astype(int)
            df['Stock Code'] = pd.to_numeric(df['Stock Code'], errors='coerce') # Handle potential non-numeric codes if any
            df = df.dropna(subset=['Stock Code']) # Remove rows where stock code could not be parsed
            df['Stock Code'] = df['Stock Code'].astype(int)

            df[r'% of the total number of Issued Shares/Units'] = df[r'% of the total number of Issued Shares/Units'].str.replace("%", "")
            df[r'% of the total number of Issued Shares/Units'] = pd.to_numeric(df[r'% of the total number of Issued Shares/Units'], errors='coerce')
            
            df['ScrapeDate'] = target_date # Add the scrape date for reference

            logger.info("Successfully fetched and parsed data for %s. Shape: %s", date_str_input, df.shape)
            return df

        except UnexpectedAlertPresentException:
            logger.warning("No data available for date %s (UnexpectedAlertPresentException).", date_str_input)
            # Try to handle the alert by accepting it, so the driver can continue
            try:
                alert = self.driver.switch_to.alert
                alert.accept()
                logger.info("Accepted an unexpected alert.")
            except Exception as alert_e:
                logger.warning("Could not handle unexpected alert: %s", alert_e)
            return None
        except TimeoutException:
            logger.error("Timeout occurred while fetching data for %s.", date_str_input, exc_info=True)
            return None
        except NoSuchElementException as e:
            logger.error("A specific element was not found for %s: %s", date_str_input, e, exc_info=True)
            return None
        except Exception as e:
            logger.error("General error fetching data for %s: %s", date_str_input, e, exc_info=True)
            return None

    def _store_dataframe_in_redis(self, df: pd.DataFrame, date_key: str):
        """
        Serializes a DataFrame to JSON and stores it in Redis.

        Args:
            df (pd.DataFrame): The DataFrame to store.
            date_key (str): The Redis key (YYYY-MM-DD format).
        """
        try:
            # Convert all column names to string to prevent issues with non-string column names
            df.columns = df.columns.astype(str)
            json_data = df.to_json(orient='split', date_format='iso')
            self.redis_client.set(date_key, json_data)
            logger.info("Data for %s stored in Redis. Size: %s bytes", date_key, len(json_data))
        except Exception as e:
            logger.error("Failed to store DataFrame for %s in Redis: %s", date_key, e, exc_info=True)

    def _publish_update_notification(self, date_key: str):
        """Publishes a message to Redis channel indicating data update."""
        try:
            self.redis_client.publish(REDIS_UPDATE_CHANNEL, date_key)
            logger.info("Published update notification to channel '%s' for date %s.", REDIS_UPDATE_CHANNEL, date_key)
        except Exception as e:
            logger.error("Failed to publish update notification for %s: %s", date_key, e, exc_info=True)


    def populate_historical_data(self, num_days: int = DAYS_FOR_INITIAL_POPULATION):
        """
        Populates Redis with historical data for the past `num_days`.

        Args:
            num_days (int): The number of past days to fetch data for.
        """
        logger.info("Starting initial population of historical data for the past %s days.", num_days)
        current_date = datetime.today()
        if not self.driver: # Ensure driver is initialized for the first run
            self._initialize_driver()
            self.driver.get(self.BASE_URL)
            self._accept_cookies()


        for i in range(num_days):
            target_date = current_date - timedelta(days=i)
            date_key = target_date.strftime("%Y-%m-%d")
            
            # Check if data already exists in Redis to avoid re-scraping unless necessary
            if self.redis_client.exists(date_key):
                logger.info("Data for %s already exists in Redis. Skipping.", date_key)
                continue

            logger.info("Attempting to fetch historical data for %s (Day %s/%s)", date_key, i + 1, num_days)
            
            df = None
            for attempt in range(DEFAULT_RETRY_ATTEMPTS):
                try:
                    df = self.get_data_from_date(target_date)
                    if df is not None and not df.empty:
                        self._store_dataframe_in_redis(df, date_key)
                        # No pub/sub for historical data unless specified
                        break  # Success
                    elif df is None: # No data for this date (e.g. holiday)
                        logger.info("No data found for %s after get_data_from_date call. Likely a non-trading day.", date_key)
                        break # Not an error, just no data
                except Exception as e:
                    logger.error("Error during historical data fetch for %s (Attempt %s/%s): %s",
                                 date_key, attempt + 1, DEFAULT_RETRY_ATTEMPTS, e, exc_info=True)
                    if attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                        logger.info("Retrying in %s seconds...", DEFAULT_RETRY_DELAY_SECONDS)
                        time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                        # Re-initialize driver if it crashed
                        if not self.driver or not self.driver.session_id:
                            logger.warning("WebDriver seems to have crashed. Re-initializing for retry.")
                            self.close_driver()
                            self._initialize_driver()
                            self.driver.get(self.BASE_URL) # Important to get URL again
                            self._accept_cookies()
                    else:
                        logger.critical("Failed to fetch historical data for %s after %s attempts.",
                                        date_key, DEFAULT_RETRY_ATTEMPTS)
            if df is None:
                 logger.info("No data obtained for historical date %s.", date_key)

        logger.info("Historical data population process finished.")


    def scrape_current_day_data(self,current_date) -> bool: # adjusted for selectable date
        """
        Scrapes data for the current day and stores it in Redis.
        Retries on failure.

        Returns:
            bool: True if data was successfully scraped and stored, False otherwise.
        """
        date_key = current_date.strftime("%Y-%m-%d")
        logger.info("Attempting to scrape data for current day: %s", date_key)

        df = None
        for attempt in range(DEFAULT_RETRY_ATTEMPTS):
            try:
                df = self.get_data_from_date(current_date)
                if df is not None and not df.empty:
                    self._store_dataframe_in_redis(df, date_key)
                    self._publish_update_notification(date_key) # Publish on successful update
                    logger.info("Successfully scraped and stored current day's data for %s.", date_key)
                    return True
                elif df is None: # No data available from get_data_from_date
                    logger.warning("No data available for current day %s (get_data_from_date returned None).", date_key)
                    # This could be due to market being closed or data not yet published.
                    # We don't treat this as a critical failure for retry, but log and wait for next cycle.
                    return False # Indicate no new data was fetched
                # If df is empty but not None, it's an issue, retry.
                elif df.empty:
                     logger.warning("Fetched empty DataFrame for current day %s. Retrying...", date_key)
                     # Fall through to retry logic

            except Exception as e:
                logger.error("Error scraping current day's data for %s (Attempt %s/%s): %s",
                             date_key, attempt + 1, DEFAULT_RETRY_ATTEMPTS, e, exc_info=True)
            
            if attempt < DEFAULT_RETRY_ATTEMPTS - 1:
                logger.info("Retrying in %s seconds...", DEFAULT_RETRY_DELAY_SECONDS)
                time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                # Re-initialize driver if it crashed
                if not self.driver or not self.driver.session_id:
                    logger.warning("WebDriver seems to have crashed. Re-initializing for retry.")
                    self.close_driver()
                    self._initialize_driver()
                    self.driver.get(self.BASE_URL)
                    self._accept_cookies()
            else:
                logger.critical("Failed to scrape current day's data for %s after %s attempts.",
                                date_key, DEFAULT_RETRY_ATTEMPTS)
        return False


    def run_continuous_scraper(self, interval_seconds: int = DEFAULT_SCRAPING_INTERVAL_SECONDS,
                               scrape_continuously_toggle: bool = SCRAPE_CURRENT_DAY_CONTINUOUSLY):
        """
        Runs the scraper continuously for the current day's data at a specified interval.

        Args:
            interval_seconds (int): The interval in seconds between scraping attempts.
            scrape_continuously_toggle (bool): If False, the continuous scraping loop will not run.
        """
        if not scrape_continuously_toggle:
            logger.info("Continuous scraping for current day is disabled by configuration.")
            return

        logger.info("Starting continuous scraping for current day data. Interval: %s seconds.", interval_seconds)
        try:
            while True:
                logger.info("--- New scraping cycle started ---")
                self.scrape_current_day_data()
                logger.info("--- Scraping cycle finished. Waiting for %s seconds... ---", interval_seconds)
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Continuous scraping stopped by user (KeyboardInterrupt).")
        except Exception as e:
            logger.critical("Critical error in continuous scraping loop: %s", e, exc_info=True)
        finally:
            self.close_driver()
            logger.info("Continuous scraper shut down.")


if __name__ == "__main__":
    scraper = HKEXScraper()
    
    # --- Initial Data Population ---
    # Check if a special flag/argument is passed to run initial population,
    # or perhaps check if Redis is empty for relevant date ranges.
    # For this example, let's assume we run it if a certain key is not present.
    # A more robust way would be a command-line argument or config.
    INITIAL_POP_MARKER_KEY = "initial_population_done_marker"

    # --- Continuous Scraping for Current Day ---
    # Ensure driver is ready for continuous mode if it was closed

    scraper.scrape_current_day_data(datetime(2025, 5, 9))

    # Fallback close driver if continuous mode is off and script somehow reaches here
    if not SCRAPE_CURRENT_DAY_CONTINUOUSLY:
        scraper.close_driver()

