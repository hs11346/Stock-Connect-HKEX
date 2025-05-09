# HKEX Daily Shareholding Reporting System

## Project Goal

This project aims to provide a real-time Streamlit visualization dashboard & email automation system to display and analyze stock shareholding data scraped from the HKEXnews website. It allows users to track daily shareholding information and identify trends and significant changes.

## System Architecture

The system is composed of Four main components:

1.  **Backend Scraper (Python):** A Python application responsible for continuously scraping daily shareholding data from the HKEXnews website. It uses Selenium for web interaction.
2.  **Data Store (Redis):** Redis serves as the primary database and cache. It stores the scraped shareholding data (Pandas DataFrames serialized as JSON) with dates as keys. The backend publishes update notifications to a Redis channel.
3.  **Frontend (Streamlit):** A Streamlit web application that subscribes to Redis updates and visualizes the data. It offers interactive charts, including time-series line charts for individual stock shareholdings and bar charts for top/bottom movers based on customizable criteria.
4. **Email Automation:** A Email Automation System that receives updates from the backend, and generates charts and updates to recipients. 

## Features

### Backend Scraper:
* Automated daily scraping of shareholding data.
* Configurable scraping frequency (default: every minute, toggleable).
* Initial population of historical data (past 60 days).
* Robust error handling with retries and detailed logging to `scraper.log`.
* Data stored in Redis with "YYYY-MM-DD" keys and DataFrame values.
* Publishes notifications to Redis channel `data_updated` upon new data.
* Selenium WebDriver management encapsulated within a class.

### Data Store (Redis):
* Stores daily shareholding DataFrames.
* Data serialized as JSON for efficient storage and retrieval.
* No data expiration (retains all historical data).
* Supports Pub/Sub mechanism for real-time frontend updates.

### Frontend (Streamlit):
* Real-time data updates via Redis Pub/Sub.
* **Line Chart:**
    * Displays 'Shareholding in CCASS' over time for a selected stock.
    * Stock selection via a searchable dropdown (by Name or Code).
* **Bar Chart (Top/Bottom Movers):**
    * Displays top/bottom N stocks based on shareholding changes.
    * Customizable change period (1, 5, 20 days).
    * Customizable change metric (% or absolute number of shares).
    * Customizable display scope (Top 5/10, Bottom 5/10).
    * Handles missing data for change calculations by using the closest available previous trading day.
* User-friendly interface with controls in the sidebar.
* Graceful error handling for data unavailability or connection issues.

### Email Automation:
* Real time updates from Backend server via Redis Pub/Sub
* Generates Top and bottom movers chart
* Display new additions to the Southbound db

## Setup and Installation

### Prerequisites
* Python (version 3.8+ recommended)
* Redis server installed and running.
* Google Chrome browser installed (the scraper uses headless Chrome).
* ChromeDriver (Selenium WebDriver will attempt to manage this automatically if `selenium-manager` is available, otherwise ensure it's in your PATH and compatible with your Chrome version).

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (or add specific versions as needed):
    ```
    streamlit
    pandas
    redis
    selenium
    plotly
    requests
    smtplib
    email
    matplotlib
    # Add any other specific libraries used
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Redis:**
    Ensure your Redis server is running. By default, the applications will try to connect to `localhost:6379`. If your Redis configuration is different, you may need to update the connection parameters in the backend scraper and frontend scripts (or use a configuration file/environment variables as implemented).

5.  **Configure Scraper (Optional):**
    * **Scraping Toggle & Interval:** The scraping frequency (default: 1 minute) and the toggle to enable/disable continuous scraping should be configurable. This might be managed via a `config.ini` file or environment variables (e.g., `SCRAPING_ENABLED=True`, `SCRAPING_INTERVAL_SECONDS=60`). Refer to the specific implementation.

4.  **Configure Env file:**
    Ensure your email credentials are in the environment file, which can be loaded by the `email_automation.py file`. 

## Running the Application

1.  **Start Redis Server:**
    If not already running, start your Redis server.
    ```bash
    redis-server
    ```
    (The command might vary based on your Redis installation.)

2.  **Run the Backend Scraper:**
    Navigate to the directory containing the backend scraper script (e.g., `backend_scraper.py`) and run it:
    ```bash
    python backend_scraper.py
    ```
    Check `scraper.log` for logging output and any errors. The scraper will first populate historical data (60 days) and then start its continuous scraping routine if enabled.

3.  **Run the Streamlit Frontend:**
    Navigate to the directory containing the Streamlit frontend script (e.g., `app.py` or `frontend_streamlit.py`) and run it:
    ```bash
    streamlit run app.py
    ```
    Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Run the Email Automation System:**
    Navigate to the directory containing the email automation script (e.g., `email_automation.py`) and run it:
    ```bash
    python email_automation.py
    ```

## Logging
The backend scraper logs its activities, errors, and retry attempts to `scraper.log` in the same directory where the scraper script is run.

## Future Enhancements (Optional Suggestions)
* More advanced error notification system (e.g., email alerts for critical failures).
* User authentication for the dashboard.
* More sophisticated data analysis features (e.g., moving averages, correlation analysis).
* Dockerization for easier deployment.
* Persistent storage solution beyond Redis for long-term archival if Redis memory becomes a concern.

## Contributing
Feel free to fork it for your own use.
