import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

import sys

sys.path.insert(0, os.path.dirname(__file__))
from hero_details_io import (  # noqa: E402
    DEFAULT_HERO_DETAILS_PATH,
    load_hero_details_by_code,
    save_hero_detail_rows,
    sorted_hero_detail_rows,
    upsert_hero_detail_row,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.chrome.options import Options

# Initialize Chrome options
chrome_options = Options()
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--headless')

# Initialize the Chrome driver
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

# Open the initial URL to establish session/cookies
initial_url = 'https://epic7.onstove.com/en/gg/herorecord'
driver.get(initial_url)

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.filterlist-wrap ul li')))
time.sleep(10) # For some reason some eleemnts are not loaded together causing errors

# Create the 'dataset' directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Optional: E7_CHARACTER_IDS_LIMIT=N to only process first N heroes (smoke test)
HERO_LIMIT = int(os.environ.get("E7_CHARACTER_IDS_LIMIT", "0") or "0")

# Find all elements with class 'filterlist-wrap' (assuming this contains hero information)
try:
    # Wait until all hero elements are present
    heros = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.filterlist-wrap ul li')))
    
    # List to store hero data
    hero_data = []

    # Iterate through each hero element and extract hero code and name
    for idx, hero_element in enumerate(heros):
        if HERO_LIMIT > 0 and idx >= HERO_LIMIT:
            print(f"Stopping at E7_CHARACTER_IDS_LIMIT={HERO_LIMIT}")
            break
        try:
            # Extract hero code and portrait URL from the page image.
            img_element = hero_element.find_element(By.TAG_NAME, 'img')
            hero_code = img_element.get_attribute('alt')
            image_url = img_element.get_attribute('src')

            # Extract hero name (from i tag)
            hero_name = hero_element.find_element(By.TAG_NAME, 'i').text
            
            # Append to hero_data list as dictionary
            hero_data.append({'code': hero_code, 'name': hero_name})
            print(f"Found hero: {hero_code} - {hero_name}")
            
            # Create a new folder for the hero in the 'dataset' directory
            hero_dir = os.path.join('dataset', hero_code)
            os.makedirs(hero_dir, exist_ok=True)
            
            # Define the local path
            image_path = os.path.join(hero_dir, 'c.png')
            
            # Download the image and save it locally
            response = requests.get(image_url, timeout=30)
            content_type = response.headers.get('Content-Type', '')
            if response.status_code == 200 and response.content and 'image' in content_type:
                with open(image_path, 'wb') as img_file:
                    img_file.write(response.content)
            else:
                print(f"Failed to download image for {hero_code}: {response.status_code} {content_type}")
            
        except Exception as e:
            print(f"Error retrieving hero information: {e}")
            continue  # Continue to next hero
    
    # Save hero metadata to hero_details.csv, preserving zh names and role/element stats.
    existing_rows = load_hero_details_by_code(DEFAULT_HERO_DETAILS_PATH)
    merged_rows = list(existing_rows.values())

    for item in hero_data:
        code = item["code"]
        previous = existing_rows.get(code, {})
        upsert_hero_detail_row(
            merged_rows,
            {
                "Hero": code,
                "name": item["name"],
                "name_zh": previous.get("name_zh", ""),
                "Role": previous.get("Role", ""),
                "Element": previous.get("Element", ""),
                "appearance_count": previous.get("appearance_count", "0"),
            },
        )

    save_hero_detail_rows(sorted_hero_detail_rows(merged_rows), DEFAULT_HERO_DETAILS_PATH)
    
    print(f"Saved hero metadata to {DEFAULT_HERO_DETAILS_PATH}")

except Exception as e:
    print(f"Error finding or saving hero elements: {e}")

# Close the driver
driver.quit()
