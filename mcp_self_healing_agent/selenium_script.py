from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import traceback
import requests
import json
import time

def run_test(locator=None):
    driver = webdriver.Chrome()
    driver.get("https://www.google.com")
    error = None
    stack = None
    try:
        # Accept cookies if the prompt appears (optional, for EU users)
        try:
            consent = driver.find_element(By.ID, "L2AGLb")
            consent.click()
            time.sleep(1)
        except Exception:
            pass

        # Enter text in the search box
        search_box = driver.find_element(By.NAME, "q")
        search_box.clear()
        search_box.send_keys("Self healing automation agent")

        # Use provided locator or default (which will fail)
        if locator:
            by, value = locator
            driver.find_element(getattr(By, by.upper()), value).click()
        else:
            # Intentionally incorrect locator for Google Search button
            driver.find_element(By.ID, "nonexistent-google-search-btn").click()

        print("Test passed: Search button clicked.")
        driver.quit()
        return True
    except Exception as e:
        error = str(e) or "N/A"
        stack = traceback.format_exc() or "N/A"
        print("Test failed!")
        print("Error:", error)
        print("Stack Trace:", stack)
        # Collect failure context, ensure all fields are non-empty
        try:
            dom_snapshot = driver.page_source
        except Exception:
            dom_snapshot = "N/A"
        context = {
            "test_name": "test_google_search_button",
            "error": error,
            "stack_trace": stack,
            "dom_snapshot": dom_snapshot
        }
        print("Failure context being sent to Lambda:", context)
        driver.quit()
        return context

if __name__ == "__main__":
    result = run_test()
    if isinstance(result, dict):
        # Send to Lambda via API Gateway
        api_url = "https://ervo9cueok.execute-api.us-west-2.amazonaws.com/Test/self-heal"
        print("Payload being sent to Lambda:", json.dumps(result, indent=2))
        response = requests.post(api_url, json=result)
        data = response.json()
        print("Lambda suggestion (raw):", data)
        # Parse new locator if available
        body = data.get("body")
        if body:
            try:
                body = json.loads(body)
            except Exception:
                print("Could not parse Lambda body as JSON:", body)
                body = {}
            new_locator = body.get("new_locator")
            suggestion = body.get("suggestion")
            print("Full Lambda suggestion text:", suggestion)
            if new_locator:
                # Accept any locator with a comma and not empty
                if new_locator.count(",") < 1 or new_locator.strip() in ("", "N/A"):
                    print("Warning: Locator from Lambda appears incomplete or invalid:", new_locator)
                else:
                    print("Applying new locator and rerunning test:", new_locator)
                    try:
                        by, value = new_locator.replace('By.', '').split(',', 1)
                        by = by.strip()
                        value = value.strip().strip('"').strip("'")
                        rerun_result = run_test(locator=(by, value))
                        if rerun_result is True:
                            print("Test passed after applying fix!")
                            # Store fix in knowledge base
                            try:
                                from mcp_self_healing_agent.fixes_store import save_fix
                                save_fix("test_google_search_button", result["error"], suggestion)
                            except Exception as store_err:
                                print("Could not save fix to knowledge base:", store_err)
                        else:
                            print("Test still failed after applying fix.")
                    except Exception as parse_err:
                        print("Could not parse or apply new locator:", parse_err)
            else:
                print("No fix suggested by Lambda.")
        else:
            print("No valid response from Lambda.")