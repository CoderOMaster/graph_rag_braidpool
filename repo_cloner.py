import os
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Modify this based on your machine's user profile path
# 1. Open Edge, go to edge://version
# 2. Find your 'Profile path'.
# 3. USER_DATA_DIR is the part of 'Profile path' BEFORE the last folder (e.g., Default, Profile 1).
#    Example: If Profile path is C:\Users\YourUser\AppData\Local\Microsoft\Edge\User Data\Profile 2
#    Then USER_DATA_DIR = "C:/Users/YourUser/AppData/Local/Microsoft/Edge/User Data"
USER_DATA_DIR = "C:/Users/keshav/AppData/Local/Microsoft/Edge/User Data" # <-- VERIFY THIS AND YOUR USERNAME
# 4. PROFILE is the LAST folder name from 'Profile path'.
#    Example: If Profile path is ...\User Data\Profile 2, then PROFILE = "Profile 2"
PROFILE = "Personal"  # <-- VERIFY THIS (e.g., "Default", "Profile 1", "Profile 2")

OUTPUT_DIR = "vercel_project_files"
VERCEL_URL = "https://vercel.com/timothyng04-7249s-projects/agentzeta_pitchdeck_longterm/2xqmd8yCMBiQsqtMYW3eM9kCVLmg/source"

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with async_playwright() as p:
        # Launch real Edge with your profile
        browser_context = await p.chromium.launch_persistent_context(
            user_data_dir=f"{USER_DATA_DIR}/{PROFILE}",
            headless=False,
            channel="msedge", # use actual Microsoft Edge
            args=["--start-maximized"]
        )
        page = browser_context.pages[0] if browser_context.pages else await browser_context.new_page()

        print("Opening Vercel project with your logged-in Edge profile...")
        await page.goto(VERCEL_URL)
        try:
            print("Checking if already logged in...")
            await page.wait_for_selector('[data-testid="file-tree"]', timeout=5000) # Short timeout to check
            print("Already logged in or on the project page.")
        except Exception:
            print("Not logged in. Attempting 'Continue with Google' login flow.")
            try:
                google_login_button_selectors = [
                    'button:has-text("Continue with Google")',
                    'button:has-text("Log In with Google")',
                    'button:has-text("Sign In with Google")',
                    'a:has-text("Continue with Google")',
                    'a:has-text("Log In with Google")',
                    'a:has-text("Sign In with Google")',
                    '[data-provider-name="google"]', 
                    'button:has(svg[aria-label*="Google"])',
                    'button:near(:text("Google"))' 
                ]
                
                clicked_button = False
                for selector in google_login_button_selectors:
                    try:
                        print(f"Attempting to click Google login button with selector: {selector}")
                        await page.click(selector, timeout=1500) # Short timeout for each attempt
                        print(f"Successfully clicked button with selector: {selector}")
                        clicked_button = True
                        break
                    except Exception:
                        # print(f"Selector {selector} not found or failed to click.") # This can be too verbose
                        continue # Try next selector
                
                if not clicked_button:
                    print("Could not automatically find or click a 'Continue with Google' button.")
                    print("Please proceed with logging in manually in the browser window.")

                print("\nPlease complete the Google login process in the browser window that has opened.")
                print("The script will wait for up to 3 minutes for you to be redirected to the Vercel project page...")
                await page.wait_for_selector('[data-testid="file-tree"]', timeout=180000) # 3 minutes for manual login
                print("Successfully logged in and redirected to Vercel project page.")
            except Exception as e_login:
                print(f"An error occurred during the login attempt or while waiting for redirection: {e_login}")
                print("Please ensure you are logged in and have navigated to the correct Vercel project source page.")
                print("If the script cannot find the file tree, it will exit.")
                await browser_context.close()
                return # Exit main function if login fails catastrophically

        file_links = await page.query_selector_all('[data-testid="file-tree"] a')

        for file_link in file_links:
            href = await file_link.get_attribute("href")
            if href: # Ensure href is not None
                filename = href.split("/")[-1]

                await page.goto(f"https://vercel.com{href}")
                await page.wait_for_selector("pre", timeout=10000)

                code_element = await page.query_selector("pre")
                if code_element: # Ensure code_element is not None
                    code = await code_element.inner_text()
                    filepath = os.path.join(OUTPUT_DIR, filename)

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(code)
                        print(f"✅ Saved: {filename}")
                else:
                    print(f"⚠️ Could not find code block for: {filename} at {f'https://vercel.com{href}'}")
            else:
                print(f"⚠️ Found a file link with no href attribute.")


        print(f"\n🎉 Done. All files saved to: `{OUTPUT_DIR}/`")
        await browser_context.close()

if __name__ == "__main__":
    # If running as a .py script:
    asyncio.run(main())
    # If running in a Jupyter Notebook cell, you might need to run it like this:
    # await main()
    # Or, if the above doesn't work due to an existing event loop:
    # loop = asyncio.get_event_loop()
    # if loop.is_running():
    #     print("Async event loop already running. Awaiting main.")
    #     # Create a task to run the main coroutine
    #     # This is a common way to run async code from within an already running loop (like in Jupyter)
    #     # However, directly awaiting might be sufficient in modern Jupyter versions.
    #     # If `await main()` directly in a cell gives an error, try this:
    #     # asyncio.ensure_future(main())
    # else:
    #     print("Starting new async event loop.")
    #     asyncio.run(main())
