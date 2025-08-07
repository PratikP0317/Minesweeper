from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Minesweeper URL (confirm it's valid in your region)
URL = "https://www.google.com/fbx?fbx=minesweeper"

# Mapping of class names to values
CLASS_MAP = {
    "blank": -1,           # unrevealed
    "bombflagged": -2,     # flag
    "open0": 0,
    "open1": 1,
    "open2": 2,
    "open3": 3,
    "open4": 4,
    "open5": 5,
    "open6": 6,
    "open7": 7,
    "open8": 8,
}

def classify_tile(class_str):
    for key in CLASS_MAP:
        if key in class_str:
            return CLASS_MAP[key]
    return -1  # default to unrevealed

def extract_board(driver):
    board = []
    rows = driver.find_elements(By.CSS_SELECTOR, '.squares-row')
    for row_elem in rows:
        row = []
        cells = row_elem.find_elements(By.CSS_SELECTOR, '.square')
        for cell in cells:
            class_name = cell.get_attribute("class")
            val = classify_tile(class_name)
            row.append(val)
        board.append(row)
    return board

def click_center_tile(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, '.squares-row')
    if not rows:
        print("No rows found.")
        return
    mid_row = len(rows) // 2
    cells = rows[mid_row].find_elements(By.CSS_SELECTOR, '.square')
    if not cells:
        print("No cells found in middle row.")
        return
    mid_cell = len(cells) // 2
    cells[mid_cell].click()

def print_board(board):
    for row in board:
        print(" ".join(f"{x:2}" for x in row))

if __name__ == "__main__":
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(URL)
        time.sleep(3)  # wait for page to load

        click_center_tile(driver)
        time.sleep(1)  # wait for board to reveal

        board = extract_board(driver)
        print_board(board)
    finally:
        driver.quit()
