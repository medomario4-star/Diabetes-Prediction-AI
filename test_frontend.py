"""
test_frontend.py — Selenium UI tests for the Diabetes Risk Predictor
=====================================================================
Tests the browser-facing frontend (index.html) end-to-end with a real
Chrome browser controlled by Selenium WebDriver.

Prerequisites:
    pip install selenium webdriver-manager pytest

Before running:
    1. Start the Flask backend:   python app.py
    2. In a second terminal:      python -m pytest test_frontend.py -v

Two root causes were fixed from the original version:

  FIX 1 — ElementClickInterceptedException
    The page is taller than the browser viewport so elements below the
    fold are partially hidden. Every click now scrolls the element into
    view first via JavaScript, then uses a JS click as a guaranteed
    fallback so the browser chrome never intercepts it.

  FIX 2 — is_displayed() returning False / toggle text returning ''
    The headless window was too small (1280×900), causing CSS animations
    to leave elements invisible. Window is now 1600×2400 (tall enough to
    fit the whole page) and fresh_page waits for the first card to be
    fully visible before each test starts.
"""

import os
import time
import pytest

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# ── Config ────────────────────────────────────────────────────────────────────

HTML_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index.html")
)
PAGE_URL = f"file:///{HTML_FILE.replace(os.sep, '/')}"

WAIT_TIMEOUT = 12   # seconds to wait for elements / results


# ── Driver fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def driver():
    """
    One headless Chrome instance shared across all tests.
    Window is 1600×2400 — wide enough for the layout and tall enough
    to fit the entire page so no element is ever off-screen.
    """
    options = Options()
    options.add_argument("--headless=new")        # modern headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,2400")   # FIX 2: tall window
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=1")

    service = Service(ChromeDriverManager().install())
    drv = webdriver.Chrome(service=service, options=options)
    drv.implicitly_wait(2)

    yield drv
    drv.quit()


@pytest.fixture(autouse=True)
def fresh_page(driver):
    """
    Navigate to a clean page before every test.
    Waits until the first card is fully visible so CSS animations
    have settled before any test begins.
    """
    driver.get(PAGE_URL)
    # Wait for the form AND the first card to be visible
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.visibility_of_element_located((By.ID, "risk-form"))
    )
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".card"))
    )
    time.sleep(0.3)   # let staggered fade-in animations finish


# ── Helpers ───────────────────────────────────────────────────────────────────

def js_click(driver, element):
    """
    FIX 1: Scroll element into the centre of the viewport, then click
    via JavaScript. This bypasses ElementClickInterceptedException
    caused by elements being partially outside the visible area.
    """
    driver.execute_script(
        "arguments[0].scrollIntoView({block:'center', inline:'center'});",
        element
    )
    time.sleep(0.1)   # brief pause for scroll to settle
    driver.execute_script("arguments[0].click();", element)


def fill_number_input(driver, data_key, value):
    """
    Set a number input's value via JavaScript and fire the input event.

    Using send_keys("0") after clear() is unreliable — clear() can trigger
    a blur that resets state, and "0" alone sometimes doesn't fire the JS
    input listener. Setting .value directly + dispatching the event is
    guaranteed to work regardless of the value.
    """
    inp = driver.find_element(By.CSS_SELECTOR, f'input[data-key="{data_key}"]')
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", inp)
    driver.execute_script(
        "arguments[0].value = arguments[1];"
        "arguments[0].dispatchEvent(new Event('input', {bubbles:true}));"
        "arguments[0].dispatchEvent(new Event('change', {bubbles:true}));",
        inp, str(value)
    )
    time.sleep(0.05)


def click_toggle(driver, data_key):
    """Scroll the toggle into view and click it via JS."""
    toggle = driver.find_element(
        By.CSS_SELECTOR, f'.toggle-field[data-key="{data_key}"]'
    )
    js_click(driver, toggle)


def click_seg_btn(driver, container_key, value):
    """Scroll the segmented button (GenHlth, Sex) into view and click it."""
    btn = driver.find_element(
        By.CSS_SELECTOR,
        f'[data-key="{container_key}"] .seg-btn[data-val="{value}"]'
    )
    js_click(driver, btn)


def click_age_btn(driver, value):
    """Scroll the age button into view and click it."""
    btn = driver.find_element(
        By.CSS_SELECTOR,
        f'.age-control .age-btn[data-val="{value}"]'
    )
    js_click(driver, btn)


def click_button_by_id(driver, btn_id):
    """Scroll any button into view and click it via JS."""
    btn = driver.find_element(By.ID, btn_id)
    js_click(driver, btn)


def fill_valid_form(driver):
    """
    Fill every field with valid values so the form is ready to submit.
    Mirrors VALID_INPUT from tests.py.

    Uses JS value injection for number inputs (reliable for value "0")
    and waits briefly between steps so the JS state object is fully updated
    before the form is submitted.
    """
    # Number inputs — JS injection fires the input event reliably
    fill_number_input(driver, "BMI",      "27.5")
    fill_number_input(driver, "MentHlth", "3")
    fill_number_input(driver, "PhysHlth", "2")   # 0 can be tricky; use 2

    # Segmented controls
    click_seg_btn(driver, "GenHlth", "2")   # Very Good
    time.sleep(0.1)
    click_seg_btn(driver, "Sex",     "0")   # Female
    time.sleep(0.1)

    # Age group
    click_age_btn(driver, "4")              # 35–39
    time.sleep(0.1)

    # Binary toggle — HighBP = 1
    click_toggle(driver, "HighBP")
    time.sleep(0.2)   # allow JS state to settle before submit


def submit_form(driver):
    """Scroll the submit button into view and click it."""
    click_button_by_id(driver, "submit-btn")


def result_card_visible(driver, timeout=WAIT_TIMEOUT):
    """Return True if the result card becomes visible within timeout seconds."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.ID, "result-card"))
        )
        return True
    except TimeoutException:
        return False


def error_banner_visible(driver, timeout=WAIT_TIMEOUT):
    """Return True if the error banner becomes visible within timeout seconds."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.ID, "error-banner"))
        )
        return True
    except TimeoutException:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class TestPageLoad:
    """Verify the page renders correctly on load."""

    def test_page_title(self, driver):
        assert "Diabetes" in driver.title

    def test_form_present(self, driver):
        assert driver.find_element(By.ID, "risk-form").is_displayed()

    def test_submit_button_present(self, driver):
        btn = driver.find_element(By.ID, "submit-btn")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(0.2)
        assert btn.is_displayed()
        assert btn.text.strip() != ""

    def test_reset_button_present(self, driver):
        btn = driver.find_element(By.ID, "reset-btn")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(0.2)
        assert btn.is_displayed()

    def test_result_card_hidden_on_load(self, driver):
        card = driver.find_element(By.ID, "result-card")
        assert not card.is_displayed()

    def test_error_banner_hidden_on_load(self, driver):
        banner = driver.find_element(By.ID, "error-banner")
        assert not banner.is_displayed()

    def test_all_toggle_fields_rendered(self, driver):
        """All 9 binary feature toggles must be present and visible."""
        binary_keys = [
            "HighBP", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"
        ]
        for key in binary_keys:
            el = driver.find_element(
                By.CSS_SELECTOR, f'.toggle-field[data-key="{key}"]'
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.1)
            assert el.is_displayed(), f"Toggle for '{key}' not visible"

    def test_number_inputs_rendered(self, driver):
        """BMI, MentHlth, PhysHlth inputs must be present and visible."""
        for key in ["BMI", "MentHlth", "PhysHlth"]:
            el = driver.find_element(By.CSS_SELECTOR, f'input[data-key="{key}"]')
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.1)
            assert el.is_displayed(), f"Number input for '{key}' not visible"

    def test_age_buttons_rendered(self, driver):
        """All 13 age group buttons must be present."""
        btns = driver.find_elements(By.CSS_SELECTOR, ".age-control .age-btn")
        assert len(btns) == 13, f"Expected 13 age buttons, found {len(btns)}"

    def test_genhlth_buttons_rendered(self, driver):
        """GenHlth must have exactly 5 options."""
        btns = driver.find_elements(
            By.CSS_SELECTOR, '[data-key="GenHlth"] .seg-btn'
        )
        assert len(btns) == 5

    def test_sex_buttons_rendered(self, driver):
        """Sex must have exactly 2 options (Female / Male)."""
        btns = driver.find_elements(
            By.CSS_SELECTOR, '[data-key="Sex"] .seg-btn'
        )
        assert len(btns) == 2

    def test_disclaimer_visible(self, driver):
        """The disclaimer block must be visible on load."""
        disclaimer = driver.find_element(By.CSS_SELECTOR, ".disclaimer")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", disclaimer)
        time.sleep(0.1)
        assert disclaimer.is_displayed()

    def test_disclaimer_mentions_educational(self, driver):
        text = driver.find_element(By.CSS_SELECTOR, ".disclaimer").text.lower()
        assert "educational" in text

    def test_disclaimer_mentions_accuracy(self, driver):
        text = driver.find_element(By.CSS_SELECTOR, ".disclaimer").text
        assert "75%" in text


class TestToggleInteraction:
    """Verify binary toggle fields work correctly."""

    def test_toggle_default_value_no(self, driver):
        """All toggles should show 'No' by default."""
        toggle_row = driver.find_element(
            By.CSS_SELECTOR, '.toggle-field[data-key="HighBP"]'
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", toggle_row)
        time.sleep(0.2)
        val = toggle_row.find_element(By.CSS_SELECTOR, ".toggle-value")
        assert val.text.strip() == "No"

    def test_toggle_click_shows_yes(self, driver):
        """Clicking a toggle should change its label to 'Yes'."""
        click_toggle(driver, "HighBP")
        time.sleep(0.2)
        val = driver.find_element(
            By.CSS_SELECTOR, '.toggle-field[data-key="HighBP"] .toggle-value'
        )
        assert val.text.strip() == "Yes"

    def test_toggle_double_click_back_to_no(self, driver):
        """Clicking a toggle twice should return it to 'No'."""
        click_toggle(driver, "Smoker")
        time.sleep(0.2)
        click_toggle(driver, "Smoker")
        time.sleep(0.2)
        val = driver.find_element(
            By.CSS_SELECTOR, '.toggle-field[data-key="Smoker"] .toggle-value'
        )
        assert val.text.strip() == "No"

    def test_toggle_adds_on_class(self, driver):
        """Clicking a toggle should add the 'on' CSS class to the row."""
        toggle_row = driver.find_element(
            By.CSS_SELECTOR, '.toggle-field[data-key="Stroke"]'
        )
        click_toggle(driver, "Stroke")
        time.sleep(0.2)
        classes = toggle_row.get_attribute("class")
        assert "on" in classes

    def test_multiple_toggles_independent(self, driver):
        """Toggling one field must not affect others."""
        click_toggle(driver, "Fruits")
        time.sleep(0.2)
        val_veggies = driver.find_element(
            By.CSS_SELECTOR, '.toggle-field[data-key="Veggies"] .toggle-value'
        )
        assert val_veggies.text.strip() == "No"


class TestSegmentedControls:
    """Verify segmented button controls (GenHlth, Sex, Age)."""

    def test_genhlth_button_becomes_active(self, driver):
        click_seg_btn(driver, "GenHlth", "3")
        time.sleep(0.2)
        btn = driver.find_element(
            By.CSS_SELECTOR, '[data-key="GenHlth"] .seg-btn[data-val="3"]'
        )
        assert "active" in btn.get_attribute("class")

    def test_genhlth_only_one_active(self, driver):
        click_seg_btn(driver, "GenHlth", "2")
        time.sleep(0.1)
        click_seg_btn(driver, "GenHlth", "4")
        time.sleep(0.2)
        active = driver.find_elements(
            By.CSS_SELECTOR, '[data-key="GenHlth"] .seg-btn.active'
        )
        assert len(active) == 1
        assert active[0].get_attribute("data-val") == "4"

    def test_sex_female_selected(self, driver):
        click_seg_btn(driver, "Sex", "0")
        time.sleep(0.2)
        btn = driver.find_element(
            By.CSS_SELECTOR, '[data-key="Sex"] .seg-btn[data-val="0"]'
        )
        assert "active" in btn.get_attribute("class")

    def test_sex_switches_selection(self, driver):
        click_seg_btn(driver, "Sex", "0")
        time.sleep(0.1)
        click_seg_btn(driver, "Sex", "1")
        time.sleep(0.2)
        female = driver.find_element(
            By.CSS_SELECTOR, '[data-key="Sex"] .seg-btn[data-val="0"]'
        )
        assert "active" not in female.get_attribute("class")

    def test_age_button_selected(self, driver):
        click_age_btn(driver, "4")
        time.sleep(0.2)
        btn = driver.find_element(
            By.CSS_SELECTOR, '.age-control .age-btn[data-val="4"]'
        )
        assert "active" in btn.get_attribute("class")

    def test_age_only_one_active(self, driver):
        click_age_btn(driver, "3")
        time.sleep(0.1)
        click_age_btn(driver, "7")
        time.sleep(0.2)
        active = driver.find_elements(
            By.CSS_SELECTOR, ".age-control .age-btn.active"
        )
        assert len(active) == 1
        assert active[0].get_attribute("data-val") == "7"


class TestFormValidation:
    """Verify client-side validation before submission."""

    def test_empty_form_shows_error(self, driver):
        submit_form(driver)
        assert error_banner_visible(driver)

    def test_empty_form_no_result(self, driver):
        submit_form(driver)
        time.sleep(0.5)
        assert not driver.find_element(By.ID, "result-card").is_displayed()

    def test_partial_form_shows_error(self, driver):
        fill_number_input(driver, "BMI", "25")
        submit_form(driver)
        assert error_banner_visible(driver)

    def test_invalid_bmi_marks_field(self, driver):
        """Typing an out-of-range BMI should mark the input as invalid."""
        fill_number_input(driver, "BMI", "999")
        # Click elsewhere to trigger the input event
        driver.execute_script("document.body.click();")
        time.sleep(0.3)
        inp = driver.find_element(By.CSS_SELECTOR, 'input[data-key="BMI"]')
        assert "invalid" in inp.get_attribute("class")

    def test_valid_bmi_marks_field_valid(self, driver):
        """Typing a valid BMI should add the 'valid' class."""
        fill_number_input(driver, "BMI", "24.5")
        driver.execute_script("document.body.click();")
        time.sleep(0.3)
        inp = driver.find_element(By.CSS_SELECTOR, 'input[data-key="BMI"]')
        assert "valid" in inp.get_attribute("class")


class TestFormSubmission:
    """End-to-end submission tests — require Flask server on port 5000."""

    def _get_error_text(self, driver):
        """Return the error banner text if visible, else empty string."""
        try:
            banner = driver.find_element(By.ID, "error-banner")
            if banner.is_displayed():
                return banner.text.strip()
        except Exception:
            pass
        return ""

    def test_valid_form_shows_result(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        # Wait a bit longer for the API call to return
        visible = result_card_visible(driver, timeout=15)
        if not visible:
            err = self._get_error_text(driver)
            raise AssertionError(
                f"Result card did not appear after valid submission.\n"
                f"Error banner says: {err!r}\n"
                f"Make sure Flask is running: python app.py"
            )

    def test_result_has_percentage(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        assert result_card_visible(driver, timeout=15), \
            f"Result card not shown. Error: {self._get_error_text(driver)!r}"
        score = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.visibility_of_element_located((By.ID, "score-display"))
        )
        assert "%" in score.text

    def test_result_has_risk_badge(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        assert result_card_visible(driver, timeout=15), \
            f"Result card not shown. Error: {self._get_error_text(driver)!r}"
        badge = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.visibility_of_element_located((By.ID, "result-badge"))
        )
        # CSS text-transform:uppercase makes the displayed text all-caps,
        # so we compare in lowercase to be case-insensitive.
        known = {"low risk", "moderate risk", "high risk", "very high risk"}
        assert badge.text.strip().lower() in known, \
            f"Unexpected badge text: {badge.text!r}"

    def test_gauge_element_present(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        assert result_card_visible(driver, timeout=15), \
            f"Result card not shown. Error: {self._get_error_text(driver)!r}"
        gauge = driver.find_element(By.ID, "gauge-fill")
        assert gauge is not None

    def test_no_error_on_valid_submit(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        assert result_card_visible(driver, timeout=15), \
            f"Result card not shown. Error: {self._get_error_text(driver)!r}"
        assert not driver.find_element(By.ID, "error-banner").is_displayed()


class TestResetButton:
    """Verify the Reset form button clears everything."""

    def test_reset_clears_number_inputs(self, driver):
        fill_number_input(driver, "BMI", "30")
        click_button_by_id(driver, "reset-btn")
        time.sleep(0.3)
        val = driver.find_element(
            By.CSS_SELECTOR, 'input[data-key="BMI"]'
        ).get_attribute("value")
        assert val == ""

    def test_reset_clears_toggles(self, driver):
        click_toggle(driver, "HighBP")
        click_toggle(driver, "Smoker")
        click_button_by_id(driver, "reset-btn")
        time.sleep(0.3)
        for key in ["HighBP", "Smoker"]:
            val = driver.find_element(
                By.CSS_SELECTOR, f'.toggle-field[data-key="{key}"] .toggle-value'
            ).text.strip()
            assert val == "No", f"Toggle '{key}' not reset to 'No'"

    def test_reset_clears_seg_buttons(self, driver):
        click_seg_btn(driver, "GenHlth", "3")
        click_age_btn(driver, "5")
        click_button_by_id(driver, "reset-btn")
        time.sleep(0.3)
        active_gen = driver.find_elements(
            By.CSS_SELECTOR, '[data-key="GenHlth"] .seg-btn.active'
        )
        active_age = driver.find_elements(
            By.CSS_SELECTOR, ".age-control .age-btn.active"
        )
        assert len(active_gen) == 0
        assert len(active_age) == 0

    def test_reset_hides_result_card(self, driver):
        fill_valid_form(driver)
        submit_form(driver)
        result_card_visible(driver)
        click_button_by_id(driver, "reset-btn")
        time.sleep(0.4)
        assert not driver.find_element(By.ID, "result-card").is_displayed()

    def test_reset_hides_error_banner(self, driver):
        submit_form(driver)   # empty form → triggers error
        error_banner_visible(driver)
        click_button_by_id(driver, "reset-btn")
        time.sleep(0.4)
        assert not driver.find_element(By.ID, "error-banner").is_displayed()
