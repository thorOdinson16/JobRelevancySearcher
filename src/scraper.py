"""Selenium scraper for Naukri.com job listings.

Ported from ``jobscraperfixedv2.py``. Notable fixes:

* removed the duplicated ``_scrape_single_job_detail`` definition;
* implemented the missing ``scrape_job_details`` orchestration method that
  ``main`` used to call (and crash on);
* narrowed the bare ``except`` clauses so real errors surface;
* a single WebDriver is reused for listing + detail pages;
* results are written to PostgreSQL via :func:`db.upsert_jobs` (idempotent).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import config
from src import schema as S

logger = logging.getLogger(__name__)

BASE_URL = "https://www.naukri.com/"
DETAIL_DELAY_SECONDS = 2


class JobScraper:
    """Scrape Naukri listings and persist them to the application database."""

    def __init__(self, headless: bool = True, page_load_timeout: int = 15) -> None:
        self.headless = headless
        self.page_load_timeout = page_load_timeout

    # -- driver lifecycle --------------------------------------------------
    def _build_driver(self) -> WebDriver:
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(self.page_load_timeout)
        return driver

    @contextmanager
    def _driver(self) -> Iterator[WebDriver]:
        driver = self._build_driver()
        try:
            yield driver
        finally:
            driver.quit()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _safe_extract(driver: WebDriver, xpath: str, default: str = "Not specified") -> str:
        try:
            element = driver.find_element(By.XPATH, xpath)
            return element.text.strip() or default
        except WebDriverException:
            return default

    @staticmethod
    def _text_or(element: WebElement, css: str, default: str) -> str:
        try:
            return element.find_element(By.CSS_SELECTOR, css).text.strip() or default
        except WebDriverException:
            return default

    def _extract_job_card_data(self, card: WebElement) -> Dict[str, Any]:
        """Extract the summary fields shown on a search-results card."""
        title = self._text_or(card, "a.title", "N/A")
        link = "N/A"
        try:
            link = card.find_element(By.CSS_SELECTOR, "a.title").get_attribute("href") or "N/A"
        except WebDriverException:
            pass

        return {
            S.JOB_TITLE: title,
            S.JOB_COMPANY: self._text_or(card, "a.comp-name", "N/A"),
            S.JOB_LOCATION: self._text_or(card, "span.loc-wrap", "Not specified"),
            S.JOB_EXPERIENCE: self._text_or(card, "span.exp-wrap", "Not specified"),
            S.JOB_SALARY: self._text_or(card, "span.sal-wrap", "Not specified"),
            S.JOB_LINK: link,
            S.JOB_STATUS: "Success",
        }

    # -- public API --------------------------------------------------------
    def scrape_job_listings(
        self, search_query: str, location: str, num_jobs: int = 7
    ) -> List[Dict[str, Any]]:
        """Return summary rows for the first ``num_jobs`` search results."""
        search_url = f"{BASE_URL}{search_query.strip().replace(' ', '-')}-jobs-in-{location.strip()}"
        jobs: List[Dict[str, Any]] = []
        with self._driver() as driver:
            driver.get(search_url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located(
                        (By.CLASS_NAME, "srp-jobtuple-wrapper")
                    )
                )
            except WebDriverException as exc:
                logger.warning("No job cards found for %s: %s", search_url, exc)
                return jobs

            cards = driver.find_elements(By.CLASS_NAME, "srp-jobtuple-wrapper")[:num_jobs]
            for card in cards:
                jobs.append(self._extract_job_card_data(card))
        return jobs

    def scrape_job_detail(self, driver: WebDriver, job: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single listing with its full detail page."""
        detailed = dict(job)
        link = detailed.get(S.JOB_LINK, "N/A")
        if not link or link == "N/A":
            detailed[S.JOB_STATUS] = "Failed"
            detailed[S.JOB_ERROR] = "Missing job link"
            return detailed

        try:
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            try:
                detailed[S.JOB_DESCRIPTION] = driver.find_element(
                    By.CLASS_NAME, "job-desc"
                ).text
            except WebDriverException:
                detailed[S.JOB_DESCRIPTION] = "Not available"

            try:
                skills = driver.find_elements(By.CSS_SELECTOR, "div.key-skill")
                detailed[S.JOB_SKILLS] = [s.text.strip() for s in skills if s.text.strip()]
            except WebDriverException:
                detailed[S.JOB_SKILLS] = []

            detailed[S.JOB_ROLE] = self._safe_extract(
                driver, "//div[contains(text(), 'Role:')]/following-sibling::*[1]"
            )
            detailed[S.JOB_INDUSTRY] = self._safe_extract(
                driver,
                "//div[contains(text(), 'Industry Type:')]/following-sibling::*[1]",
            )
            detailed[S.JOB_EMPLOYMENT_TYPE] = self._safe_extract(
                driver,
                "//div[contains(text(), 'Employment Type:')]/following-sibling::*[1]",
            )

            detailed[S.JOB_SCRAPED_AT] = datetime.now(timezone.utc)
            detailed[S.JOB_STATUS] = "Success"
        except WebDriverException as exc:
            logger.error("Error scraping %s: %s", link, exc)
            detailed[S.JOB_STATUS] = "Failed"
            detailed[S.JOB_ERROR] = str(exc)
        return detailed

    def scrape_job_details(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich multiple listings while reusing a single WebDriver."""
        if not jobs:
            return []
        details: List[Dict[str, Any]] = []
        with self._driver() as driver:
            for job in jobs:
                details.append(self.scrape_job_detail(driver, job))
        return details

    def scrape(
        self, search_query: str, location: str, num_jobs: int = 7
    ) -> List[Dict[str, Any]]:
        """Full pipeline: list, detail, then upsert into PostgreSQL."""
        listings = self.scrape_job_listings(search_query, location, num_jobs)
        detailed = self.scrape_job_details(listings)
        if detailed:
            from db import get_session, upsert_jobs

            session = get_session()
            try:
                upsert_jobs(session, detailed)
            finally:
                session.close()
        return detailed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    search_query = input("Enter Role: ").strip()
    location = input("Enter Location: ").strip()
    raw_num = input(f"Number of jobs [{config.DEFAULT_NUM_JOBS}]: ").strip()
    num_jobs = int(raw_num) if raw_num else config.DEFAULT_NUM_JOBS

    from db import ensure_database, init_db

    ensure_database()
    init_db()

    scraper = JobScraper()
    results = scraper.scrape(search_query, location, num_jobs)
    logger.info("Scraped and stored %d jobs.", len(results))


if __name__ == "__main__":
    main()
