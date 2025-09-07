from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
from bson import json_util
import json
import time
from datetime import datetime

class JobScraper:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.mongodb_uri = "mongodb://localhost:27017/"
        
    def scrape_job_listings(self, search_query, location, num_jobs=7):
        driver = webdriver.Chrome(options=self.options)
        base_url = "https://www.naukri.com/"
        search_url = f"{base_url}{search_query}-jobs-in-{location}"
        driver.get(search_url)
        
        jobs = []
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "srp-jobtuple-wrapper"))
            )
            
            job_cards = driver.find_elements(By.CLASS_NAME, "srp-jobtuple-wrapper")
            job_cards = job_cards[:num_jobs]
            
            for card in job_cards:
                job_data = self._extract_job_card_data(card)
                jobs.append(job_data)

        except Exception as e:
            print(f"An error occurred while scraping listings: {e}")
        finally:
            driver.quit()

        return jobs

    def _extract_job_card_data(self, card):
        """Helper method to extract data from job card"""
        try:
            title_element = card.find_element(By.CSS_SELECTOR, "a.title")
            title = title_element.text
            link = title_element.get_attribute("href")
        except:
            title = "N/A"
            link = "N/A"

        try:
            company = card.find_element(By.CSS_SELECTOR, "a.comp-name").text
        except:
            company = "N/A"

        try:
            location = card.find_element(By.CSS_SELECTOR, "span.loc-wrap").text
        except:
            location = "Not specified"

        try:
            experience = card.find_element(By.CSS_SELECTOR, "span.exp-wrap").text
        except:
            experience = "Not specified"

        try:
            salary = card.find_element(By.CSS_SELECTOR, "span.sal-wrap").text
        except:
            salary = "Not specified"

        return {
            "Title": title,
            "Company": company,
            "Location": location,
            "Experience": experience,
            "Salary": salary,
            "Link": link
        }

    def save_to_mongodb(self, data, db_name="JobScrapingDB", collection_name="NaukriJobs"):
        """Save job listings to MongoDB"""
        client = MongoClient(self.mongodb_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        if data:
            collection.insert_many(data)
            print(f"Data successfully inserted into '{db_name}.{collection_name}' collection.")
        else:
            print("No data to insert.")
        
        return collection

    def _scrape_single_job_detail(self, driver, job_basic):
        """Enhanced method to scrape detailed information from a single job page"""
        detailed_job = dict(job_basic)  # Create a copy of basic job info
        
        try:
            driver.get(detailed_job['Link'])
            time.sleep(2)  # Allow page to load
            
            # Description
            try:
                detailed_job['Description'] = driver.find_element(By.CLASS_NAME, "job-desc").text
            except:
                detailed_job['Description'] = "Not available"

            # Skills
            try:
                skills = driver.find_elements(By.CSS_SELECTOR, "div.key-skill")
                detailed_job['Required_Skills'] = [skill.text for skill in skills]
            except:
                detailed_job['Required_Skills'] = []

            # Role
            try:
                detailed_job['Role'] = driver.find_element(By.CSS_SELECTOR, "div.role-desc").text
            except:
                detailed_job['Role'] = "Not specified"

            # Industry Type
            try:
                industry = driver.find_element(By.CSS_SELECTOR, "div.job-details-wrapper").text
                detailed_job['Industry'] = industry
            except:
                detailed_job['Industry'] = "Not specified"
                
            # Additional Details
            try:
                details_list = driver.find_elements(By.CSS_SELECTOR, "div.details-info")
                for detail in details_list:
                    label = detail.find_element(By.CSS_SELECTOR, "label").text.strip()
                    value = detail.find_element(By.CSS_SELECTOR, "span").text.strip()
                    detailed_job[label] = value
            except:
                pass

            detailed_job['Scraped_Date'] = datetime.now().isoformat()
            detailed_job['Status'] = 'Success'
            
            return detailed_job
            
        except Exception as e:
            print(f"Error scraping details for job at {detailed_job['Link']}: {e}")
            detailed_job['Status'] = 'Failed'
            detailed_job['Error'] = str(e)
            return detailed_job

    def _scrape_single_job_detail(self, driver, job_basic):
        """Enhanced method to scrape detailed information from a single job page"""
        detailed_job = dict(job_basic)
    
        try:
            driver.get(detailed_job['Link'])
            time.sleep(3)  # Ensure page loads completely
        
            # Basic Job Information
            try:
            # Role Information
                detailed_job.update({
                'Role': self._safe_extract(driver, "//div[contains(text(), 'Role:')]/following-sibling::*[1]"),
                'Industry_Type': self._safe_extract(driver, "//div[contains(text(), 'Industry Type:')]/following-sibling::*[1]"),
                'Department': self._safe_extract(driver, "//div[contains(text(), 'Department:')]/following-sibling::*[1]"),
                'Employment_Type': self._safe_extract(driver, "//div[contains(text(), 'Employment Type:')]/following-sibling::*[1]"),
                'Role_Category': self._safe_extract(driver, "//div[contains(text(), 'Role Category:')]/following-sibling::*[1]"),
            })
            except Exception as e:
                print(f"Error extracting basic information: {e}")

            # Organizational Context
            detailed_job['Organizational_Context'] = self._safe_extract(
                driver, 
                "//div[contains(text(), 'Organizational Context:')]/following-sibling::div[1]"
            )

            # Responsibilities
            responsibilities = []
            try:
                resp_elements = driver.find_elements(By.XPATH, 
                "//div[contains(text(), 'Responsibilities:')]/following-sibling::div//text()|" +
                "//h2[contains(text(), 'Responsibilities')]/following-sibling::ul/li")
                responsibilities = [elem.text.strip() for elem in resp_elements if elem.text.strip()]
            except Exception as e:
                print(f"Error extracting responsibilities: {e}")
            detailed_job['Responsibilities'] = responsibilities

            # Qualifications and Skills
            qualifications = []
            try:
                qual_elements = driver.find_elements(By.XPATH,
                "//div[contains(text(), 'Qualifications and Skills:')]/following-sibling::div//text()|" +
                "//h2[contains(text(), 'Qualifications')]/following-sibling::ul/li")
                qualifications = [elem.text.strip() for elem in qual_elements if elem.text.strip()]
            except Exception as e:
                print(f"Error extracting qualifications: {e}")
            detailed_job['Qualifications'] = qualifications

            # Additional Skills/Experience (Plus Points)
            plus_points = []
            try:
                plus_elements = driver.find_elements(By.XPATH, 
                    "//*[contains(text(), 'is a plus')]/parent::*")
                plus_points = [elem.text.strip() for elem in plus_elements if elem.text.strip()]
            except Exception as e:
                print(f"Error extracting plus points: {e}")
            detailed_job['Plus_Points'] = plus_points

            # Benefits
            benefits = []
            try:
                benefit_elements = driver.find_elements(By.XPATH,
                "//div[contains(text(), 'Benefits include:')]/following-sibling::div//text()|" +
                "//h2[contains(text(), 'Benefits')]/following-sibling::ul/li")
                benefits = [elem.text.strip() for elem in benefit_elements if elem.text.strip()]
            except Exception as e:
                print(f"Error extracting benefits: {e}")
            detailed_job['Benefits'] = benefits

            # Company Policies and Statements
            policies = []
            try:
                policy_elements = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'American Express is an equal opportunity employer')]|" +
                "//*[contains(text(), 'Offer of employment')]")
                policies = [elem.text.strip() for elem in policy_elements if elem.text.strip()]
            except Exception as e:
                print(f"Error extracting policies: {e}")
            detailed_job['Company_Policies'] = policies

            detailed_job['Scraped_Date'] = datetime.now().isoformat()
            detailed_job['Status'] = 'Success'
        
            return detailed_job
        
        except Exception as e:
            print(f"Error scraping details for job at {detailed_job['Link']}: {e}")
            detailed_job['Status'] = 'Failed'
            detailed_job['Error'] = str(e)
        return detailed_job

    def save_to_json(self, data, filename=None):
        """Save the detailed job data to a JSON file"""
        if not filename:
            filename = f"naukri_detailed_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=json_util.default, indent=4, ensure_ascii=False)
            print(f"Data successfully saved to {filename}")
            
            # Print the first job as a sample
            if data:
                print("\nSample Job Details (First Entry):")
                print(json.dumps(data[0], default=json_util.default, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"Error saving to JSON: {e}")

def main():
    scraper = JobScraper()
    
    search_query = input("Enter Role: ")
    location = input("Enter Location: ")
    num_jobs = int(input())  # Fixed to 7 jobs
    
    # Step 1: Scrape basic listings and save to MongoDB
    jobs = scraper.scrape_job_listings(search_query, location, num_jobs)
    if jobs:
        scraper.save_to_mongodb(jobs)
    
    # Step 2: Scrape and save detailed information
    detailed_jobs = scraper.scrape_job_details()
    if detailed_jobs:
        scraper.save_to_json(detailed_jobs)

if __name__ == "__main__":
    main()