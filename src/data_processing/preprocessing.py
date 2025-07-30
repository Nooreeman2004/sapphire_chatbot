"""
Data preprocessing module for Sapphire RAG pipeline.
Handles JSON cleaning and flattening if needed before chunking.
"""

import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    A class for preprocessing raw JSON data.
    Currently, the chunking logic handles most of the flattening and cleaning.
    This class can be extended for more complex preprocessing steps if required.
    """

    def __init__(self):
        logger.info("Preprocessor initialized.")

    def flatten_json(self, data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """
        Flattens a nested JSON dictionary. (Not actively used by current chunking, but available)
        """
        items = {}
        for k, v in data.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_json(v, new_key))
            else:
                items[new_key] = v
        return items

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning (e.g., removing extra whitespace, special characters).
        """
        if not isinstance(text, str):
            return ""
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())
        return text

    def process_product_data(self, product_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cleaning to product descriptions and other text fields.
        """
        processed_products = []
        for product in product_list:
            cleaned_product = product.copy()
            for key, value in product.items():
                if isinstance(value, str):
                    cleaned_product[key] = self.clean_text(value)
                elif isinstance(value, dict):
                    # Recursively clean nested dictionaries if necessary
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            cleaned_product[key][sub_key] = self.clean_text(sub_value)
            processed_products.append(cleaned_product)
        return processed_products

    def process_faq_data(self, faq_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cleaning to FAQ questions and answers.
        """
        processed_faqs = []
        for faq in faq_list:
            cleaned_faq = faq.copy()
            cleaned_faq["question"] = self.clean_text(faq.get("question", ""))
            cleaned_faq["answer"] = self.clean_text(faq.get("answer", ""))
            processed_faqs.append(cleaned_faq)
        return processed_faqs

    def process_policy_data(self, policy_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cleaning to policy content and details.
        """
        processed_policies = []
        for policy in policy_list:
            cleaned_policy = policy.copy()
            cleaned_policy["content"] = self.clean_text(policy.get("content", ""))
            if "details" in cleaned_policy and isinstance(cleaned_policy["details"], list):
                cleaned_policy["details"] = [self.clean_text(d) for d in cleaned_policy["details"]]
            processed_policies.append(cleaned_policy)
        return processed_policies

    def process_web_content_data(self, web_content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies cleaning to web page text content.
        """
        processed_web_content = []
        for page in web_content_list:
            cleaned_page = page.copy()
            cleaned_page["text_content"] = self.clean_text(page.get("text_content", ""))
            processed_web_content.append(cleaned_page)
        return processed_web_content

if __name__ == "__main__":
    # Example Usage (assuming raw data files are available)
    preprocessor = Preprocessor()

    # Example for products (assuming you load data first)
    # with open("path/to/sapphire_products.json", "r", encoding="utf-8") as f:
    #     product_data = json.load(f)
    # processed_products = preprocessor.process_product_data(product_data.get("products", []))
    # print(f"Processed {len(processed_products)} products.")

    # Example for FAQs
    # with open("path/to/faqs.json", "r", encoding="utf-8") as f:
    #     faq_data = json.load(f)
    # processed_faqs = preprocessor.process_faq_data(faq_data)
    # print(f"Processed {len(processed_faqs)} FAQs.")

    logger.info("Preprocessing module created. It can be integrated with DataProcessor if needed.")


