import PyPDF2
import re
import json
from typing import Dict, Optional, List
import logging
import os
from datetime import datetime
import unicodedata
import google.generativeai as genai
import time

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")


class PaperSectionExtractor:
    """Extract sections from scientific papers in PDF format and export to JSON."""

    def __init__(self):
        self.section_patterns = {
            'title': r'^.*?(?=\s*(?:\d\s*Department|Abstract:|ABSTRACT))',
            'authors': r'(?<=\n).+?(?=\s*(?:\d\s*Department|Correspondence|Abstract:|ABSTRACT))',
            'affiliations': r'(?:\d+\s*Department|Institute|University|Laboratory).+?(?=(?:Abstract:|ABSTRACT|\*\s*Correspondence))',
            'abstract': r'(?:Abstract:|ABSTRACT)[\s:]*(.*?)(?=(?:\s*Keywords:|KEYWORDS|Introduction|1\.|\n{2,}))',
            'keywords': r'(?:Keywords:|KEYWORDS)[\s:]*(.*?)(?=(?:\s*(?:1\.|Introduction|\n{2,})))',

            'introduction': r'(?:1\.|I\.|Introduction|INTRODUCTION)(?:\s|-|:|\.)*([^2IV]+?)(?=(?:\s*(?:2\.|II\.|Methods|Materials|Results)))',
            'methods': r'(?:2\.|II\.|Methods|Materials\sand\sMethods|METHODS|Experimental)(?:\s|-|:|\.)*([^3IV]+?)(?=(?:\s*(?:3\.|III\.|Results)))',
            'results': r'(?:3\.|III\.|Results|RESULTS)(?:\s|-|:|\.)*([^4IV]+?)(?=(?:\s*(?:4\.|IV\.|Discussion)))',
            'results_and_discussion': r'(?:3\.|III\.|Results\sand\sDiscussion|RESULTS\sAND\sDISCUSSION)(?:\s|-|:|\.)*([^4IV]+?)(?=(?:\s*(?:4\.|IV\.|Conclusion)))',
            'discussion': r'(?:4\.|IV\.|Discussion|DISCUSSION)(?:\s|-|:|\.)*([^5V]+?)(?=(?:\s*(?:5\.|V\.|Conclusion|References)))',
            'conclusion': r'(?:5\.|V\.|Conclusion|CONCLUSION|Concluding\sRemarks)(?:\s|-|:|\.)*([^6VI]+?)(?=(?:\s*(?:References|REFERENCES|Acknowledgments)))',

            'acknowledgments': r'(?:Acknowledgments?|ACKNOWLEDGMENTS?)(?:\s|-|:|\.)*([^R]+?)(?=(?:\s*(?:References|REFERENCES)))',
            'author_contributions': r'(?:Author\sContributions?|AUTHOR\sCONTRIBUTIONS?)(?:\s|-|:|\.)*([^R]+?)(?=(?:\s*(?:References|REFERENCES|Acknowledgments?)))',
            'competing_interests': r'(?:Competing\sInterests?|COMPETING\sINTERESTS?|Conflicts?\sof\sInterest)(?:\s|-|:|\.)*([^R]+?)(?=(?:\s*(?:References|REFERENCES)))',
            'data_availability': r'(?:Data\sAvailability|DATA\sAVAILABILITY)(?:\s|-|:|\.)*([^R]+?)(?=(?:\s*(?:References|REFERENCES)))',
            'supplementary_materials': r'(?:Supplementary\sMaterials?|SUPPLEMENTARY\sMATERIALS?)(?:\s|-|:|\.)*([^R]+?)(?=(?:\s*(?:References|REFERENCES)))',

            'references': r'(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)(?:\s|-|:|\.)*(.+?)(?=(?:\s*(?:Supplementary|Appendix|$)))',
        }

        # Citation patterns for different styles
        self.citation_patterns = [
            r'\[\d+(?:,\s*\d+)*\]',           # [1] or [1,2,3]
            r'\[\w+\s*(?:et\s*al\.)?,\s*\d{4}\]',  # [Smith et al., 2020]
            r'\(\w+\s*(?:et\s*al\.)?,\s*\d{4}\)',  # (Smith et al., 2020)
            r'\[\w+\s*&\s*\w+,\s*\d{4}\]',    # [Smith & Jones, 2020]
            r'\(\w+\s*&\s*\w+,\s*\d{4}\)',    # (Smith & Jones, 2020)
            # Vancouver style: 1 or 1,2,3
            r'(?<=\s)\d{1,2}(?:,\s*\d{1,2})*(?=[\s,\.])',
            r'(?<=\s)\[\w+\d{2}\w*\]',        # [Smi20a]
            r'(?<=\s)\(\w+\d{2}\w*\)',        # (Smi20a)
        ]

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def query_gemini(self, text: str, section_name: str) -> Dict:
        """Query Gemini to assess section content and perform additional analysis."""
        try:
            prompt = f"""
            You are a scientific paper section analyzer. You are given a section name and
            text extracted from that section of a PDF. Analyze the text and provide the
            following information in a JSON format:

            Section Name: {section_name}
            Extracted Text: {text}

            JSON Response Format:
            {{
                "is_correct_section": true/false,  // Indicates if the text belongs to the given section
                "confidence": float,  // Confidence score for the section prediction (0.0 to 1.0)
                "reasoning": "Explanation of why the text does or does not belong to the section",
                "key_phrases": ["list", "of", "important", "phrases"],
                "main_topics": ["main", "topics", "covered"],
                "summary": "A concise summary of the section."
            }}
            """
            response = model.generate_content(prompt)

            try:
                cleaned_response = re.sub(
                    r"^```json\n|\n```$", "", response.text, flags=re.MULTILINE)
                analysis = json.loads(cleaned_response.strip())
                return analysis
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding Gemini JSON: {
                                  e}. Raw response: {response.text}")
                return {"is_correct_section": False, "confidence": 0.0, "reasoning": "Invalid JSON from Gemini"}

        except Exception as e:
            self.logger.error(f"Error querying Gemini: {e}")
            return {"is_correct_section": False, "confidence": 0.0, "reasoning": f"Error: {e}"}

    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning for PDF extraction.

        Args:
            text: Raw text extracted from PDF

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Step 1: Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Step 2: Fix common PDF extraction issues

        # Replace various types of hyphens and dashes with standard ones
        text = re.sub(r'[\u2010-\u2015]', '-', text)

        # Fix broken words (words split across lines with hyphens)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Step 3: Clean up whitespace

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:?!)])', r'\1', text)
        text = re.sub(r'([({])\s+', r'\1', text)

        # Step 4: Fix common academic text issues

        # Fix reference citations [1] [2] -> [1,2]
        text = re.sub(r'\]\s*\[', ',', text)

        # Fix percentage signs
        text = re.sub(r'(\d+)\s+%', r'\1%', text)

        # Fix common unit spacing
        text = re.sub(r'(\d+)\s+(mg|ml|kg|cm|mm|µm|nm|°C|°F)', r'\1\2', text)

        # Step 5: Fix section headers

        # Ensure consistent spacing after section numbers
        text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)

        # Step 6: Fix special characters

        # Fix Greek letters
        greek_letters = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'mu': 'μ', 'sigma': 'σ', 'omega': 'ω'
        }
        for name, symbol in greek_letters.items():
            text = re.sub(rf'\b{name}\b', symbol, text, flags=re.IGNORECASE)

        # Step 7: Remove header/footer artifacts

        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove running headers/footers (common in academic papers)
        text = re.sub(r'\n.*(Page|Vol\.|Volume).*\n', '\n', text)

        # Step 8: Fix common formatting issues

        # Fix spacing around mathematical operators
        text = re.sub(r'(\d+)\s*([+\-×÷=])\s*(\d+)', r'\1 \2 \3', text)

        # Fix superscript/subscript numbers
        text = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^\2', text)
        text = re.sub(r'(\d+)\s*_\s*(\d+)', r'\1_\2', text)

        # Step 9: Final cleanup

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trim whitespace
        text = text.strip()

        return text

    def preprocess_pdf_text(self, text: str) -> str:
        """
        Preprocess PDF text before section extraction.

        Args:
            text: Raw text from PDF

        Returns:
            Preprocessed text
        """
        # Remove headers and footers that might interfere with section detection
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Fix section headers that might be split across lines
        section_headers = ['Abstract', 'Introduction', 'Methods',
                           'Results', 'Discussion', 'Conclusion', 'References']
        for header in section_headers:
            pattern = rf'([^.]\n+)({header}:?)'
            text = re.sub(pattern, r'\1\n\2', text)

        # Fix common PDF artifacts around section numbers
        text = re.sub(r'(\d+\s*\.\s*)(\n+)([A-Z])', r'\1\3', text)

        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction with better error handling."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                self.logger.info(f"Processing PDF with {num_pages} pages")

                # Extract text page by page with preprocessing
                text_parts = []
                for page_num in range(num_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        self.logger.debug(
                            f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {
                                            page_num + 1}: {str(e)}")
                        continue

                # Combine all pages and preprocess
                full_text = '\n'.join(text_parts)
                processed_text = self.preprocess_pdf_text(full_text)

                return processed_text

        except Exception as e:
            self.logger.error(f"Error reading PDF: {str(e)}")
            raise

    def find_citations(self, text: str) -> List[str]:
        """
        Find citations in text using multiple citation styles.

        Args:
            text: Text to search for citations

        Returns:
            List of found citations
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append(match.group())
        return list(set(citations))  # Remove duplicates

    def analyze_section(self, section_text: str, section_name: str) -> Dict:
        """
        Analyze the content of a section.

        Args:
            section_text: Text content of the section

        Returns:
            Dictionary containing analysis results
        """
        # Split into sentences (improved)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Find citations
        citations = self.find_citations(section_text)

        # Count words (excluding references and citations)
        # First, remove citations
        text_without_citations = section_text
        for citation in citations:
            text_without_citations = text_without_citations.replace(
                citation, '')

        # Count words
        words = re.findall(r'\b\w+\b', text_without_citations)

        # Find numerical values (improved)
        numerical_values = re.findall(
            r'(?:^|[^\w])((?:\d*\.)?\d+(?:[eE][+-]?\d+)?%?)(?:[^\w]|$)', section_text)

        standard_analysis = {  # Store the standard analysis in a dictionary
            'word_count': len(words),
            'sentence_count': len(sentences),
            'citation_count': len(citations),
            'numerical_values_count': len(numerical_values),
            'first_sentence': sentences[0] if sentences else "",
            'citations': citations,
            'numerical_values': numerical_values[:10]
        }

        try:
            gemini_analysis = self.query_gemini(section_text, section_name)
            standard_analysis['gemini_analysis'] = gemini_analysis

            time.sleep(4)
        except Exception as e:
            self.logger.error(f"Error in Gemini analysis: {e}")
            # Include error information in the analysis
            standard_analysis['gemini_analysis'] = {"error": str(e)}

        return standard_analysis

    def extract_and_analyze_pdf(self, pdf_path: str, output_json_path: Optional[str] = None) -> Dict:
        """
        Extract sections from PDF, analyze them, and optionally save to JSON.

        Args:
            pdf_path: Path to the PDF file
            output_json_path: Optional path to save JSON output

        Returns:
            Dictionary containing extracted sections and their analysis
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)

        # Initialize result dictionary
        result = {
            'metadata': {
                'filename': os.path.basename(pdf_path),
                'extraction_date': datetime.now().isoformat(),
                'pdf_path': pdf_path
            },
            'sections': {}
        }

        # Extract sections
        for section_name, pattern in self.section_patterns.items():
            try:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1) if section_name not in [
                        'title', 'authors', 'affiliations'] else match.group(0)
                    cleaned_content = self.clean_text(content)

                    # Store section content
                    result['sections'][section_name] = {
                        'content': cleaned_content
                    }

                    self.logger.info(f"Successfully extracted {
                                     section_name} section")
            except Exception as e:
                self.logger.error(f"Error processing {section_name}: {str(e)}")
                result['sections'][section_name] = {
                    'content': '',
                    'error': str(e)
                }

        # Analyze sections
        for section_name, section_data in result['sections'].items():
            try:
                content = section_data['content']
                if content:
                    analysis = self.analyze_section(content, section_name)
                    result['sections'][section_name]['analysis'] = analysis
                    self.logger.info(f"Successfully analyzed {
                                     section_name} section")
            except Exception as e:
                self.logger.error(f"Error analyzing {section_name}: {str(e)}")
                result['sections'][section_name]['analysis'] = {
                    "error": str(e)}

        # Save to JSON if output path is provided
        if output_json_path:
            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Successfully saved results to {
                                 output_json_path}")
            except Exception as e:
                self.logger.error(f"Error saving JSON: {str(e)}")
                raise

        return result


def main():
    """Example usage of the PaperSectionExtractor with JSON export"""
    try:
        # Initialize the extractor
        extractor = PaperSectionExtractor()

        # Specify input and output paths
        pdf_path = "example.pdf"  # Replace with your PDF path
        output_json = "example.json"  # Output JSON file
        # Extract sections and save to JSON
        result = extractor.extract_and_analyze_pdf(
            pdf_path, output_json)

        # Print summary with citation details
        print("\nExtraction Summary:")
        print(f"Input PDF: {pdf_path}")
        print(f"Output JSON: {output_json}")
        print("\nExtracted sections:")
        for section_name, section_data in result['sections'].items():
            content = section_data['content']
            analysis = section_data.get('analysis', {})
            print(f"\n=== {section_name.upper()} ===")
            print(f"Words: {analysis.get('word_count', 'N/A')}")
            print(f"Sentences: {analysis.get('sentence_count', 'N/A')}")
            print(f"Citations found: {len(analysis.get('citations', []))}")
            if analysis.get('citations'):
                print("Sample citations:")
                for citation in analysis['citations'][:5]:
                    print(f"  - {citation}")
            if "gemini_analysis" in analysis:
                gemini_result = analysis["gemini_analysis"]
                print(f"Gemini analysis for {section_name}:")
                print(f"  Is Correct Section: {
                      gemini_result.get('is_correct_section', False)}")
                print(f"  Confidence: {gemini_result.get('confidence', 0.0)}")
                print(f"  Reasoning: {gemini_result.get('reasoning', 'N/A')}")
                print(f"  Key Phrases: {', '.join(
                    gemini_result.get('key_phrases', []))}")
                print(f"  Main Topics: {', '.join(
                    gemini_result.get('main_topics', []))}")
                print(f"  Summary: {gemini_result.get('summary', 'N/A')}")
            else:
                print("Gemini analysis not available for this section.")
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error in main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
