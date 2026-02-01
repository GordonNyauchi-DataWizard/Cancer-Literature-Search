#!/usr/bin/env python3
"""
Improved Cancer Research Paper Downloader

Downloads papers from multiple sources including bioRxiv, medRxiv, and arXiv.

Usage:
    python download_papers_v2.py --count 150
"""

import os
import time
import requests
import argparse
from pathlib import Path
import json


class MultiSourceDownloader:
    """Download papers from multiple open-access sources."""
    
    def __init__(self, output_dir="papers", delay=2.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.downloaded = 0
        
    def download_biorxiv_papers(self, count=150):
        """
        Download papers from bioRxiv (preprint server).
        bioRxiv has lots of cancer biology papers with direct PDF access.
        """
        print("\n🔬 Searching bioRxiv for cancer papers...")
        
        # bioRxiv API endpoint
        base_url = "https://api.biorxiv.org/details/biorxiv"
        
        # Search parameters
        cursor = 0
        papers_found = []
        
        # Collect paper metadata
        while len(papers_found) < count * 2:  # Get extra in case some fail
            try:
                # Fetch papers from a date range
                url = f"{base_url}/2020-01-01/2024-12-31/{cursor}"
                print(f"   Fetching batch starting at {cursor}...")
                
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    break
                
                data = response.json()
                collection = data.get('collection', [])
                
                if not collection:
                    break
                
                # Filter for cancer-related papers
                for paper in collection:
                    title = paper.get('title', '').lower()
                    abstract = paper.get('abstract', '').lower()
                    
                    # Check if cancer-related
                    cancer_keywords = ['cancer', 'tumor', 'oncology', 'carcinoma', 
                                      'melanoma', 'leukemia', 'lymphoma', 'immunotherapy']
                    
                    if any(keyword in title or keyword in abstract for keyword in cancer_keywords):
                        papers_found.append(paper)
                        
                        if len(papers_found) >= count * 2:
                            break
                
                cursor += len(collection)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"   Error fetching batch: {e}")
                break
        
        print(f"✓ Found {len(papers_found)} cancer-related papers on bioRxiv")
        
        # Download PDFs
        print(f"\n📥 Downloading PDFs...")
        
        for i, paper in enumerate(papers_found[:count]):
            if self.downloaded >= count:
                break
            
            doi = paper.get('doi', '')
            title = paper.get('title', f'paper_{i}')
            
            # Clean title for filename
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            clean_title = clean_title[:80]
            filename = f"{self.downloaded+1:03d}_{clean_title}.pdf"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                print(f"[{i+1}/{count}] ⏭️  Skipping (exists): {title[:60]}")
                self.downloaded += 1
                continue
            
            # Construct PDF URL
            pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
            
            print(f"[{i+1}/{count}] 📥 Downloading: {title[:60]}...", end=" ")
            
            try:
                response = requests.get(pdf_url, timeout=30, allow_redirects=True)
                
                if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print("✓")
                    self.downloaded += 1
                else:
                    print("✗")
                
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"✗ ({e})")
    
    def download_medrxiv_papers(self, count=150):
        """
        Download papers from medRxiv (medical preprints).
        """
        print("\n🏥 Searching medRxiv for cancer papers...")
        
        base_url = "https://api.biorxiv.org/details/medrxiv"
        cursor = 0
        papers_found = []
        
        while len(papers_found) < count and self.downloaded < count:
            try:
                url = f"{base_url}/2020-01-01/2024-12-31/{cursor}"
                print(f"   Fetching batch starting at {cursor}...")
                
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    break
                
                data = response.json()
                collection = data.get('collection', [])
                
                if not collection:
                    break
                
                # Filter for cancer papers
                for paper in collection:
                    title = paper.get('title', '').lower()
                    abstract = paper.get('abstract', '').lower()
                    
                    cancer_keywords = ['cancer', 'tumor', 'oncology', 'carcinoma', 
                                      'melanoma', 'leukemia', 'immunotherapy']
                    
                    if any(keyword in title or keyword in abstract for keyword in cancer_keywords):
                        papers_found.append(paper)
                
                cursor += len(collection)
                time.sleep(1)
                
            except Exception as e:
                print(f"   Error: {e}")
                break
        
        print(f"✓ Found {len(papers_found)} papers on medRxiv")
        
        # Download
        for i, paper in enumerate(papers_found):
            if self.downloaded >= count:
                break
            
            doi = paper.get('doi', '')
            title = paper.get('title', f'paper_{i}')
            
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            clean_title = clean_title[:80]
            filename = f"{self.downloaded+1:03d}_{clean_title}.pdf"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                self.downloaded += 1
                continue
            
            pdf_url = f"https://www.medrxiv.org/content/{doi}v1.full.pdf"
            
            print(f"[{self.downloaded+1}/{count}] 📥 {title[:60]}...", end=" ")
            
            try:
                response = requests.get(pdf_url, timeout=30)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print("✓")
                    self.downloaded += 1
                else:
                    print("✗")
                time.sleep(self.delay)
            except:
                print("✗")
    
    def download_papers(self, count=150):
        """Main download function."""
        print("\n" + "="*80)
        print("CANCER RESEARCH PAPER DOWNLOADER V2")
        print("="*80)
        print(f"\nTarget: {count} papers")
        print(f"Output: {self.output_dir.absolute()}\n")
        
        # Try bioRxiv first
        remaining = count - self.downloaded
        if remaining > 0:
            self.download_biorxiv_papers(remaining)
        
        # Try medRxiv if needed
        remaining = count - self.downloaded
        if remaining > 0:
            self.download_medrxiv_papers(remaining)
        
        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE")
        print("="*80)
        print(f"✓ Successfully downloaded: {self.downloaded} papers")
        print(f"📁 Location: {self.output_dir.absolute()}")
        
        if self.downloaded < count:
            print(f"\n⚠️  Only downloaded {self.downloaded}/{count} papers.")
            print("   This is because:")
            print("   - Some papers don't have PDFs available")
            print("   - Rate limiting to avoid being blocked")
            print("\n   To get more papers:")
            print("   1. Run the script again (it will skip existing files)")
            print("   2. Try manual download from sources below")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download cancer research papers")
    
    parser.add_argument('--count', type=int, default=150,
                       help='Number of papers to download (default: 150)')
    parser.add_argument('--output', type=str, default='papers',
                       help='Output directory (default: papers)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between downloads (default: 2.0)')
    
    args = parser.parse_args()
    
    downloader = MultiSourceDownloader(args.output, args.delay)
    downloader.download_papers(args.count)
    
    # Print manual alternatives
    print("\n📚 MANUAL DOWNLOAD SOURCES:")
    print("="*80)
    print("\nIf you need more papers, try these sources:\n")
    print("1. bioRxiv - https://www.biorxiv.org/")
    print("   Search: 'cancer' → Filter by 'Cancer Biology'")
    print("\n2. PubMed Central - https://www.ncbi.nlm.nih.gov/pmc/")
    print("   Search: 'cancer' → Filter 'Free full text'")
    print("\n3. PLOS ONE - https://journals.plos.org/plosone/")
    print("   Search: 'cancer' → Download PDFs")
    print("\n4. Europe PMC - https://europepmc.org/")
    print("   Search: 'cancer' → Filter 'Open Access'")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
