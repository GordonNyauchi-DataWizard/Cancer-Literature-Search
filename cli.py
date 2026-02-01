#!/usr/bin/env python3
"""
Command-Line Interface for Cancer Medical Literature Search

This provides an interactive CLI for searching cancer research papers
and getting LLM-enhanced answers to questions.

Usage:
    python cli.py                    # Interactive mode
    python cli.py --rebuild          # Rebuild index from scratch
    python cli.py --query "your query"  # Single query mode
"""

import argparse
import sys
from semantic_search import CancerSearchApp, Config, print_results


class CLI:
    """Interactive command-line interface for semantic search."""
    
    def __init__(self, rebuild: bool = False):
        """
        Initialize CLI.
        
        Args:
            rebuild: Force rebuild of search index
        """
        print("\n🔬 Cancer Medical Literature Search")
        print("="*80)
        
        self.app = CancerSearchApp()
        
        # Build or load index
        try:
            self.app.build_or_load_index(force_rebuild=rebuild)
        except FileNotFoundError:
            print("\n❌ Error: No PDF files found in 'papers/' directory")
            print("\nPlease add at least 100 cancer research papers to the 'papers/' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error building index: {e}")
            sys.exit(1)
    
    def run_interactive(self):
        """Run interactive command-line interface."""
        
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  search <query>     - Search for relevant papers")
        print("  ask <question>     - Ask a question (RAG mode)")
        print("  compare <query>    - Compare results across papers")
        print("  help               - Show this help message")
        print("  quit               - Exit the program")
        print("\nTip: Queries work best when specific (e.g., 'BRCA1 mutations' vs 'cancer')")
        print("="*80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("🔍 > ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                query = parts[1] if len(parts) > 1 else ""
                
                # Execute command
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == 'help':
                    self._show_help()
                
                elif command == 'search':
                    if not query:
                        print("❌ Usage: search <query>")
                        continue
                    self._handle_search(query)
                
                elif command == 'ask':
                    if not query:
                        print("❌ Usage: ask <question>")
                        continue
                    self._handle_question(query)
                
                elif command == 'compare':
                    if not query:
                        print("❌ Usage: compare <query>")
                        continue
                    self._handle_compare(query)
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_search(self, query: str):
        """
        Handle search command.
        
        Args:
            query: Search query string
        """
        print(f"\n🔍 Searching for: '{query}'")
        print("-"*80)
        
        try:
            response = self.app.search(query, top_k=Config.TOP_K, enhance=True)
            print_results(response)
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    def _handle_question(self, question: str):
        """
        Handle question answering (RAG mode).
        
        Args:
            question: User's question
        """
        print(f"\n💬 Question: '{question}'")
        print("-"*80)
        print("\n🤔 Thinking...\n")
        
        try:
            answer = self.app.answer_question(question, top_k=Config.TOP_K)
            print("📝 Answer:")
            print("-"*80)
            print(answer)
            print()
        except Exception as e:
            print(f"❌ Failed to answer: {e}")
    
    def _handle_compare(self, query: str):
        """
        Handle comparative analysis.
        
        Args:
            query: Search query for comparison
        """
        print(f"\n🔍 Comparing results for: '{query}'")
        print("-"*80)
        
        try:
            # Get search results
            results = self.app.index.search(query, top_k=Config.TOP_K)
            
            # Generate comparison
            comparison = self.app.llm.compare_results(results)
            
            print("\n📊 Comparative Analysis:")
            print("-"*80)
            print(comparison)
            print()
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    
    def _show_help(self):
        """Show help message."""
        print("\n" + "="*80)
        print("HELP")
        print("="*80)
        print("\nAvailable Commands:")
        print()
        print("  search <query>")
        print("    Perform semantic search and get top relevant chunks")
        print("    Example: search immunotherapy side effects")
        print()
        print("  ask <question>")
        print("    Ask a question and get an AI-generated answer based on papers")
        print("    Example: ask what are the latest advances in CAR-T therapy?")
        print()
        print("  compare <query>")
        print("    Get comparative analysis across multiple relevant papers")
        print("    Example: compare checkpoint inhibitors vs traditional chemotherapy")
        print()
        print("  help")
        print("    Show this help message")
        print()
        print("  quit / exit / q")
        print("    Exit the program")
        print()
        print("Tips:")
        print("  - Be specific in your queries for better results")
        print("  - Use medical terminology when appropriate")
        print("  - Ask follow-up questions to dive deeper")
        print("="*80 + "\n")
    
    def run_single_query(self, query: str, mode: str = "search"):
        """
        Run a single query and exit.
        
        Args:
            query: Query string
            mode: Query mode ('search', 'ask', or 'compare')
        """
        if mode == "search":
            self._handle_search(query)
        elif mode == "ask":
            self._handle_question(query)
        elif mode == "compare":
            self._handle_compare(query)
        else:
            print(f"❌ Unknown mode: {mode}")


def main():
    """Main entry point for CLI."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Semantic search system for cancer medical literature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                              # Interactive mode
  python cli.py --rebuild                    # Rebuild index
  python cli.py --query "BRCA1 mutations"    # Single search query
  python cli.py --ask "What is immunotherapy?" # Single question
        """
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of search index"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Perform a single search query and exit"
    )
    
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a single question and exit"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        help="Perform comparative analysis and exit"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=Config.TOP_K,
        help=f"Number of results to return (default: {Config.TOP_K})"
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.top_k:
        Config.TOP_K = args.top_k
    
    # Initialize CLI
    try:
        cli = CLI(rebuild=args.rebuild)
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)
    
    # Run appropriate mode
    if args.query:
        cli.run_single_query(args.query, mode="search")
    elif args.ask:
        cli.run_single_query(args.ask, mode="ask")
    elif args.compare:
        cli.run_single_query(args.compare, mode="compare")
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
