import pandas as pd
import gradio as gr
from typing import List, Optional
from engine import LexicalSearch, SemanticSearch, reciprocal_rank_fusion
from sentence_transformers import CrossEncoder

# Load data
print("📊 Loading dataset...")
df = pd.read_parquet("./data/news_with_embeddings.parquet")
print(f"✅ Loaded {len(df)} news articles")

# Constants
RETRIVERS_PATH = './data/retrivers/'
INDEX_PATH = './data/'

# Combined search function
def search_and_rerank(
    query: str,
    search_type: str = "hybrid",
    use_rerank: bool = True,
    k: int = 20,
    limit: int = 10,
    categories: List[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    year_range = None
    if start_year is not None and end_year is not None:
        year_range = (int(start_year), int(end_year))
    
    # Perform searches based on selected type
    bm25_results = []
    faiss_results = []
    
    if search_type in ["lexical", "hybrid"]:
        bm25_results = lexical.search(query, k=k, limit=limit, category=categories, year_range=year_range)
    
    if search_type in ["semantic", "hybrid"]:
        faiss_results = semantic.search(query, k=k, limit=limit, category=categories, year_range=year_range)
    
    # Combine results if hybrid
    if search_type == "hybrid":
        combined_results = reciprocal_rank_fusion([bm25_results, faiss_results], k=60)
    elif search_type == "lexical":
        combined_results = bm25_results
    else:  # semantic
        combined_results = faiss_results
    
    # Apply reranking if selected
    if use_rerank and len(combined_results) > 0:
        rerank_inputs = []
        for result in combined_results:
            content = result['headline']
            if result.get('short_description'):
                content += " " + result['short_description']
            rerank_inputs.append((query, content))
        
        # Get similarity scores
        rerank_scores = rerank_model.predict(rerank_inputs)
        
        # Update scores
        for i, score in enumerate(rerank_scores):
            combined_results[i]['rerank_score'] = float(score)
            combined_results[i]['original_score'] = combined_results[i]['score']
            combined_results[i]['score'] = float(score)
        
        # Sort by reranking score
        reranked_results = sorted(combined_results, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_results[:limit]
    
    # Return results without reranking
    return combined_results[:limit]

# Format results for display
def format_results(results):
    """Format search results for display in Gradio interface."""
    if not results:
        return "🔍 No results found. Try adjusting your search terms or filters."
    
    formatted = []
    for i, res in enumerate(results, 1):
        score_info = f"(Score: {res['score']:.4f})"
        if 'rerank_score' in res:
            score_info = f"(Rerank: {res['rerank_score']:.4f}, Original: {res['original_score']:.4f})"
        
        date_str = res.get('date', 'Unknown date')
        if isinstance(date_str, pd.Timestamp):
            date_str = date_str.strftime('%Y-%m-%d')
        
        entry = f"**{i}. [{res['headline']}]({res['link']})** {score_info}\n"
        entry += f"**Category:** {res['category']} | **Date:** {date_str}\n"
        entry += f"**Authors:** {res.get('authors', 'Unknown')}\n"
        entry += f"**Description:** {res.get('short_description', 'No description available.')}\n\n"
        
        formatted.append(entry)
    
    return "".join(formatted)

# Get unique categories from the dataset
available_categories = sorted(df['category'].unique().tolist())

# Gradio Interface
with gr.Blocks(title="📰 News Search Engine") as app:
    gr.Markdown("# 📰 News Search Engine")
    gr.Markdown("Search through news articles using keywords or semantic meaning")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="🔎 Search Query",
                placeholder="Enter your search query here...",
                lines=1
            )
        
        with gr.Column(scale=1):
            search_button = gr.Button("🔍 Search", variant="primary")
    
    with gr.Row():
        with gr.Column():
            search_type = gr.Radio(
                label="🔄 Search Type",
                choices=["hybrid", "lexical", "semantic"],
                value="hybrid",
                info="Choose between lexical (keyword-based), semantic (meaning-based), or hybrid search"
            )
            
            use_rerank = gr.Checkbox(
                label="📊 Apply Reranking",
                value=True,
                info="Use a cross-encoder model to rerank results for better relevance"
            )
            
            k_slider = gr.Slider(
                minimum=5,
                maximum=50,
                value=20,
                step=5,
                label="📈 Top-K Initial Results",
                info="Number of initial results to retrieve before filtering and reranking"
            )
            
            limit_slider = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="📋 Results to Display",
                info="Number of results to show after processing"
            )
            
        with gr.Row():
        
            with gr.Column():
                category_selector = gr.Dropdown(
                    choices=available_categories,
                    label="🏷️ Filter by Categories",
                    multiselect=True,
                    info="Filter results by news categories"
                )
                
                with gr.Row():
                    start_year = gr.Number(
                        label="📅 Start Year",
                        value=2020,
                        precision=0,
                        info="Filter results from this publication year onward"
                    )
                    end_year = gr.Number(
                        label="📅 End Year",
                        value=2025,
                        precision=0,
                        info="Filter results up to this publication year"
                    )
    
    results_output = gr.Markdown(label="🔍 Search Results")
    
    # Connect the interface
    search_button.click(
        fn=lambda query, stype, rerank, k, limit, cats, start, end: format_results(
            search_and_rerank(
                query=query,
                search_type=stype,
                use_rerank=rerank,
                k=k,
                limit=limit,
                categories=cats,
                start_year=start,
                end_year=end
            )
        ),
        inputs=[query_input, search_type, use_rerank, k_slider, limit_slider, category_selector, start_year, end_year],
        outputs=results_output
    )
    
    query_input.submit(
        fn=lambda query, stype, rerank, k, limit, cats, start, end: format_results(
            search_and_rerank(
                query=query,
                search_type=stype,
                use_rerank=rerank,
                k=k,
                limit=limit,
                categories=cats,
                start_year=start,
                end_year=end
            )
        ),
        inputs=[query_input, search_type, use_rerank, k_slider, limit_slider, category_selector, start_year, end_year],
        outputs=results_output
    )

# Launch the app
if __name__ == "__main__":
    lexical = LexicalSearch(df)
    semantic = SemanticSearch(df)
    try:
        lexical.load(RETRIVERS_PATH)
    except Exception as e:
        print(e)
        lexical.index()
        lexical.save(RETRIVERS_PATH)
    try:
        semantic.load(INDEX_PATH)
    except Exception as e:
        print(e)
        semantic.build()
        semantic.save(INDEX_PATH)

    print("🔄 Loading reranking model...")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✅ Reranking model loaded")
    print("🚀 Starting News Search Engine...")
    app.launch()
    print("✅ Application closed")