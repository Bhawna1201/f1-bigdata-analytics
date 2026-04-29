"""
Text-to-SQL Page — Add as a new page in your Streamlit dashboard
Users type natural language questions → Gemini converts to SQL → Results shown

Setup:
  pip install google-generativeai
  Set GEMINI_API_KEY in environment or Streamlit secrets

Integration:
  1. Add "🔍 Ask the Data" to your sidebar radio options
  2. Add this as a new elif page block in app.py
  3. Or import and call: render_text_to_sql_page(df, data)
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile


def render_text_to_sql_page(df, data, selected_season=None):
    """
    Render the Text-to-SQL page.
    Add this to your app.py as a new page.
    """
    
    F1_RED = "#C41E1E"
    GRAY = "#949594"
    
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Ask the Data</h1>",
        unsafe_allow_html=True)
    
    st.markdown(
        f"<div style='background:#F8F8F8;border-left:4px solid {F1_RED};padding:16px 20px;"
        f"border-radius:0 8px 8px 0;margin:10px 0;font-size:0.95rem;color:#2d2d2d;line-height:1.7'>"
        f"Ask any question about F1 data in plain English. Our AI agent converts your question "
        f"into a database query and returns the results instantly."
        f"<br><br><b>Powered by:</b> Google Gemini 2.0 Flash + SQLite"
        f"</div>", unsafe_allow_html=True)
    
    # Example questions
    st.markdown("**Try these example questions:**")
    examples = [
        "Who won the most races in 2024?",
        "What is the average finishing position for Max Verstappen?",
        "Which constructor has the most podiums across all seasons?",
        "Show me all races where the grid position 1 driver did not win",
        "What is the average qualifying gap to pole for each team in 2025?",
        "Which driver gained the most positions from grid to finish?",
        "How many races has each constructor won?",
        "Show the top 5 drivers by ELO rating",
    ]
    
    col1, col2 = st.columns(2)
    for i, ex in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        if col.button(f"📊 {ex}", key=f"ex_{i}", use_container_width=True):
            st.session_state["sql_question"] = ex
    
    st.markdown("---")
    
    # Question input
    question = st.text_input(
        "🔍 Type your question about F1 data:",
        value=st.session_state.get("sql_question", ""),
        placeholder="e.g., Who won the most races in 2024?",
        key="sql_input"
    )
    
    if question:
        # Get API key
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            # Try Streamlit secrets
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except:
                pass
        
        if not api_key:
            st.warning("⚠️ Please set GEMINI_API_KEY in your environment or Streamlit secrets to use this feature.")
            st.code("# In terminal:\nexport GEMINI_API_KEY='your_key_here'\n\n# Or in .streamlit/secrets.toml:\nGEMINI_API_KEY = 'your_key_here'")
            return
        
        with st.spinner("🤖 AI is converting your question to SQL..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Build schema description from actual DataFrame
                columns_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
                    columns_info.append(f"  - {col} ({dtype}), example: {sample}")
                
                schema = "\n".join(columns_info[:50])  # Limit to 50 columns
                
                prompt = f"""You are a SQL expert. Convert the user's question into a SQLite query.

The database has ONE table called 'race_data' with these columns:
{schema}

Key relationships:
- driver_name = full driver name (e.g., 'Max Verstappen')
- constructor_name = team name (e.g., 'Red Bull')  
- season = year (2018-2026)
- round = race number within season
- race_name = Grand Prix name
- target_position = finishing position (1 = winner)
- target_podium = 1 if finished top 3, 0 otherwise
- target_winner = 1 if won, 0 otherwise
- grid = starting grid position
- constructor_rolling_points = team strength metric
- rolling_avg_position_5 = driver's avg finish last 5 races
- driver_elo_rating = chess-style skill rating

User question: {question}

Rules:
1. Return ONLY the SQL query — no explanation, no markdown, no backticks
2. Use standard SQLite syntax
3. Limit results to 20 rows unless the question asks for all
4. Use meaningful column aliases for readability
5. Order results logically (most wins first, best position first, etc.)"""

                response = model.generate_content(prompt)
                sql_query = response.text.strip()
                
                # Clean up any markdown formatting
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                # Show the generated SQL
                st.markdown("**Generated SQL:**")
                st.code(sql_query, language="sql")
                
                # Execute against DataFrame using SQLite
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                    tmp_path = tmp.name
                
                conn = sqlite3.connect(tmp_path)
                df.to_sql('race_data', conn, if_exists='replace', index=False)
                
                try:
                    result = pd.read_sql(sql_query, conn)
                    conn.close()
                    os.unlink(tmp_path)
                    
                    if len(result) == 0:
                        st.info("Query returned no results. Try rephrasing your question.")
                    else:
                        st.markdown(f"**Results ({len(result)} rows):**")
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        
                        # Quick stats
                        if len(result.columns) >= 2 and result[result.columns[-1]].dtype in ['int64', 'float64']:
                            num_col = result.columns[-1]
                            st.markdown(
                                f"<div style='background:#E8F5EE;border-left:4px solid #059669;"
                                f"padding:10px 16px;border-radius:0 8px 8px 0;margin-top:10px;"
                                f"font-size:0.9rem'>"
                                f"📊 <b>Quick stats for {num_col}:</b> "
                                f"Min={result[num_col].min():.2f}, "
                                f"Max={result[num_col].max():.2f}, "
                                f"Mean={result[num_col].mean():.2f}"
                                f"</div>", unsafe_allow_html=True)
                
                except Exception as sql_err:
                    conn.close()
                    os.unlink(tmp_path)
                    st.error(f"SQL execution error: {sql_err}")
                    st.info("The AI generated invalid SQL. Try rephrasing your question.")
                    
            except ImportError:
                st.error("Please install: pip install google-generativeai")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.8rem;color:{GRAY};text-align:center'>"
        f"Powered by Google Gemini 2.0 Flash • Queries run on local SQLite • No data leaves your machine"
        f"</div>", unsafe_allow_html=True)


# ── Standalone test ──
if __name__ == "__main__":
    st.set_page_config(page_title="F1 Ask the Data", layout="wide")
    
    # Load sample data
    import glob
    gold_files = glob.glob("data/gold/*.parquet")
    if gold_files:
        df = pd.read_parquet(gold_files[-1])
        render_text_to_sql_page(df, {})
    else:
        st.error("No Gold data found. Run the pipeline first.")
