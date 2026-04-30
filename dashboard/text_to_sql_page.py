import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile


def render_text_to_sql_page(df, data, selected_season=None):

    st.markdown(
        "<h1 style='font-family:Playfair Display,serif;color:#C41E1E'>Ask the Data</h1>",
        unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#F8F8F8;border-left:4px solid #C41E1E;padding:16px 20px;"
        "border-radius:0 8px 8px 0;margin:10px 0;font-size:0.95rem;color:#2d2d2d;line-height:1.7'>"
        "Ask any question about F1 data in plain English. Our AI agent converts your question "
        "into a database query and returns the results instantly."
        "<br><br><b>Powered by:</b> Groq LLaMA 3.1 + SQLite"
        "</div>", unsafe_allow_html=True)

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
        if col.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["sql_input"] = ex
            st.rerun()

    st.markdown("---")

    question = st.text_input(
        "Type your question about F1 data:",
        placeholder="e.g., Who won the most races in 2024?",
        key="sql_input"
    )

    if question:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass

        if not api_key:
            st.warning("Please set GROQ_API_KEY in your environment or Streamlit secrets.")
            st.code("export GROQ_API_KEY='your_key_here'")
            return

        with st.spinner("AI is converting your question to SQL..."):
            try:
                from groq import Groq
                client = Groq(api_key=api_key)

                columns_info = []
                for col_name in df.columns:
                    dtype = str(df[col_name].dtype)
                    sample = str(df[col_name].dropna().iloc[0]) if len(df[col_name].dropna()) > 0 else "N/A"
                    columns_info.append(f"  - {col_name} ({dtype}), example: {sample}")

                schema = "\n".join(columns_info[:50])

                prompt = (
                    "You are a SQLite expert. Convert the user question into a SQLite query.\n\n"
                    "CRITICAL RULES:\n"
                    "- Use ONLY SQLite syntax. No YEAR(), no MONTH(), no DATE_FORMAT()\n"
                    "- For year filtering use: CAST(season AS INTEGER) = 2024\n"
                    "- The table has a season column (integer year) - use it instead of parsing dates\n"
                    "- target_winner = 1 means the driver won that race\n"
                    "- target_podium = 1 means top 3 finish\n"
                    "- target_position = finishing position (1 = first place)\n\n"
                    "The database has ONE table called race_data with these columns:\n"
                    + schema + "\n\n"
                    "Key columns:\n"
                    "- driver_name = full driver name (e.g. Max Verstappen)\n"
                    "- constructor_name = team name (e.g. Red Bull)\n"
                    "- season = year (2018-2026)\n"
                    "- round = race number within season\n"
                    "- race_name = Grand Prix name\n"
                    "- target_position = finishing position (1 = winner)\n"
                    "- target_podium = 1 if finished top 3, 0 otherwise\n"
                    "- target_winner = 1 if won, 0 otherwise\n"
                    "- grid = starting grid position\n"
                    "- constructor_rolling_points = team strength metric\n"
                    "- rolling_avg_position_5 = driver avg finish last 5 races\n\n"
                    "User question: " + question + "\n\n"
                    "Return ONLY the raw SQL query. No explanation, no markdown, no backticks. "
                    "Limit to 20 rows. Order results logically. "
                    "When asking about drivers or teams, use GROUP BY to return one row per driver or team. "
                    "For ratings or averages, use the MAX or AVG of the latest season data."
                )

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                sql_query = response.choices[0].message.content.strip()

                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                st.markdown("**Generated SQL:**")
                st.code(sql_query, language="sql")

                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_path = tmp.name

                conn = sqlite3.connect(tmp_path)
                df.to_sql("race_data", conn, if_exists="replace", index=False)

                try:
                    result = pd.read_sql(sql_query, conn)
                    conn.close()
                    os.unlink(tmp_path)

                    if len(result) == 0:
                        st.info("Query returned no results. Try rephrasing your question.")
                    else:
                        st.markdown(f"**Results ({len(result)} rows):**")
                        st.dataframe(result, use_container_width=True, hide_index=True)

                        if len(result.columns) >= 2:
                            last_col = result.columns[-1]
                            if result[last_col].dtype in ["int64", "float64"]:
                                st.markdown(
                                    "<div style='background:#E8F5EE;border-left:4px solid #059669;"
                                    "padding:10px 16px;border-radius:0 8px 8px 0;margin-top:10px;"
                                    "font-size:0.9rem'>"
                                    "Quick stats for " + last_col + ": "
                                    "Min=" + str(round(result[last_col].min(), 2)) + ", "
                                    "Max=" + str(round(result[last_col].max(), 2)) + ", "
                                    "Mean=" + str(round(result[last_col].mean(), 2)) +
                                    "</div>", unsafe_allow_html=True)

                except Exception as sql_err:
                    conn.close()
                    os.unlink(tmp_path)
                    st.error(f"SQL execution error: {sql_err}")
                    st.info("The AI generated invalid SQL. Try rephrasing your question.")

            except ImportError:
                st.error("Please install: pip install groq")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;color:#949594;text-align:center'>"
        "Powered by Groq LLaMA 3.1 - Queries run on local SQLite"
        "</div>", unsafe_allow_html=True)
