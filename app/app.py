# =============================================================================
# app/streamlit_app.py
# PURPOSE: Production-grade Streamlit web application for spam detection
#
# RUN COMMAND:
#   streamlit run app/streamlit_app.py
# =============================================================================

import sys
import os

# Add project root to path so we can import from 'src'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

# -----------------------------------------------------------------------
# PAGE CONFIGURATION — must be FIRST streamlit command
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="📧 Spam Email Detector",
    page_icon="📧",
    layout="wide",               # Use full browser width
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------
# CUSTOM CSS — makes the app look professional
# -----------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Spam result box */
    .spam-box {
        background: linear-gradient(135deg, #FF6B6B, #ee5a24);
        color: white;
        padding: 25px 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255,107,107,0.4);
    }

    /* Ham result box */
    .ham-box {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
        padding: 25px 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(38,222,129,0.4);
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4ECDC4;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Header styling */
    h1 { color: #2c3e50; }
    h2 { color: #34495e; }
    h3 { color: #7f8c8d; }

    /* Sidebar */
    .css-1d391kg { background-color: #2c3e50; }

    /* Warning text */
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# LOAD MODEL — use Streamlit caching for performance
# @st.cache_resource = load once, reuse for all users (doesn't reload on every click)
# -----------------------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    """
    Load and cache the trained model pipeline.
    st.cache_resource keeps this in memory so it's not reloaded on every page interaction.
    """
    try:
        from src.predict import predict_email, get_spam_keywords, load_pipeline
        load_pipeline()
        return predict_email, get_spam_keywords, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"


predict_email, get_spam_keywords, load_error = load_model_pipeline()


# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------
def create_gauge_chart(spam_prob: float) -> go.Figure:
    """
    Create a semicircular gauge chart showing spam probability.
    Green = safe, Yellow = borderline, Red = spam
    """
    if spam_prob < 30:
        color = "#26de81"    # Green — safe
        label = "SAFE"
    elif spam_prob < 60:
        color = "#f7b731"    # Yellow — suspicious
        label = "SUSPICIOUS"
    else:
        color = "#FF6B6B"    # Red — spam
        label = "SPAM"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=spam_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Spam Probability<br><span style='font-size:0.8em;color:{color}'>{label}</span>",
               'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar':  {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0,  30], 'color': "#d4efdf"},   # Light green zone
                {'range': [30, 60], 'color': "#fdebd0"},   # Light yellow zone
                {'range': [60, 100], 'color': "#fadbd8"},  # Light red zone
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': spam_prob
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def create_probability_bar(spam_prob: float, ham_prob: float) -> go.Figure:
    """Horizontal stacked bar showing spam vs ham probability."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Probability"], x=[ham_prob],
        name="Ham (Safe)", orientation='h',
        marker_color='#26de81',
        text=f"HAM: {ham_prob:.1f}%",
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=14, family='Arial Black')
    ))
    fig.add_trace(go.Bar(
        y=["Probability"], x=[spam_prob],
        name="Spam", orientation='h',
        marker_color='#FF6B6B',
        text=f"SPAM: {spam_prob:.1f}%",
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=14, family='Arial Black')
    ))
    fig.update_layout(
        barmode='stack',
        height=100,
        margin=dict(t=10, b=10, l=0, r=0),
        showlegend=False,
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def format_text_stats(result: dict):
    """Format email statistics into a clean display."""
    stats = {
        "📝 Characters"       : result['char_count'],
        "📖 Words"            : result['word_count'],
        "💲 Has Currency ($£€)": "Yes ⚠️" if result['has_currency'] else "No ✅",
        "🔠 Uppercase Ratio"  : f"{result['uppercase_ratio']*100:.1f}%",
        "🔍 Cleaned Words"    : len(result['cleaned'].split()) if result['cleaned'] else 0,
    }
    return stats


# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📧 Spam Detector")
    st.markdown("---")

    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses **Machine Learning** to detect spam emails.

    **Model:** TF-IDF + Logistic Regression / Naive Bayes

    **Pipeline:**
    1. Clean your text
    2. Convert to TF-IDF numbers
    3. Predict spam probability
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    show_advanced = st.checkbox("Show Advanced Details", value=False)
    show_keywords = st.checkbox("Show Spam Keywords", value=True)

    st.markdown("---")
    st.markdown("### 📊 Threshold")
    threshold = st.slider(
        "Spam threshold (%)",
        min_value=10, max_value=90, value=50, step=5,
        help="Emails with spam probability above this are marked as spam"
    )

    st.markdown("---")
    st.info("💡 **Tip:** Real spam emails often contain words like 'FREE', 'WIN', 'CLAIM', currency symbols, and urgent language.")

    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("🐍 Python • 🤖 Scikit-learn • 🌟 Streamlit")


# -----------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------

# Header
st.markdown("# 📧 Spam Email Detector")
st.markdown("##### Powered by Machine Learning — TF-IDF + Logistic Regression")
st.markdown("---")

# -----------------------------------------------------------------------
# MODEL STATUS CHECK
# -----------------------------------------------------------------------
if load_error:
    st.error(f"""
    ### ❌ Model Not Found

    The trained model files are missing. Please run the training pipeline first:

    ```bash
    python run.py
    ```

    **Error details:** `{load_error}`
    """)
    st.stop()   # Stop rendering the rest of the page

else:
    st.success("✅ Model loaded and ready!")

# -----------------------------------------------------------------------
# TAB LAYOUT
# -----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Detect Spam", "📊 Batch Analysis", "📖 About the Model"])


# ===========================
# TAB 1: Single Email Detection
# ===========================
with tab1:
    st.markdown("### Enter an Email to Analyze")

    # Example emails dropdown
    examples = {
        "Select an example or type your own...": "",
        "🚫 SPAM: Prize Winner": "Congratulations! You've won a $1000 Walmart gift card. Click here NOW to claim your prize before it expires tonight! Text WIN to 88600. Limited time offer!",
        "🚫 SPAM: Loan Offer": "URGENT: Your loan of $10,000 has been APPROVED! Get cash deposited into your account today. No credit check needed. Call 0800-LOAN-NOW immediately!",
        "✅ HAM: Work Email": "Hi team, the project deadline has been extended to next Friday. Please review the latest requirements document and let me know if you have any questions.",
        "✅ HAM: Personal": "Hey! Are you coming to dinner on Sunday? Mom is cooking and wants to know how many people to expect. Let me know by tomorrow please.",
        "✅ HAM: Delivery": "Your order #12345 has been shipped and is on its way. Expected delivery is in 3-5 business days. Track your package using the link in the order confirmation email.",
    }

    selected = st.selectbox("Try an example:", list(examples.keys()))
    default_text = examples[selected]

    # Text area for email input
    email_input = st.text_area(
        "Email message:",
        value=default_text,
        height=180,
        placeholder="Paste or type your email message here...",
        help="Enter the full email text you want to analyze"
    )

    # Character counter
    col_count1, col_count2 = st.columns([3, 1])
    with col_count2:
        st.caption(f"Characters: {len(email_input)} | Words: {len(email_input.split()) if email_input else 0}")

    # Analyze button
    analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # PREDICTION OUTPUT
    # -----------------------------------------------------------------------
    if analyze_clicked:
        if not email_input.strip():
            st.warning("⚠️ Please enter an email message first.")
        elif len(email_input.strip()) < 5:
            st.warning("⚠️ Message is too short. Please enter a longer email.")
        else:
            # Show loading spinner during prediction
            with st.spinner("🔍 Analyzing email..."):
                time.sleep(0.4)  # Small delay for better UX
                result = predict_email(email_input)

            # Apply custom threshold
            is_spam = result['spam_prob'] >= threshold

            st.markdown("---")
            st.markdown("### 📊 Results")

            # --- Result Banner ---
            if is_spam:
                st.markdown(f"""
                <div class="spam-box">
                    🚫 SPAM DETECTED &nbsp;|&nbsp; {result['spam_prob']:.1f}% Confidence
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ham-box">
                    ✅ SAFE EMAIL (Not Spam) &nbsp;|&nbsp; {result['ham_prob']:.1f}% Confidence
                </div>
                """, unsafe_allow_html=True)

            # --- Three Columns: Gauge | Bar | Stats ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Spam Probability Gauge")
                st.plotly_chart(
                    create_gauge_chart(result['spam_prob']),
                    use_container_width=True
                )

            with col2:
                st.markdown("##### Spam vs Ham Breakdown")
                st.plotly_chart(
                    create_probability_bar(result['spam_prob'], result['ham_prob']),
                    use_container_width=True
                )
                st.markdown("##### Email Statistics")
                stats = format_text_stats(result)
                for key, val in stats.items():
                    col_k, col_v = st.columns([2, 1])
                    col_k.markdown(f"**{key}**")
                    col_v.markdown(f"`{val}`")

            # --- Advanced Details ---
            if show_advanced:
                with st.expander("🔬 Advanced Analysis Details"):
                    st.markdown("**Cleaned Text (after preprocessing):**")
                    st.code(result['cleaned'] if result['cleaned'] else "(empty after cleaning)")

                    st.markdown(f"""
                    **Raw Probabilities:**
                    - Spam Probability: `{result['spam_prob']:.4f}%`
                    - Ham Probability : `{result['ham_prob']:.4f}%`
                    - Decision Threshold: `{threshold}%`
                    """)

            # --- Spam Keywords ----
            if show_keywords:
                with st.expander("🔑 Top Spam Indicator Keywords (Model Learned)"):
                    keywords = get_spam_keywords(top_n=20)
                    if keywords:
                        kw_df = pd.DataFrame(keywords, columns=['Word', 'Spam Score'])
                        fig = px.bar(
                            kw_df.head(15),
                            x='Spam Score', y='Word',
                            orientation='h',
                            color='Spam Score',
                            color_continuous_scale='Reds',
                            title='Words Most Associated with SPAM'
                        )
                        fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Keywords not available for this model type.")

# ===========================
# TAB 2: Batch Analysis
# ===========================
with tab2:
    st.markdown("### 📊 Batch Email Analysis")
    st.markdown("Paste multiple emails (one per line) to analyze them all at once.")

    batch_input = st.text_area(
        "Enter multiple emails (one per line):",
        height=250,
        placeholder="Email 1 text here\nEmail 2 text here\nEmail 3 text here",
        help="Each line is treated as a separate email"
    )

    if st.button("🔍 Analyze All", type="primary", use_container_width=True):
        if not batch_input.strip():
            st.warning("Please enter at least one email.")
        else:
            emails = [e.strip() for e in batch_input.strip().split('\n') if e.strip()]

            if len(emails) > 50:
                st.warning("⚠️ Maximum 50 emails at once. Using first 50.")
                emails = emails[:50]

            st.markdown(f"**Analyzing {len(emails)} emails...**")
            progress = st.progress(0)
            results_list = []

            for i, email in enumerate(emails):
                res = predict_email(email)
                is_spam = res['spam_prob'] >= threshold
                results_list.append({
                    'Email Preview' : email[:60] + ('...' if len(email) > 60 else ''),
                    'Prediction'    : '🚫 SPAM' if is_spam else '✅ HAM',
                    'Spam %'        : f"{res['spam_prob']:.1f}%",
                    'Ham %'         : f"{res['ham_prob']:.1f}%",
                    'Words'         : res['word_count'],
                })
                progress.progress((i + 1) / len(emails))

            # Summary metrics
            spam_count = sum(1 for r in results_list if 'SPAM' in r['Prediction'])
            ham_count  = len(results_list) - spam_count

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Emails", len(results_list))
            m2.metric("🚫 Spam", spam_count, delta=f"{spam_count/len(results_list)*100:.0f}%")
            m3.metric("✅ Ham (Safe)", ham_count, delta=f"{ham_count/len(results_list)*100:.0f}%")

            # Results table
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True)

            # Pie chart
            if spam_count > 0 or ham_count > 0:
                fig = px.pie(
                    values=[spam_count, ham_count],
                    names=['🚫 Spam', '✅ Ham'],
                    color_discrete_sequence=['#FF6B6B', '#26de81'],
                    title='Batch Analysis Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv,
                file_name="spam_analysis_results.csv",
                mime="text/csv"
            )


# ===========================
# TAB 3: About the Model
# ===========================
with tab3:
    st.markdown("### 📖 How This Model Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔄 The ML Pipeline

        **Step 1: Text Cleaning**
        - Convert to lowercase
        - Remove URLs, numbers, punctuation
        - Remove stopwords (the, is, and...)

        **Step 2: TF-IDF Vectorization**
        - Convert each word to a numerical score
        - TF = how often the word appears in THIS email
        - IDF = how rare the word is across ALL emails
        - High TF-IDF = the word is important for this email

        **Step 3: Model Prediction**
        - Logistic Regression finds the best decision boundary
        - Returns probability for spam and ham
        - If spam probability > threshold → SPAM

        **Step 4: Result**
        - Label: SPAM or HAM
        - Confidence percentage
        """)

    with col2:
        st.markdown("""
        #### 🤖 Why Logistic Regression?

        | Feature | Detail |
        |---|---|
        | **Speed** | Predicts in < 1ms |
        | **Accuracy** | 97-99% on real data |
        | **Interpretable** | Can see which words matter |
        | **Reliable** | Works great for text |

        #### 📊 Typical Performance
        | Metric | Score |
        |---|---|
        | Accuracy | ~98% |
        | Precision | ~97% |
        | Recall | ~96% |
        | F1 Score | ~96% |

        *(Performance varies with dataset size)*

        #### 📦 Training Dataset
        - **SMS Spam Collection** from UCI/Kaggle
        - 5,572 real SMS/email messages
        - Labelled by humans as spam or ham
        """)

    st.markdown("---")
    st.markdown("#### 💡 Tips for Better Detection")
    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.info("📌 For best results, paste the **full email text** including the subject line.")
    with tips_col2:
        st.info("📌 The model works best with **English text**. Other languages may give less accurate results.")

# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7f8c8d; font-size:13px;'>"
    "📧 Spam Email Detector | Built with Python + Scikit-learn + Streamlit | NLP Practice Project"
    "</div>",
    unsafe_allow_html=True
)
