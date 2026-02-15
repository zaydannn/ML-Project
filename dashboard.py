import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

orders_path = os.path.join(BASE_DIR, "olist_orders_dataset.csv")

orders = pd.read_csv(orders_path)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Veridi Logistics",
    page_icon=":material/local_shipping:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR METRICS & ICONS ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #555;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_and_process_data():
    try:
        # Load datasets
        orders = pd.read_csv(orders_path)
        reviews = pd.read_csv('olist_order_reviews_dataset.csv')
        customers = pd.read_csv('olist_customers_dataset.csv')
        products = pd.read_csv('olist_products_dataset.csv')
        items = pd.read_csv('olist_order_items_dataset.csv')
        translations = pd.read_csv('product_category_name_translation.csv')
        
        # Load product data (Optional, for advanced filters)
            # Merge Product Details
        product_full = items.merge(products, on='product_id', how='left')
        product_full = product_full.merge(translations, on='product_category_name', how='left')
            
        # Getting first product category per order for easier analysi
        order_products = product_full.groupby('order_id').first().reset_index()


        #Removing duplicate Reviews (Deleting older ones, keeping most recent) frm reviews dataset
        reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'])
        reviews_pp = reviews.sort_values(by='review_answer_timestamp', ascending=False).drop_duplicates(subset=['order_id'], keep='first')

        #Merging Orders, Reviews, and Customers datasets
        df = orders.merge(reviews_pp, on='order_id', how='left')
        df = df.merge(customers, on='customer_id', how='left')
        
        if not order_products.empty:
            df = df.merge(order_products[['order_id', 'product_category_name_english']], on='order_id', how='left')

        # Date Conversion & Filtering using pandas datetime and the relevant date columns
        cols = ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # Dropping rows where delivery date is missing or incomplete
        delivered = df.dropna(subset=['order_delivered_customer_date']).copy()

        # Feature Engineering:
        # Calculating Days Difference (Estimated - Actual) Positive = Early, Negative = Late
        delivered['Days_Difference'] = (delivered['order_estimated_delivery_date'] - delivered['order_delivered_customer_date']).dt.total_seconds() / (24*60*60)
        
        def classify_status(days):
            if days >= 0: return "On Time"
            elif days >= -5: return "Late"
            else: return "Super Late"
            
        delivered['Delivery_Status'] = delivered['Days_Difference'].apply(classify_status)
        delivered['Delivery Delay Days'] = delivered['Days_Difference'].round()
        
        # Calculate order number per customer to identify repeat customers
        delivered = delivered.sort_values(by=['customer_unique_id', 'order_purchase_timestamp'])
        delivered['order_number'] = delivered.groupby('customer_unique_id').cumcount() + 1
        
        # Identifying customers who have made more than one purchase
        repeat_customers = delivered[delivered['order_number'] > 1]['customer_unique_id'].unique()
        delivered['is_repeat_customer'] = delivered['customer_unique_id'].isin(repeat_customers)

        return delivered

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_and_process_data()

if df.empty:
    st.warning("Data not found. Please ensure CSV files are in the directory.")
    st.stop()

# SIDEBAR 
with st.sidebar:
    st.title("Filters")
    
    # Date Range Filter
    min_date = df['order_purchase_timestamp'].min()
    max_date = df['order_purchase_timestamp'].max()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # State Filter
    all_states = sorted(df['customer_state'].unique())
    selected_states = st.multiselect("Filter by State", all_states)
    
    # Product Category Filter
    if 'product_category_name_english' in df.columns:
        cats = df['product_category_name_english'].dropna().unique()
        selected_cats = st.multiselect("Filter by Category", sorted(cats))
    else:
        selected_cats = []


# FILTERING DATA 
mask = (df['order_purchase_timestamp'].dt.date >= date_range[0]) & (df['order_purchase_timestamp'].dt.date <= date_range[1])
if selected_states:
    mask = mask & df['customer_state'].isin(selected_states)
if selected_cats:
    mask = mask & df['product_category_name_english'].isin(selected_cats)

filtered_df = df[mask]

#Home Page

st.title("Veridi Logistics Dashboard")
st.markdown("Audit tool to analyze how **delivery performance** impacts **customer sentiments** and **retention**.")

# TOP KPI ROW
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Orders", value=f"{len(filtered_df):,}", delta="Filtered")

with kpi2:
    avg_score = filtered_df['review_score'].mean()
    st.metric(label="Avg Review Score", value=f"{avg_score:.2f} ")

with kpi3:
    on_time_pct = (len(filtered_df[filtered_df['Delivery_Status'] == 'On Time']) / len(filtered_df)) * 100
    st.metric(label="On-Time Delivery %", value=f"{on_time_pct:.1f}%", delta_color="normal")

with kpi4:
    # Calculate simple retention rate for filtered view
    retention_rate = (filtered_df['is_repeat_customer'].sum() / len(filtered_df)) * 100
    st.metric(label="Repeat Customer Rate", value=f"{retention_rate:.1f}%", help="Percentage of orders from returning customers")

st.markdown("---")

# --- TABS FOR ANALYSIS sections ---
tab1, tab2, tab3 = st.tabs([
    ":material/monitoring: Executive Summary", 
    ":material/local_shipping: Delivery Logistics", 
    ":material/reviews: Customer Satisfaction"
])

#TAB 1: EXECUTIVE SUMMARY (RETENTION & IMPACT OF DELIVERY SPEED)
with tab1:
    st.header("Business Impact: Retention & Reviews")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Does Delivery Speed Affect Retention?")
        # Referencing the retention logic from the data processing step, we can analyze how first delivery experience impacts repeat purchases.
        # Filter for first orders only within the selection to see if they came back
        first_orders = filtered_df[filtered_df['order_number'] == 1]
        
        if not first_orders.empty:
            retention_stats = first_orders.groupby('Delivery_Status').agg(
                total=('customer_unique_id', 'count'),
                returned=('is_repeat_customer', 'sum')
            ).reset_index()
            retention_stats['Retention Rate (%)'] = (retention_stats['returned'] / retention_stats['total']) * 100
            
            fig_retention = px.bar(
                retention_stats,
                x='Delivery_Status',
                y='Retention Rate (%)',
                color='Delivery_Status',
                color_discrete_map={"On Time": "#2ecc71", "Late": "#f1c40f", "Super Late": "#e74c3c"},
                title="Customer Retention Rate by First Delivery Experience",
                text_auto='.2f'
            )
            fig_retention.update_layout(xaxis_title="First Order Status", yaxis_title="Retention Rate (%)")
            st.plotly_chart(fig_retention, use_container_width=True)
        else:
            st.info("Not enough first-order data in current filter for retention analysis.")

    with col_b:
        st.subheader("Delivery Status Split")
        status_counts = filtered_df['Delivery_Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_pie = px.pie(
            status_counts, 
            values='Count', 
            names='Status', 
            hole=0.4,
            color='Status',
            color_discrete_map={"On Time": "#2ecc71", "Late": "#f1c40f", "Super Late": "#e74c3c"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

#  LOGISTICS PERFORMANCE BY REGION
with tab2:
    st.header("Logistics Performance by Region")
    
    # Analysis late delivery percentages by state
    state_group = filtered_df.groupby('customer_state').agg(
        total_orders=('order_id', 'count'),
        late_orders=('Delivery_Status', lambda x: (x != 'On Time').sum())
    ).reset_index()
    state_group['Late %'] = (state_group['late_orders'] / state_group['total_orders']) * 100
    state_group = state_group.sort_values('Late %', ascending=False)
    
    fig_state = px.bar(
        state_group, 
        x='customer_state', 
        y='Late %',
        color='Late %',
        color_continuous_scale='RdYlGn_r', # Red is high late %, Green is low
        title="Percentage of Late Deliveries by State"
    )
    st.plotly_chart(fig_state, use_container_width=True)
    
    st.markdown("!! Problem Areas")
    st.dataframe(
        state_group.head(5).style.format({"Late %": "{:.2f}%"}), 
        use_container_width=True
    )

#  CUSTOMER SATISFACTION 
with tab3:
    st.header("Correlation: Speed vs. Satisfaction")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Review Score by Status
        avg_score_status = filtered_df.groupby('Delivery_Status')['review_score'].mean().reset_index()
        fig_score_bar = px.bar(
            avg_score_status,
            x='Delivery_Status',
            y='review_score',
            color='Delivery_Status',
            color_discrete_map={"On Time": "#2ecc71", "Late": "#f1c40f", "Super Late": "#e74c3c"},
            range_y=[1, 5],
            text_auto='.2f',
            title="Average Review Score: On Time vs Late"
        )
        st.plotly_chart(fig_score_bar, use_container_width=True)
        
    with c2:
        # Review Score vs Days Delay (Line Chart)
        # Group by rounded delay days for cleaner line plot
        delay_impact = filtered_df.groupby('Delivery Delay Days')['review_score'].mean().reset_index()
        # Filter extreme outliers for better visualization (e.g., -20 to +20 days)
        delay_impact = delay_impact[(delay_impact['Delivery Delay Days'] >= -15) & (delay_impact['Delivery Delay Days'] <= 15)]
        
        fig_line = px.line(
            delay_impact,
            x='Delivery Delay Days',
            y='review_score',
            markers=True,
            title="Review Score Trend as Delay Increases",
            labels={'Delivery Delay Days': 'Days Difference (Positive = Early, Negative = Late)'}
        )
        # Add reference line
        fig_line.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Promised Date")
        fig_line.update_traces(line_color="#3498db")
        st.plotly_chart(fig_line, use_container_width=True)

# Viewing the raw data
with st.expander("View Raw Data"):

    st.dataframe(filtered_df[['order_id', 'customer_state', 'order_estimated_delivery_date', 'order_delivered_customer_date', 'Delivery_Status', 'review_score']].head(100))
