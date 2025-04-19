import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Nova GPU Manufacturing Operations Dashboard", layout="wide")

# --- Data Generation and Loading Functions ---
def load_or_create_inventory_data():
    filepath = "data/inventory_data.csv"
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        # Ensure Safety Stock column exists and has non-zero values
        if 'Safety Stock' not in data.columns or data['Safety Stock'].sum() == 0:
            data['Safety Stock'] = np.random.randint(10, 50, len(data))
            data.to_csv(filepath, index=False)  # Save updated data
        return data
    else:
        data = pd.DataFrame({
            'Component': ['GPU Chip', 'Memory Module', 'PCB', 'Cooling Fan', 'Casing', 'VRM Module', 'Thermal Paste'],
            'Initial Inventory': np.random.randint(100, 300, 7),
            'Lead Time (days)': [10, 7, 15, 5, 8, 9, 2],
            'Scheduled Receipts': np.random.randint(50, 150, 7),
            'Safety Stock': np.random.randint(10, 50, 7)  # Adding Safety Stock
        })
        os.makedirs("data", exist_ok=True)
        data.to_csv(filepath, index=False)
        return data

def load_or_create_demand_data():
    filepath = "data/demand_data.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=['Month'])
    else:
        months = pd.date_range(start='2025-01-01', periods=12, freq='M')
        demand = np.random.poisson(lam=300, size=12)
        data = pd.DataFrame({'Month': months, 'Demand': demand})
        os.makedirs("data", exist_ok=True)
        data.to_csv(filepath, index=False)
        return data

def load_or_create_bom_data():
    filepath = "data/bom_data.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        data = pd.DataFrame({
            'Component': ['GPU Chip', 'Memory Module', 'PCB', 'Cooling Fan', 'Casing', 'VRM Module', 'Thermal Paste'],
            'Quantity per GPU': [1, 4, 1, 2, 1, 1, 1]
        })
        os.makedirs("data", exist_ok=True)
        data.to_csv(filepath, index=False)
        return data

    
def load_or_create_jobs_data():
    filepath = "data/jobs_data.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        jobs = ['Design', 'Wafer Fabrication', 'Photolithography', 'Etching', 'Metallization',
                'Testing (Pre-dicing)', 'Dicing', 'Packaging', 'Final Testing', 'Shipping']
        processing_times = np.random.randint(2, 10, len(jobs))
        due_dates = np.random.randint(10, 30, len(jobs))
        data = pd.DataFrame({'Job': jobs, 'Processing Time (days)': processing_times, 'Due Date (days from today)': due_dates})
        os.makedirs("data", exist_ok=True)
        data.to_csv(filepath, index=False)
        return data

# --- Forecasting Methods ---
def moving_average_forecast(demand, window=3):
    return demand.rolling(window=window).mean()

def moving_average_forecast_with_future(demand, window=3):
    forecast = demand.copy()
    for i in range(len(demand)):
        if i >= window:
            forecast.iloc[i] = demand.iloc[i-window:i].mean()
        elif pd.isna(demand.iloc[i]):  # Handle future months
            forecast.iloc[i] = forecast.iloc[i-window:i].mean()
    return forecast

def exponential_smoothing_forecast(demand, alpha=0.5):
    model = ExponentialSmoothing(demand, trend=None, seasonal=None)
    fit = model.fit(smoothing_level=alpha, optimized=False)
    return fit.fittedvalues

# Gantt Chart for Scheduling
def create_gantt_chart(jobs_df, sequence):
    start = 0
    gantt_data = []
    for job in sequence:
        duration = jobs_df[jobs_df['Job'] == job]['Processing Time (days)'].values[0]
        gantt_data.append({"Task": job, "Start": start, "Finish": start + duration})
        start += duration
    gantt_df = pd.DataFrame(gantt_data)
    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", title="Nova Job Scheduling Gantt Chart")
    fig.update_yaxes(autorange="reversed")
    return fig

# Facility Layout Calculations
def calculate_facility_layout():
    stages = ['Design', 'Wafer Fabrication', 'Photolithography', 'Etching', 'Metallization',
              'Testing (Pre-dicing)', 'Dicing', 'Packaging', 'Final Testing', 'Shipping']
    flows = [300, 290, 285, 280, 275, 270, 260, 255, 250]
    distances = [10, 15, 8, 6, 12, 9, 7, 5, 20]
    total_cost = sum(f*d for f, d in zip(flows, distances))
    return stages, flows, distances, total_cost

# House of Quality Data
def create_hoq_data():
    customer_reqs = ['High Processing Speed', 'Low Power Consumption', 'Affordable Price', 'Durability', 'Quiet Operation']
    technical_attrs = ['GPU Chip Quality', 'Cooling Efficiency', 'PCB Material', 'Thermal Paste Quality', 'Memory Type']
    relationship_matrix = [
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1]
    ]
    return pd.DataFrame(relationship_matrix, index=customer_reqs, columns=technical_attrs)

def create_competitor_analysis():
    competitors = ['Nova', 'NVIDIA', 'AMD', 'Intel']
    metrics = ['High Processing Speed', 'Low Power Consumption', 'Affordable Price', 'Durability', 'Quiet Operation']
    scores = np.random.randint(6, 10, size=(4, 5))
    return pd.DataFrame(scores, index=competitors, columns=metrics)


# Tabs
st.title("Nova GPU Manufacturing Operations Management Dashboard")

inventory_data = load_or_create_inventory_data()
demand_data = load_or_create_demand_data()
jobs_data = load_or_create_jobs_data()
bom_data = load_or_create_bom_data()

# Sidebar for navigation
with st.sidebar:
    st.image("images/logo.png", width=50)  # Display a small image in the sidebar
    tab = st.radio("Select Section", [
        "About Nova", 
        "Inventory Management", 
        "Demand Forecasting", 
        "Material Requirement Planning",  # MRP moved before Job Scheduling
        "Job Scheduling", 
        "Facility Layout Design", 
        "House of Quality"
    ])

# --- About Nova Tab ---
if tab == "About Nova":
    st.header("About Nova")
    
    # Display the company logo
    st.image("images/logo.png", width=120)

    # Company Description
    st.markdown("""
    ### Company Overview
    **Nova** is a global leader in next-generation GPU manufacturing, founded in 2020 by visionary engineers aiming to revolutionize the computing world.
    Our mission is to deliver high-performance GPUs that power everything from gaming to AI applications.
    We are committed to innovation, quality, and sustainability in our operations.
    
    Through **Project Aurora**, Nova is driving operational excellence by:
    - Optimizing production processes
    - Reducing inventory waste
    - Enhancing alignment with dynamic customer demands

    ### Key Goals
    - ✅ Improve forecast accuracy
    - ✅ Minimize inventory waste (MUDA)
    - ✅ Optimize scheduling and production efficiency
    """)


if tab == "Inventory Management":
    st.header("Inventory Management")
    st.subheader("Inventory Data")
    st.dataframe(inventory_data)

elif tab == "Demand Forecasting":
    st.header("Demand Forecasting")
    st.subheader("Simulated Demand")
    st.dataframe(demand_data)

    # Extend demand_data to include only one future month
    last_month = demand_data['Month'].iloc[-1]
    future_month = pd.date_range(start=last_month + pd.offsets.MonthEnd(1), periods=1, freq='M')
    future_data = pd.DataFrame({'Month': future_month, 'Demand': [np.nan]})
    demand_data = pd.concat([demand_data, future_data], ignore_index=True)

    method = st.selectbox("Choose Forecasting Method", ["Moving Average", "Exponential Smoothing"])

    if method == "Moving Average":
        window = st.slider("Select Moving Average Window", 2, 6, 3)
        demand_data['Forecast'] = moving_average_forecast_with_future(demand_data['Demand'], window)
        st.latex(r'Forecast_t = \frac{Demand_{t-1} + Demand_{t-2} + ...}{Window}')
    else:
        alpha = st.slider("Select Smoothing Constant Alpha", 0.1, 1.0, 0.5)
        demand_data['Forecast'] = exponential_smoothing_forecast(demand_data['Demand'], alpha)
        st.latex(r'Forecast_t = \alpha \times Demand_{t-1} + (1-\alpha) \times Forecast_{t-1}')

    # Calculate RMSE and MAPE for the last 6 months (July to December)
    last_6_months = demand_data[(demand_data['Month'] >= '2025-07-01') & (demand_data['Month'] <= '2025-12-31')]
    actual = last_6_months['Demand'].dropna()
    forecast = last_6_months['Forecast'].iloc[:len(actual)]
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))

    # Plot the forecast graph
    fig = px.line(demand_data, x='Month', y=['Demand', 'Forecast'], markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Display metrics below the graph
    st.subheader("Forecast Accuracy Metrics (Last 6 Months)")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # Save Forecast for Next Month Only
    if st.button("Save Next Month Forecast"):
        next_month_forecast = demand_data.iloc[-1:][['Month', 'Forecast']]
        forecast_filepath = "data/next_month_forecast.csv"
        next_month_forecast.to_csv(forecast_filepath, index=False)
        st.success(f"Next month's forecast saved to {forecast_filepath}")

# --- MRP Tab ---
elif tab == "Material Requirement Planning":
    st.header("Material Requirement Planning (MRP)")

    forecast_path = "data/next_month_forecast.csv"
    if not os.path.exists(forecast_path):
        st.error("No forecast found! Please first generate and save the next month's forecast under Demand Forecasting.")
        st.stop()

    forecast_data = pd.read_csv(forecast_path, parse_dates=['Month'])

    # Debugging: Print the forecast dataframe
    st.subheader("Debug: Forecast Dataframe")
    st.dataframe(forecast_data)

    st.subheader("Forecasted Demand for Next Month")
    st.dataframe(forecast_data)

    st.subheader("Bill of Materials (BOM)")
    st.dataframe(bom_data)

    # Use the single saved forecast for MRP planning
    forecast_qty = forecast_data['Forecast'].values[0]
    selected_month = forecast_data['Month'].iloc[0]

    # Handle NaN in forecast_qty by using the latest demand
    if pd.isna(forecast_qty):
        st.warning(f"Forecast for {selected_month.strftime('%B %Y')} is missing. Using the latest demand instead.")
        latest_demand_data = load_or_create_demand_data()
        forecast_qty = latest_demand_data.loc[latest_demand_data['Month'] == selected_month, 'Demand'].values[0]

    st.write(f"**Forecasted GPUs to produce in {selected_month.strftime('%B %Y')}:** {int(forecast_qty)}")

    # MRP Explosion
    mrp = bom_data.copy()
    mrp['Gross Requirements'] = mrp['Quantity per GPU'] * forecast_qty

    inventory_data = load_or_create_inventory_data()
    inventory_map = inventory_data.set_index('Component').to_dict('index')

    mrp['Available Inventory'] = mrp['Component'].map(lambda x: inventory_map.get(x, {}).get('Initial Inventory', 0))
    mrp['Scheduled Receipts'] = mrp['Component'].map(lambda x: inventory_map.get(x, {}).get('Scheduled Receipts', 0))
    mrp['Safety Stock'] = mrp['Component'].map(lambda x: inventory_map.get(x, {}).get('Safety Stock', 0))

    # Adjust Net Requirements to include Safety Stock
    mrp['Net Requirements'] = (mrp['Gross Requirements'] - mrp['Available Inventory'] - mrp['Scheduled Receipts'] + mrp['Safety Stock']).clip(lower=0)

    st.subheader("MRP Calculation")
    st.latex(r'Net\ Requirements = Gross\ Requirements - Scheduled\ Receipts - Available\ Inventory + Safety\ Stock')

    st.dataframe(mrp[['Component', 'Gross Requirements', 'Available Inventory', 'Scheduled Receipts', 'Safety Stock', 'Net Requirements']])

    if st.button("Save MRP Plan"):
        mrp[['Component', 'Gross Requirements', 'Available Inventory', 'Scheduled Receipts', 'Safety Stock', 'Net Requirements']].to_csv("data/mrp_data.csv", index=False)
        st.success("MRP Plan saved!")

elif tab == "Job Scheduling":
    st.header("Job Scheduling")
    st.subheader("Jobs Data")
    st.dataframe(jobs_data)

    st.subheader("Sequencing Rule")
    rule = st.selectbox("Choose Scheduling Rule", ["FCFS", "SPT", "EDD"])

    if rule == "FCFS":
        st.write("**First Come First Serve (FCFS):** Jobs are scheduled in the order they arrive.")
        sequence = jobs_data['Job'].tolist()
    elif rule == "SPT":
        st.write("**Shortest Processing Time (SPT):** Jobs with shortest processing times are scheduled first.")
        sequence = jobs_data.sort_values(by='Processing Time (days)')['Job'].tolist()
    elif rule == "EDD":
        st.write("**Earliest Due Date (EDD):** Jobs with earliest due dates are scheduled first.")
        sequence = jobs_data.sort_values(by='Due Date (days from today)')['Job'].tolist()

    st.write(f"**Job Sequence:** {' -> '.join(sequence)}")
    st.plotly_chart(create_gantt_chart(jobs_data, sequence), use_container_width=True)

elif tab == "Facility Layout Design":
    st.header("Nova Facility Layout Design")
    stages, flows, distances, total_cost = calculate_facility_layout()
    st.subheader("From-To Flow and Distance Matrix")
    df_layout = pd.DataFrame({
        'From': stages[:-1],
        'To': stages[1:],
        'Flow Volume': flows,
        'Distance (m)': distances,
        'Flow x Distance': [f*d for f,d in zip(flows,distances)]
    })
    st.dataframe(df_layout)
    st.subheader(f"Total Material Handling Cost: **{total_cost} units-meters**")

    st.subheader("Facility Block Layout")
    fig, ax = plt.subplots(figsize=(14, 2))
    for i, stage in enumerate(stages):
        ax.add_patch(plt.Rectangle((i*1.5, 0), 1.5, 1, fill=True, edgecolor='black', facecolor='lightblue'))
        ax.text(i*1.5 + 0.75, 0.5, stage, ha='center', va='center', fontsize=8)
    plt.axis('off')
    st.pyplot(fig)

elif tab == "House of Quality":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from fpdf import FPDF
    import base64

    st.header("Nova House of Quality (QFD Matrix)")

    # Step 1: Data Setup
    def create_hoq_data():
        customer_reqs = ['High Processing Speed', 'Low Power Consumption', 'Affordable Price', 'Durability', 'Quiet Operation']
        technical_attrs = ['GPU Chip Quality', 'Cooling Efficiency', 'PCB Material', 'Thermal Paste Quality', 'Memory Type']
        relationship_matrix = [
            [9, 3, 0, 9, 0],
            [3, 9, 0, 3, 0],
            [0, 0, 9, 0, 9],
            [3, 3, 3, 9, 0],
            [0, 3, 0, 0, 9]
        ]
        return pd.DataFrame(relationship_matrix, index=customer_reqs, columns=technical_attrs)

    def create_competitor_analysis():
        competitors = ['Nova', 'NVIDIA', 'AMD', 'Intel']
        metrics = ['High Processing Speed', 'Low Power Consumption', 'Affordable Price', 'Durability', 'Quiet Operation']
        scores = np.random.randint(6, 10, size=(4, 5))
        return pd.DataFrame(scores, index=competitors, columns=metrics)

    def create_roof_matrix():
        roof = np.array([
            [ 0,  1,  0,  1,  0],
            [ 1,  0,  0, -1,  0],
            [ 0,  0,  0,  0,  1],
            [ 1, -1,  0,  0,  0],
            [ 0,  0,  1,  0,  0]
        ])
        return pd.DataFrame(roof, columns=technical_attrs, index=technical_attrs)

    hoq_df = create_hoq_data()
    comp_df = create_competitor_analysis()
    technical_attrs = hoq_df.columns.tolist()
    customer_reqs = hoq_df.index.tolist()
    roof_df = create_roof_matrix()

    # Step 2: Visuals
    st.subheader("1️⃣ Customer Requirements vs Technical Requirements (Relationship Matrix)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.heatmap(hoq_df, annot=True, cmap="YlGnBu", cbar=False, linewidths=1, linecolor='gray', ax=ax1)
    ax1.set_title("Relationship Matrix", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig1)

    st.subheader("2️⃣ Technical Correlation Matrix (Roof)")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(roof_df, annot=True, cmap="coolwarm", center=0, cbar=False, linewidths=0.5, linecolor='gray', ax=ax2)
    ax2.set_title("Roof Matrix (Correlation between Technical Requirements)", fontsize=12)
    st.pyplot(fig2)

    st.subheader("3️⃣ Competitor Analysis")
    st.dataframe(comp_df.style.highlight_max(axis=0, color='darkgreen'))
    fig3 = comp_df.T.plot(kind="bar", figsize=(10, 4), title="Competitor Comparison").get_figure()
    st.pyplot(fig3)

    st.subheader("4️⃣ Final House of Quality Layout")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.axis('off')

    # Draw relationship matrix
    matrix_ax = fig4.add_axes([0.25, 0.1, 0.6, 0.6])
    sns.heatmap(hoq_df, annot=True, cmap="YlGnBu", cbar=False, ax=matrix_ax, linewidths=1, linecolor='black')
    matrix_ax.set_xticklabels(technical_attrs, rotation=45, ha='right')
    matrix_ax.set_yticklabels(customer_reqs, rotation=0)
    matrix_ax.set_title("Final House of Quality")

    # Draw roof
    roof_ax = fig4.add_axes([0.25, 0.7, 0.6, 0.2])
    sns.heatmap(roof_df, annot=True, cmap="coolwarm", center=0, cbar=False, ax=roof_ax, linewidths=1, linecolor='black')
    roof_ax.set_xticks([])
    roof_ax.set_yticks([])

    st.pyplot(fig4)

    hoq_img_path = "data/hoq_final.png"
    fig4.savefig(hoq_img_path, bbox_inches="tight")

    # Step 3: PDF generation
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Nova - House of Quality Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Customer vs Technical Requirements", ln=True)
        pdf.image(hoq_img_path, w=180)
        pdf.ln(10)

        pdf.cell(0, 10, "Competitor Ratings (Table)", ln=True)
        pdf.set_font("Arial", "", 10)
        for idx, row in comp_df.iterrows():
            pdf.cell(0, 8, f"{idx}: " + ", ".join([f"{col}: {val}" for col, val in row.items()]), ln=True)

        pdf.output("data/HOQ_Report_Nova.pdf")
        return "data/HOQ_Report_Nova.pdf"

    if st.button("📄 Download Final HOQ Report as PDF"):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="HOQ_Report_Nova.pdf">Click here to download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success("PDF ready!")

st.sidebar.info("MIOM 2025 - Nova GPU Manufacturing Operations Management Dashboard")
