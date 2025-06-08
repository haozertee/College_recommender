import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("CollegeData2024.csv")
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    # Convert numeric columns
    df['tuition'] = pd.to_numeric(df['tuition'], errors='coerce')
    df['undergrad pop'] = pd.to_numeric(df['undergrad pop'], errors='coerce')
    for col in ['sat low', 'sat high', 'act low', 'act high', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Format tuition
    df['tuition_formatted'] = df['tuition'].apply(
        lambda x: f"${int(x):,}" if pd.notnull(x) else "N/A"
    )
    return df

def parse_date_safe(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return pd.NaT

def set_background():
    st.markdown(
        """
        <style>
        .banner {
            margin-top: 2rem;
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            background-color: #ADD8E6;
            color: #000;
            border-radius: 0.5rem;
        }
        /* make selectbox labels the same as your subheader */
        .stSelectbox > label {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="College Recommender", page_icon="🎓", layout="wide")
    set_background()
    df = load_data()

    st.markdown('<div class="banner">🎓 AI Guidance Counselor</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["College Search", "Dates"])

    with tab1:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("🔍 Select your preferences")
            region_options = ["All"] + sorted(df['region'].dropna().unique())
            region = st.selectbox("📍 U.S. Region", region_options)

            use_sat = st.checkbox("Use SAT Score (400–1600). Uncheck to use ACT (1–36)", value=True)
            if use_sat:
                score = st.number_input("✏️ SAT Score", min_value=400, max_value=1600, value=1200, step=10)
            else:
                score = st.number_input("🧠 ACT Score", min_value=1, max_value=36, value=25, step=1)

            tuition_min = int(df['tuition'].min(skipna=True))
            tuition_max = int(df['tuition'].max(skipna=True))
            tuition_range = st.slider(
                "💰 Tuition Range ($)",
                min_value=tuition_min,
                max_value=tuition_max,
                value=(tuition_min, tuition_max),
                step=1000
            )

            num_colleges = st.selectbox("📊 Number of Colleges to Show", [5, 10, 15], index=0)

            # Apply filters
            if region == "All":
                if use_sat:
                    filtered = df[df['sat low'] <= score]
                else:
                    filtered = df[df['act low'] <= score]
            else:
                if use_sat:
                    filtered = df[(df['sat low'] <= score) & (df['region'] == region)]
                else:
                    filtered = df[(df['act low'] <= score) & (df['region'] == region)]

            filtered = filtered[
                (filtered['tuition'] >= tuition_range[0]) &
                (filtered['tuition'] <= tuition_range[1])
            ]
            filtered = filtered.sort_values(by='usn_rank').head(num_colleges)

            st.subheader(f"🏫 Top {num_colleges} Matching Colleges")
            if filtered.empty:
                st.warning("⚠️ No colleges found. Try adjusting your filters.")
            else:
                for _, row in filtered.iterrows():
                    st.markdown(f"- **{row['college']}** (USN Rank: {row['usn_rank']})")

        with col_right:
            if not filtered.empty:
                selected_college = st.selectbox(
                    "🏫 Select a college for more details",
                    filtered['college'].tolist()
                )
                college_info = df[df['college'] == selected_college].iloc[0]

                st.markdown(f"### 📋 Details for **{selected_college}**")
                st.markdown(f"**Undergraduate Population:** {int(college_info['undergrad pop']):,}")
                st.markdown(f"📍 **{college_info.get('state', 'No Info')}**")
                st.markdown(f"🧪 SAT/ACT Requirement: {college_info.get('sat act req', 'No Info')}")
                st.markdown(f"**SAT 25th:** {college_info.get('sat low', 'No Info')}")
                st.markdown(f"**SAT 75th:** {college_info.get('sat high', 'No Info')}")
                st.markdown(f"**ACT 25th:** {college_info.get('act low', 'No Info')}")
                st.markdown(f"**ACT 75th:** {college_info.get('act high', 'No Info')}")
                st.markdown(f"**Acceptance Rate:** {college_info.get('acceptance', 'No Info')}")
                st.markdown(f"**Tuition:** {college_info.get('tuition_formatted', 'No Info')}")
                st.markdown(f"**Application Deadline:** {college_info.get('app deadline', 'No Info')}")

                # Show map if coordinates exist
                if pd.notnull(college_info.get('latitude')) and pd.notnull(college_info.get('longitude')):
                    map_df = pd.DataFrame([{
                        'latitude': college_info['latitude'],
                        'longitude': college_info['longitude']
                    }])
                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=map_df,
                        get_position='[longitude, latitude]',
                        get_radius=8000,
                        get_fill_color='[255, 0, 0, 160]',
                        pickable=True
                    )
                    view_state = pdk.ViewState(
                        latitude=college_info['latitude'],
                        longitude=college_info['longitude'],
                        zoom=6,
                        pitch=0
                    )
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    with tab2:
        st.subheader("📅 Application Deadlines Timeline")
        if not filtered.empty:
            timeline_rows = []
            for _, row in filtered.iterrows():
                for label, date_str in {
                    "App Deadline": row.get("app deadline"),
                    "SAT/ACT Deadline": row.get("sat act deadline"),
                    "EA Deadline": row.get("ea deadline")
                }.items():
                    date = parse_date_safe(date_str)
                    if pd.notnull(date):
                        timeline_rows.append({
                            "College + Deadline": f"{row['college']} – {label}",
                            "Start": date,
                            "End": date + pd.Timedelta(days=1),
                            "Type": label
                        })
            timeline_df = pd.DataFrame(timeline_rows)
            if not timeline_df.empty:
                fig = px.timeline(
                    timeline_df,
                    x_start="Start",
                    x_end="End",
                    y="College + Deadline",
                    color="Type",
                    title="Important Admission Dates"
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(xaxis_title="Date", yaxis_title="", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid admission date information available.")

if __name__ == "__main__":
    main()
