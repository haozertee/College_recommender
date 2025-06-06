import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import pydeck as pdk


# =====================
# Styling and Background
# =====================
def set_background(image_path="background.png"):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <style>
        html, body {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        header, section[data-testid="stHeader"] {{
            display: none !important;
        }}

        .block-container {{
            padding-top: 0rem !important;
            margin-top: -2rem !important;
        }}

        .card {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)


# =====================
# Load and Prepare Data
# =====================
@st.cache_data
def load_data(file_path="CollegeData2024.csv"):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
    df.columns = df.columns.str.strip().str.lower()

    df['tuition'] = df['tuition'].astype(str).str.replace(r'[\$,]', '', regex=True).replace('nan', pd.NA)
    df['tuition'] = pd.to_numeric(df['tuition'], errors='coerce')
    df['tuition_formatted'] = df['tuition'].apply(
        lambda x: f"${int(x):,}" if pd.notnull(x) and isinstance(x, (int, float)) else "Not Reported")

    df['sat low'] = pd.to_numeric(df['sat low'], errors='coerce')
    df['sat high'] = pd.to_numeric(df['sat high'], errors='coerce')
    df['undergrad pop'] = pd.to_numeric(df['undergrad pop'], errors='coerce').fillna(0).astype(int)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    return df


def parse_date_safe(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None
    return pd.to_datetime(value, errors='coerce')


# =====================
# Main App
# =====================
def main():
    st.set_page_config(page_title="College Recommender", page_icon="🎓", layout="wide")
    set_background()

    df = load_data()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title("🎓 AI Guidance Counselor")
        st.markdown("Select your preferences to discover colleges recommendation choices.")

        st.subheader("🔍 Your Preferences")
        region_options = ["All"] + sorted(df['region'].dropna().unique())
        region = st.selectbox("📍 U.S. Region", region_options)
        sat_score = st.number_input("✏️ SAT Score (400–1600)", min_value=400, max_value=1600, value=1400)
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

        filtered = df[df['sat low'] <= sat_score] if region == "All" \
            else df[(df['region'] == region) & (df['sat low'] <= sat_score)]

        filtered = filtered[(filtered['tuition'] >= tuition_range[0]) & (filtered['tuition'] <= tuition_range[1])]
        filtered = filtered.sort_values(by='usn_rank').head(num_colleges)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if filtered.empty:
            st.warning("⚠️ No colleges found. Try a different SAT score or region.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        st.subheader(f"🏫 Top {num_colleges} Matching Colleges")
        st.dataframe(
            filtered[[
                'college', 'usn_rank', 'sat low', 'sat high', 'acceptance',
                'tuition_formatted', 'app deadline']].rename(columns={
                'college': 'College',
                'usn_rank': 'USN Rank',
                'sat low': 'SAT 25th Percentile',
                'sat high': 'SAT 75th Percentile',
                'acceptance': 'Acceptance Rate',
                'tuition_formatted': 'Tuition',
                'app deadline': 'Application Deadline'
            }),
            use_container_width=True,
            hide_index=True
        )

        college_names = filtered['college'].tolist()
        selected_college = st.selectbox("Choose a college", college_names)

        if selected_college:
            college_info = df[df['college'] == selected_college].iloc[0]
            st.markdown(f"### 📋 More detailed admission info for **{selected_college}**")
            st.markdown(f"**Undergraduate Population:** {int(college_info['undergrad pop']):,}")
            st.markdown(f"📍 Located in: **{college_info.get('state', 'No Info')}**")
            st.markdown(f"🧮 SAT/ACT Requirement: **{college_info.get('sat act req', 'No Info')}**")

            st.markdown(
                f"🗓️ **Application Deadline:** {college_info.get('app deadline', 'No Info')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**SAT/ACT Deadline:** {college_info.get('sat act deadline', 'No Info')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**EA Deadline:** {college_info.get('ea deadline', 'No Info')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**ED/EA Type:** {college_info.get('ed or ea', 'No Info')}",
                unsafe_allow_html=True
            )

            selected_row = filtered[filtered['college'] == selected_college]
            if not selected_row.empty:
                st.subheader("📍 Selected College on Map")
                map_df = selected_row[['latitude', 'longitude', 'college']].dropna()
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[longitude, latitude]',
                    get_radius=8000,
                    get_fill_color='[255, 0, 0, 160]',
                    pickable=True
                )
                view_state = pdk.ViewState(
                    latitude=map_df['latitude'].iloc[0],
                    longitude=map_df['longitude'].iloc[0],
                    zoom=6,
                    pitch=0
                )
                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{college}"})
                st.pydeck_chart(r)

        # Timeline chart
        timeline_rows = []
        for _, row in filtered.iterrows():
            for label, date_str in {
                "Application Deadline": row["app deadline"],
                "SAT/ACT Deadline": row["sat act deadline"],
                "EA Deadline": row["ea deadline"]
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
                title=f"📊 Important Admission Dates for the selected ({num_colleges} Colleges)"
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="",
                height=600,
                margin=dict(l=40, r=40, t=60, b=20),
                legend_title_text="Deadline Type",
                xaxis=dict(
                    type="date",
                    range=[
                        timeline_df["Start"].min() - pd.Timedelta(days=5),
                        timeline_df["End"].max() + pd.Timedelta(days=5)
                    ]
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid admission date information available for timeline.")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
