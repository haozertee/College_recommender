import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="College Recommender",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("CollegeData2024.csv")
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
    df.columns = df.columns.str.strip().str.lower()

    df['tuition'] = (
        df['tuition']
        .astype(str)
        .str.replace(r'[\$,]', '', regex=True)
        .replace('nan', pd.NA)
    )
    df['tuition'] = pd.to_numeric(df['tuition'], errors='coerce')

    df['tuition_formatted'] = df['tuition'].apply(
        lambda x: f"${int(x):,}" if pd.notnull(x) and isinstance(x, (int, float)) else "Not Reported"
    )

    df['sat low'] = pd.to_numeric(df['sat low'], errors='coerce')
    df['sat high'] = pd.to_numeric(df['sat high'], errors='coerce')
    df['undergrad pop'] = pd.to_numeric(df['undergrad pop'], errors='coerce').fillna(0).astype(int)

    return df

def parse_date_safe(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        return None

def main():
    st.title("🎓 College Recommender App")

    df = load_data()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Filters")

        region_options = ["All"] + sorted(df['region'].dropna().unique())
        region = st.selectbox("📍 U.S. Region", region_options)

        sat_score = st.number_input("✏️ SAT Score (400–1600)", min_value=400, max_value=1600, value=1200)

        num_colleges = st.selectbox("📊 Number of Colleges to Show", [5, 10, 15], index=0)

        if region == "All":
            filtered = df[df['sat low'] <= sat_score]
        else:
            filtered = df[(df['region'] == region) & (df['sat low'] <= sat_score)]

        filtered = filtered.sort_values(by='usn_rank').head(num_colleges)

    with col2:
        if not filtered.empty:
            st.subheader(f"🏫 Top {num_colleges} Matching Colleges")

            college_names = filtered['college'].tolist()
            selected_college = st.selectbox("Choose a college", college_names)

            if selected_college:
                st.markdown(f"### 📋 More detailed admission info for **{selected_college}**")
                college_info = df[df['college'] == selected_college].iloc[0]

                st.markdown(f"**Undergraduate Population:** {int(college_info['undergrad pop']):,}")

                state = college_info['state'] if pd.notna(college_info['state']) else "No Info"
                st.markdown(f"📍 This college is located at **{state}**.")

                sat_act_req = college_info.get('sat act req', None)
                if pd.isna(sat_act_req) or str(sat_act_req).strip() == "":
                    sat_act_req = "No Info"
                st.markdown(f"🧪 Status of SAT or ACT: **{sat_act_req}**")

                app_deadline = college_info['app deadline'] if pd.notna(college_info['app deadline']) else "No Info"
                sat_act_deadline = college_info['sat act deadline'] if pd.notna(college_info['sat act deadline']) else "No Info"
                ea_deadline = college_info['ea deadline'] if pd.notna(college_info['ea deadline']) else "No Info"
                ed_ea_type = college_info['ed or ea'] if pd.notna(college_info['ed or ea']) else "No Info"

                st.markdown(
                    f"📅 **Application Deadline:** {app_deadline} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**SAT/ACT Deadline:** {sat_act_deadline} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**EA Deadline:** {ea_deadline} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**ED/EA Type:** {ed_ea_type}",
                    unsafe_allow_html=True
                )

            # ✅ Display table without index
            st.dataframe(
                filtered[[
                    'college', 'usn_rank', 'sat low', 'sat high', 'acceptance',
                    'tuition_formatted', 'app deadline'
                ]].rename(columns={
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

            # Timeline data
            timeline_rows = []
            for _, row in filtered.iterrows():
                college = row["college"]
                deadlines = {
                    "Application Deadline": row["app deadline"],
                    "SAT/ACT Deadline": row["sat act deadline"],
                    "EA Deadline": row["ea deadline"],
                }
                for label, date_str in deadlines.items():
                    date = parse_date_safe(date_str)
                    if pd.notnull(date):
                        timeline_rows.append({
                            "College + Deadline": f"{college} – {label}",
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
                    title=f"📊 Admission Deadlines Comparison ({num_colleges} Colleges)"
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
        else:
            st.warning("⚠️ No colleges found. Try a different SAT score or region.")

if __name__ == "__main__":
    main()
