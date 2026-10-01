"""
Streamlit Web Dashboard for DriveGuard AI.
Renders webcam stream, real-time metrics cards, fatigue gauges, and Plotly dynamic charts.
"""

import streamlit as st


def main():
    st.set_page_config(
        page_title="DriveGuard AI | Driver Fatigue Monitoring",
        page_icon="🚗",
        layout="wide"
    )

    st.title("DriveGuard AI 🚗💤")
    st.subheader("Real-Time Driver Drowsiness & Fatigue Vision Alert System")

    # Sidebar parameters
    st.sidebar.header("⚙️ System Configuration")
    
    # Placeholder layout sections
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("Live Video Feed Stream Placeholder")

    with col2:
        st.metric("Driver State", "ALERT")
        st.metric("Fatigue Index", "0%")


if __name__ == "__main__":
    main()
