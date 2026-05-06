import streamlit as st
import psutil
import platform
from config import DISEASE_CLASSES

def get_sys_stats():
    """Get current system statistics."""
    stats = {
        'cpu': psutil.cpu_percent(interval=1),
        'ram': psutil.virtual_memory().percent
    }
    
    # Try to get CPU temperature
    try:
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if 'cpu' in name.lower() or 'core' in entry.label.lower():
                            stats['temp'] = round(entry.current, 1)
                            break
    except Exception:
        pass
    
    return stats

def render_sidebar() -> dict:
    """Render sidebar controls and return selected parameters."""
    with st.sidebar:
        st.header("🔍 Detection Settings", divider=True)
        
        # Confidence Threshold
        st.subheader("📊 Confidence Threshold")
        confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Minimum probability required to report a disease detection"
        )
        st.write(f"**Current:** {confidence:.1f}%")
        st.divider()
        
        # Disease Classes Selection
        st.subheader("🦠 Disease Classes")
        selected_classes = st.multiselect(
            "Choose diseases to detect",
            options=DISEASE_CLASSES,
            default=DISEASE_CLASSES,
            help="Select which disease classes to actively detect"
        )

        if selected_classes:
            st.write("**Active Filters:**")
            for disease in selected_classes:
                cls = "healthy" if "healthy" in disease.lower() else "diseased"
                st.markdown(
                    f'<span class="disease-tag {cls}" style="'
                    f'background: {"#4CAF50" if "healthy" in disease.lower() else "#FF6B6B"}; '
                    f'color: white; padding: 4px 12px; border-radius: 12px; '
                    f'display: inline-block; margin: 2px; font-size: 0.85em;">'
                    f'✓ {disease}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ No diseases selected!")
        
        st.divider()
        
        # ROI Configuration
        st.subheader("🎯 Region of Interest (ROI)")
        st.markdown("*Focus analysis on specific leaf area*")
        
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.slider(
                "X Min (Left)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
            y_min = st.slider(
                "Y Min (Top)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
        with col2:
            x_max = st.slider(
                "X Max (Right)",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.01
            )
            y_max = st.slider(
                "Y Max (Bottom)",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.01
            )
        
        # ROI Validation
        if x_min >= x_max or y_min >= y_max:
            st.error("⚠️ Invalid ROI: Min values must be less than Max values!")
        
        # Calculate ROI area
        roi_area = (x_max - x_min) * (y_max - y_min) * 100
        st.caption(f"ROI Coverage: {roi_area:.1f}%")
        
        st.divider()
        
        # System Health Monitor
        st.subheader("💻 System Health")
        with st.expander("📊 Resource Monitor", expanded=True):
            stats = get_sys_stats()
            
            # CPU Usage
            st.caption("CPU Usage")
            cpu_progress = stats['cpu'] / 100
            st.progress(cpu_progress)
            st.write(f"**{stats['cpu']:.1f}%**")
            
            # RAM Usage
            st.caption("RAM Usage")
            ram_progress = stats['ram'] / 100
            st.progress(ram_progress)
            st.write(f"**{stats['ram']:.1f}%**")
            
            # Warnings
            if stats['ram'] > 90:
                st.error("⚠️ Critical: High Memory Usage!")
            elif stats['ram'] > 75:
                st.warning("⚠️ Warning: Memory usage above 75%")
            
            # Temperature (if available)
            if stats.get('temp'):
                st.divider()
                temp = stats['temp']
                st.write(f"🌡️ **Temperature:** {temp}°C")
                if temp > 80:
                    st.error("⚠️ High CPU temperature!")
            
            # System Info
            st.divider()
            st.caption("System Info")
            st.write(f"**OS:** {platform.system()}")
            st.write(f"**CPU Count:** {psutil.cpu_count()} cores")
        
        st.divider()
        
        # Quick Stats Summary
        st.subheader("📈 Quick Stats")
        st.metric(
            label="Active Filters",
            value=len(selected_classes),
            delta=f"{len(selected_classes)}/{len(DISEASE_CLASSES)}"
        )
        
        st.metric(
            label="ROI Coverage",
            value=f"{roi_area:.1f}%",
            delta="Area analyzed" if roi_area > 50 else "Small area"
        )
    
    # Return parameters as dictionary
    return {
        "confidence": confidence,
        "selected_classes": selected_classes,
        "roi": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        }
    }
