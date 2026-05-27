"""Shared Streamlit-layer model cache.

A single @st.cache_resource entry point so every tab (inference, evaluation,
Grad-CAM) shares one loaded model instance per path, avoiding redundant loads.
"""
import streamlit as st
import tensorflow as tf


@st.cache_resource
def load_cached_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
