RECOMMENDATIONS = {
    "Healthy": "**Great news!** Your tomato plant appears healthy. Continue regular care.",
    "Early Blight": "**Treatment:** Remove lower leaves, apply copper fungicide, mulch soil.",
    "Late Blight": "**Treatment:** Destroy infected foliage immediately, improve air flow.",
    "Bacterial Spot": "**Treatment:** Use copper sprays preventatively, avoid wet leaves."
}

def get_recommendation(disease: str) -> str:
    return RECOMMENDATIONS.get(disease, "Monitor plant regularly and consult local experts.")
