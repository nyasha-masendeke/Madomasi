RECOMMENDATIONS = {
    "Tomato___healthy": "**Great news!** Your tomato plant appears healthy. Continue regular care and monitor weekly.",
    "Tomato___Early_blight": "**Treatment:** Remove lower infected leaves. Apply copper-based fungicide. Mulch around the base to prevent soil splash.",
    "Tomato___Late_blight": "**Treatment:** Remove and destroy infected foliage immediately. Apply systemic fungicide. Improve air circulation and avoid overhead watering.",
    "Tomato___Bacterial_spot": "**Treatment:** Apply copper sprays preventatively. Avoid working with wet plants. Remove severely infected leaves.",
    "Tomato___Leaf_Mold": "**Treatment:** Improve ventilation in greenhouse. Reduce humidity. Apply appropriate fungicide.",
    "Tomato___Septoria_leaf_spot": "**Treatment:** Remove infected leaves. Apply fungicide at first sign. Avoid overhead irrigation.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "**Treatment:** Spray with water to dislodge mites. Apply miticide or neem oil. Increase humidity around plants.",
    "Tomato___Target_Spot": "**Treatment:** Apply fungicide. Remove infected plant debris. Ensure good air circulation.",
    "Tomato___Tomato_mosaic_virus": "**Treatment:** No cure — remove and destroy infected plants. Disinfect tools. Control aphid vectors.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "**Treatment:** No cure — remove infected plants. Control whitefly populations with insecticide. Use resistant varieties.",
}


def get_recommendation(disease: str) -> str:
    return RECOMMENDATIONS.get(disease, "Monitor plant regularly and consult a local agricultural expert.")
