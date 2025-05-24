def inclusion_score(segmentation_map):
    area = (segmentation_map > 0.5).sum().item()
    score = 100 - min(100, area / 1000)
    if score > 90: return "IF"
    elif score > 75: return "VVS"
    elif score > 60: return "VS"
    elif score > 45: return "SI"
    else: return "I"
