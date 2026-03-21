class ThreatEstimator:
    def __init__(self, action_threats: dict):
        self.action_threats = action_threats
    
    def estimate(self, dist, size, action):
        dist_score =  1 / (dist + 1)
        size_score = min(size /10, 1)
        action_score = self.action_threats.get(action, 0.3)

        threat = (
            0.5 * dist_score +
            0.3 * action_score +
            0.2 * size_score
        )

        return round(threat, 2)