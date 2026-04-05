class Priority:
    def __init__(self, threat_scores: dict):
        self.threat_scores = threat_scores
        self.current_priority = None

    def choose_target(self):
        if not self.threat_scores:
            return None

        # new_priority - index of the object with highest score 
        new_priority = max(self.threat_scores, key=self.threat_scores.get)

        if self.current_priority is None:
            # here we assign new value to self.current_priority, which os declared in __init__
            self.current_priority = new_priority
            return new_priority

        # highest score for a current moment
        current_score = self.threat_scores.get(self.current_priority, 0)

        # new highest score 
        new_score = self.threat_scores[new_priority]


        if new_score > current_score + 0.1:
            self.current_priority = new_priority

        return self.current_priority

    def priority_list(self, current_priority):
        if not self.threat_scores: 
            return {}
        
        priority_queue = {}
        priority_queue[1] = current_priority

        other_targets = {
            k: v for k, v in self.threat_scores.items()
            if k != current_priority
        }

        sorted_objects = sorted(other_targets.items(), key=lambda x: x[1], reverse=True)

        for i, (idx, _) in enumerate(sorted_objects[:2], start=2):
            priority_queue[i] = idx
        
        return priority_queue