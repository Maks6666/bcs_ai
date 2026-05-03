import numpy as np



class GroupTracker:
    def __init__(self, prev_groups, next_group_id):
        self.prev_groups = prev_groups
        self.next_group_id = next_group_id
    def track_groups(self, groups):
            
            group_centers = {}

            for label, group in groups.items():

                x_s = []
                y_s = []

                for (x1, y1, x2, y2) in group:
                    x_s.append((x1+x2)//2)
                    y_s.append((y1+y2)//2)
                
                c_x = int(np.mean(x_s))
                c_y = int(np.mean(y_s))

                group_centers[label] = (c_x, c_y)
            
            tracked_groups = {}
            final_groups = {}

            for label, (c_x, c_y) in group_centers.items():

                matched_id = None
                min_dist = 1e9

                for g_id, (p_x, p_y) in self.prev_groups.items():
                    dist = np.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)

                    if dist < min_dist and dist < 150:
                        min_dist = dist
                        matched_id = g_id

                if matched_id is None:
                    matched_id = self.next_group_id
                    self.next_group_id +=1 
                
                tracked_groups[matched_id] = (c_x, c_y)
                final_groups[matched_id] = groups[label]
    
            self.prev_groups = tracked_groups

            return final_groups