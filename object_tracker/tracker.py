import numpy as np
from scipy.optimize import linear_sum_assignment


class ObjectTracker:
    """Tracks objects across video frames using the SORT algorithm."""
    
    def __init__(self, max_age=30, min_hits=3):
        """
        Initialize the ObjectTracker with SORT (Simple Online and Realtime Tracking).
        
        SORT works by:
        1. Predicting where each tracked object should be in the current frame
        2. Matching detections to tracked objects based on IoU overlap
        3. Assigning track IDs to maintain object identity across frames
        4. Removing tracks that haven't been seen recently
        
        Args:
            max_age (int): Number of frames to keep a track alive without detections
            min_hits (int): Number of detections needed to confirm a track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []  # List of active tracks
        self.next_track_id = 1  # Counter for assigning new track IDs
    
    # Match detections to existing tracks using IoU-based cost
    def _match_detections_to_tracks(self, detections):
        """
        Match detected objects to existing tracks using the Hungarian algorithm.
        
        This creates a cost matrix where each cell represents the IoU-based cost
        between a detection and a track box. The Hungarian algorithm finds the
        optimal assignment that minimizes total cost.
        
        Args:
            detections (list): List of detection dicts with bounding boxes
        
        Returns:
            tuple: (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Create cost matrix: IoU-based distance between each detection and track
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, detection in enumerate(detections):
            for t, track in enumerate(self.tracks):
                # Calculate Intersection over Union (IoU) - better metric than distance
                # IoU measures how much two boxes overlap
                
                # Get box coordinates
                det_x1, det_y1, det_x2, det_y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                track_x1, track_y1, track_x2, track_y2 = track['x1'], track['y1'], track['x2'], track['y2']
                
                # Calculate intersection area
                inter_x1 = max(det_x1, track_x1)
                inter_y1 = max(det_y1, track_y1)
                inter_x2 = min(det_x2, track_x2)
                inter_y2 = min(det_y2, track_y2)
                
                if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                    # No intersection
                    iou = 0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    
                    # Calculate union area
                    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                    track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                    union_area = det_area + track_area - inter_area
                    
                    # Calculate IoU
                    iou = inter_area / union_area if union_area > 0 else 0
                
                # Convert IoU to cost (lower cost = better match)
                # Cost ranges from 0 (perfect match) to infinity (no overlap)
                cost = 1 - iou if iou > 0 else 1000
                cost_matrix[d, t] = cost
        
        # Use Hungarian algorithm to find optimal assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches: only keep pairs with good IoU (cost < 0.5 means IoU > 0.5)
        matched_pairs = []
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            # Accept matches with IoU > 0.4 (cost < 0.6)
            if cost_matrix[det_idx, track_idx] < 0.6:
                matched_pairs.append((det_idx, track_idx))
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) 
                               if i not in matched_detections]
        unmatched_tracks = [i for i in range(len(self.tracks)) 
                           if i not in matched_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    # Update tracker with new detections from the current frame
    def update(self, detections):
        """
        Update tracks with new detections and return tracked objects.
        
        This method:
        1. Matches detections to existing tracks
        2. Updates matched tracks with new detection data
        3. Creates new tracks for unmatched detections
        4. Removes old tracks that haven't been seen recently
        5. Returns all active tracks with track IDs
        
        Args:
            detections (list): List of detection dicts from ObjectDetector
        
        Returns:
            list: List of tracked objects, each with keys:
                  x1, y1, x2, y2, track_id, class_name
        """
        # Handle empty detections gracefully
        if len(detections) == 0:
            detections = []
        
        # Match detections to existing tracks
        matched_pairs, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks with new detection data
        for det_idx, track_idx in matched_pairs:
            detection = detections[det_idx]
            self.tracks[track_idx]['x1'] = detection['x1']
            self.tracks[track_idx]['y1'] = detection['y1']
            self.tracks[track_idx]['x2'] = detection['x2']
            self.tracks[track_idx]['y2'] = detection['y2']
            self.tracks[track_idx]['class_name'] = detection['class_name']
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['age'] = 0  # Reset age when matched
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            new_track = {
                'x1': detection['x1'],
                'y1': detection['y1'],
                'x2': detection['x2'],
                'y2': detection['y2'],
                'track_id': self.next_track_id,
                'class_name': detection['class_name'],
                'hits': 1,
                'age': 0
            }
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Increment age for unmatched tracks (tracks without new detections)
        for track_idx in unmatched_trks:
            self.tracks[track_idx]['age'] += 1
        
        # Remove old tracks that haven't been matched for too long
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        # Return only confirmed tracks (those with enough hits)
        confirmed_tracks = [t for t in self.tracks if t['hits'] >= self.min_hits]
        
        # Return tracks with required fields only
        result = []
        for track in confirmed_tracks:
            result.append({
                'x1': track['x1'],
                'y1': track['y1'],
                'x2': track['x2'],
                'y2': track['y2'],
                'track_id': track['track_id'],
                'class_name': track['class_name']
            })
        
        return result
