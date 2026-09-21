# ============================================================
# LITTERING DECISION - SIMPLE & STABLE
# Works with or without dustbin
# ============================================================

import time


class LitteringTracker:
    """
    Simple littering tracker.
    
    Logic:
    - Object in hands (HOLDING) -> OK
    - Object dropped (DROPPED) -> Start 10 second countdown
    - If picked up -> CORRECTED
    - If left for 10+ seconds -> LITTERING
    """
    
    def __init__(self, grace_time=10, near_dustbin_bonus=5):
        self.grace_time = grace_time
        self.near_dustbin_bonus = near_dustbin_bonus
        
        # Track dropped objects
        self.dropped_objects = {}  # id -> {start_time, box, class_name}
        self.next_id = 0
    
    def check_littering(self, garbage_boxes, dustbin_boxes, person_boxes=None, garbage_in_hands=None):
        """
        Check littering status.
        
        Returns:
            results: [(box, status), ...]
            triggered: True if LITTERING confirmed
            message: Status message
        """
        current_time = time.time()
        results = []
        triggered = False
        message = ""
        
        if garbage_in_hands is None:
            garbage_in_hands = [False] * len(garbage_boxes)
        
        seen_drops = set()
        
        for i, box in enumerate(garbage_boxes):
            in_hands = garbage_in_hands[i] if i < len(garbage_in_hands) else False
            
            if self._is_in_dustbin(box, dustbin_boxes):
                results.append((box, "SAFE"))
                message = "In dustbin - SAFE"
                self._remove_drop_by_box(box)
                continue
            
            if in_hands:
                results.append((box, "HOLDING"))
                message = "Object in hands - OK"
                self._remove_drop_by_box(box)
                continue
            
            # Object is DROPPED - track countdown
            drop_id = self._find_or_create_drop(box)
            seen_drops.add(drop_id)
            
            drop = self.dropped_objects[drop_id]
            elapsed = current_time - drop['start_time']
            
            total_grace = self.grace_time
            if self._is_near_dustbin(box, dustbin_boxes):
                total_grace += self.near_dustbin_bonus
            
            remaining = total_grace - elapsed
            
            if remaining > 3:
                results.append((box, "WAIT"))
                message = f"Pick up! ({int(remaining)}s left)"
            elif remaining > 0:
                results.append((box, "WARNING"))
                message = f"⚠️ WARNING! ({int(remaining)}s left)"
            else:
                results.append((box, "LITTERING"))
                triggered = True
                message = "🚨 LITTERING CONFIRMED!"
        
        return results, triggered, message
    
    def _find_or_create_drop(self, box, threshold=100):
        """Find existing drop or create new one."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        for drop_id, drop in self.dropped_objects.items():
            dx1, dy1, dx2, dy2 = drop['box']
            dcx = (dx1 + dx2) // 2
            dcy = (dy1 + dy2) // 2
            
            dist = ((cx - dcx)**2 + (cy - dcy)**2)**0.5
            if dist < threshold:
                drop['box'] = box
                return drop_id
        
        drop_id = self.next_id
        self.next_id += 1
        
        self.dropped_objects[drop_id] = {
            'start_time': time.time(),
            'box': box
        }
        print(f"🔴 Started {self.grace_time}s countdown...")
        
        return drop_id
    
    def _remove_drop_by_box(self, box, threshold=100):
        """Remove drop tracking for this box."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        to_remove = None
        for drop_id, drop in self.dropped_objects.items():
            dx1, dy1, dx2, dy2 = drop['box']
            dcx = (dx1 + dx2) // 2
            dcy = (dy1 + dy2) // 2
            
            dist = ((cx - dcx)**2 + (cy - dcy)**2)**0.5
            if dist < threshold:
                to_remove = drop_id
                break
        
        if to_remove is not None:
            elapsed = time.time() - self.dropped_objects[to_remove]['start_time']
            if elapsed > 0.5:
                print(f"✅ Object retrieved!")
            del self.dropped_objects[to_remove]
    
    def _is_in_dustbin(self, garbage_box, dustbin_boxes):
        """Check if garbage is inside dustbin."""
        if not dustbin_boxes:
            return False
        
        gx1, gy1, gx2, gy2 = garbage_box
        gcx = (gx1 + gx2) // 2
        gcy = (gy1 + gy2) // 2
        
        for dx1, dy1, dx2, dy2 in dustbin_boxes:
            if dx1 <= gcx <= dx2 and dy1 <= gcy <= dy2:
                return True
        
        return False
    
    def _is_near_dustbin(self, garbage_box, dustbin_boxes, threshold=150):
        """Check if near dustbin."""
        if not dustbin_boxes:
            return False
        
        gx1, gy1, gx2, gy2 = garbage_box
        gcx = (gx1 + gx2) // 2
        gcy = (gy1 + gy2) // 2
        
        for dx1, dy1, dx2, dy2 in dustbin_boxes:
            dcx = (dx1 + dx2) // 2
            dcy = (dy1 + dy2) // 2
            
            dist = ((gcx - dcx)**2 + (gcy - dcy)**2)**0.5
            if dist < threshold:
                return True
        
        return False
    
    def reset(self):
        """Reset tracking."""
        self.dropped_objects = {}
        self.next_id = 0
        print("🔄 Tracker reset")
