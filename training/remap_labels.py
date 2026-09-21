"""
Label Remapping Script for 4-Class Waste Classification
========================================================
Converts original dataset classes to our target classes:

TARGET CLASSES (as per problem statement):
- 0: RECYCLABLE (🟦 BLUE BIN)   - Plastic, Paper, Metal
- 1: ORGANIC    (🟩 GREEN BIN)  - Food, Leaves, Wood
- 2: HAZARDOUS  (🟥 RED BIN)    - Batteries, Bulbs, Paint
- 3: GENERAL    (⬜ GREY BIN)   - Chip bags, Tissues

ORIGINAL DATASET CLASSES:
- 0: general waste    → 3 (GENERAL)
- 1: hazardous waste  → 2 (HAZARDOUS)
- 2: organic          → 1 (ORGANIC)
- 3: recyclable       → 0 (RECYCLABLE)
"""

import os
from pathlib import Path
import shutil

def remap_labels():
    """Remap all label files from original to target 4 classes"""
    
    dataset_path = Path(__file__).resolve().parent / 'dataset'
    
    # Class mapping: old_class -> new_class
    # TARGET: RECYCLABLE=0, ORGANIC=1, HAZARDOUS=2, GENERAL=3
    class_mapping = {
        0: 3,  # general waste -> GENERAL
        1: 2,  # hazardous waste -> HAZARDOUS
        2: 1,  # organic -> ORGANIC
        3: 0,  # recyclable -> RECYCLABLE
    }
    
    print("=" * 60)
    print("LABEL REMAPPING: Original → Target 4 Classes")
    print("=" * 60)
    print("\nMapping:")
    print("  Old 0 (general waste)   → New 3 (GENERAL)")
    print("  Old 1 (hazardous waste) → New 2 (HAZARDOUS)")
    print("  Old 2 (organic)         → New 1 (ORGANIC)")
    print("  Old 3 (recyclable)      → New 0 (RECYCLABLE)")
    print()
    
    total_files = 0
    total_labels = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Count per new class
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"⚠️ Skipping {split} - labels folder not found")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        print(f"📁 Processing {split}: {len(label_files)} label files...")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:  # class + 4 bbox coords minimum
                        continue
                    
                    old_class = int(parts[0])
                    
                    # Skip if class not in mapping (shouldn't happen)
                    if old_class not in class_mapping:
                        print(f"  ⚠️ Unknown class {old_class} in {label_file.name}")
                        continue
                    
                    new_class = class_mapping[old_class]
                    class_counts[new_class] += 1
                    
                    # Rebuild line with new class
                    parts[0] = str(new_class)
                    new_lines.append(' '.join(parts))
                    total_labels += 1
                
                # Write remapped labels back
                with open(label_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                    if new_lines:
                        f.write('\n')
                
                total_files += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {label_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ REMAPPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"   Total files processed: {total_files}")
    print(f"   Total labels remapped: {total_labels}")
    print(f"\n📈 Class Distribution (after remapping):")
    print(f"   RECYCLABLE (0): {class_counts[0]:,} labels")
    print(f"   ORGANIC (1):    {class_counts[1]:,} labels")
    print(f"   HAZARDOUS (2):  {class_counts[2]:,} labels")
    print(f"   GENERAL (3):    {class_counts[3]:,} labels")
    
    return True


if __name__ == "__main__":
    print("\n🗑️ Waste Classification Label Remapper (4-Class) 🗑️\n")
    
    success = remap_labels()
    
    if success:
        print("\n✅ Labels remapped successfully!")
        print("   Dataset is now ready for 3-class training.")
    else:
        print("\n❌ Remapping failed!")
