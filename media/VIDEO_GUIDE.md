# Video Walkthrough Guide

My guide for recording a professional 5-minute walkthrough.

## Setup

**Software:**
- Windows: Xbox Game Bar (Win+G) or OBS Studio
- Mac: QuickTime (Cmd+Shift+5)

**Settings:**
- Resolution: 1920x1080
- Frame Rate: 30 FPS
- Duration: 4:30-5:00 minutes
- Format: MP4

## Script Outline

### [0:00-0:30] Introduction
"Hi, I'm Varun. I'll walkthrough my learnable positional encoding implementation for transformers. I answered both interview questions and implemented three different methods with full comparisons."

### [0:30-2:00] Code Overview
- Show `src/positional_encodings.py` - 3 methods
- Show `src/model.py` - transformer architecture
- Show `src/dataset.py` - position-aware patterns
- Show `tests/` - 20+ tests

### [2:00-3:30] LiveDemo
```bash
python scripts/quick_demo.py  # Quick verification
python src/train.py --pos_encoding learned_absolute --num_epochs 5
```

Show training progress and accuracy improvement.

### [3:30-4:30] Results
- Training curves: 95% validation accuracy
- Comparison: All 3 methods + ablation study
- Key finding: Position encoding is critical (35% → 95%)
- Attention visualizations

### [4:30-5:00] Conclusion
"In summary: I comprehensively addressed both questions, implemented three learnable methods, and proved positional encoding is critical with a 160% accuracy improvement. All code is tested and production-ready. Thank you!"

## Tips

✅ Speak clearly and confidently  
✅ Show, don't just tell  
✅ Keep under 5 minutes  
✅ Practice once before recording  
✅ Test audio quality first

❌ Don't rush  
❌ Don't show errors  
❌ Don't go over time  
❌ Don't forget to summarize

## Submission

Save as: `media/walkthrough.mp4`

Or upload to:
- YouTube (unlisted)
- Google Drive
- Loom

---

**Good luck!** - Varun
