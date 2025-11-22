# Understanding Polarity in Neuromorphic Datasets

## What is Polarity?

In neuromorphic event-based datasets, **polarity** indicates the direction of change that triggered a spike. The format is typically:
- **Time**: When the event occurred (in microseconds or milliseconds)
- **Neuron/Channel**: Which sensor element or neuron fired
- **Polarity**: The type of change (ON vs OFF, or positive vs negative)

## Why SHD Has Only Polarity = 1

The **Spiking Heidelberg Dataset (SHD)** is an **audio dataset**, not a vision dataset. It represents spoken digits converted to spike trains. In audio applications:

1. **Single Polarity is Common**: Audio signals are often represented with only one polarity because:
   - Audio waveforms naturally oscillate around zero
   - The important information is the **magnitude** and **timing** of changes, not the direction
   - Many audio-to-spike conversion methods produce unipolar events

2. **Why Keep the Field?**: Even though SHD always uses polarity=1, the field exists because:
   - It maintains consistency with the standard neuromorphic data format `(t, x, y, p)` or `(t, neuron, polarity)`
   - It allows for potential future extensions
   - It keeps compatibility with processing pipelines designed for vision datasets

## Datasets with Multiple Polarities

### Vision Datasets (DVS - Dynamic Vision Sensor)

Vision-based neuromorphic datasets **require** multiple polarities because they capture brightness changes:

#### 1. **N-MNIST** / **N-Caltech101**
- **Polarity = 1**: ON events (brightness **increases**)
- **Polarity = 0**: OFF events (brightness **decreases**)
- **Why needed**: 
  - Detecting both increases and decreases provides complete visual information
  - ON events show objects getting brighter (e.g., light turning on, object entering frame)
  - OFF events show objects getting darker (e.g., shadows, objects leaving)
  - Together they create a richer representation of motion and contrast

#### 2. **DVS Gesture Dataset**
- **Polarity = 1**: ON events (brightness increase)
- **Polarity = 0**: OFF events (brightness decrease)
- **Why needed**: 
  - Hand gestures create both brightening and darkening regions
  - Capturing both allows better tracking of hand movement
  - Essential for recognizing complex gestures

#### 3. **DAVIS (Dynamic and Active-pixel Vision Sensor)**
- Can produce both ON and OFF events
- **Why needed**:
  - Provides more complete scene understanding
  - Better edge detection (edges have both bright and dark sides)
  - Improved motion tracking

### Example: Why Polarity Matters in Vision

Imagine a ball moving across a scene:

```
Frame 1: [Dark] [Dark] [Dark] [Ball] [Light] [Light]
Frame 2: [Dark] [Dark] [Ball] [Light] [Light] [Light]
```

- **ON events** fire where the ball enters (dark → light transition)
- **OFF events** fire where the ball leaves (light → dark transition)
- **Both together** allow tracking the ball's motion and boundaries

Without polarity, you'd lose information about whether brightness is increasing or decreasing, making motion detection much harder.

## When Polarity is Less Important

### Audio Datasets (like SHD)
- **Single polarity** is often sufficient
- Audio signals are typically processed as magnitude changes
- The timing and channel (frequency) are more important than direction

### Other Sensor Modalities
- **Tactile sensors**: May use single polarity
- **Proprioceptive sensors**: Often unipolar
- **Custom sensors**: Depends on the physical phenomenon being measured

## Technical Details

### Standard Event Format

Most neuromorphic datasets use one of these formats:

1. **Vision (DVS)**: `(t, x, y, p)`
   - `t`: timestamp
   - `x, y`: pixel coordinates
   - `p`: polarity (0 or 1, or -1 and 1)

2. **Audio/Other**: `(t, channel, polarity)`
   - `t`: timestamp
   - `channel`: neuron/sensor index
   - `p`: polarity (often always 1 for audio)

### Processing Considerations

When working with datasets that have multiple polarities:

```python
# Separate ON and OFF events
on_events = events[events['p'] == 1]
off_events = events[events['p'] == 0]

# Or process together but weight differently
# Some algorithms treat ON and OFF events as separate channels
```

### Why This Matters for Your Analysis

In your manifold analysis:

1. **SHD (single polarity)**: 
   - All events have the same polarity
   - Polarity doesn't add information
   - You can ignore it or treat it as a constant

2. **Vision datasets (multiple polarities)**:
   - Polarity is a crucial feature
   - You might want to:
     - Separate ON and OFF events into different manifolds
     - Treat polarity as an additional dimension
     - Create separate analyses for each polarity

## Summary

| Dataset Type | Typical Polarities | Why |
|-------------|-------------------|-----|
| **Vision (DVS)** | 0 and 1 (or -1 and 1) | Captures both brightness increases and decreases |
| **Audio (SHD)** | Always 1 | Audio changes are typically unidirectional in spike representation |
| **Tactile** | Often 1 | Pressure changes are often unipolar |
| **Proprioceptive** | Often 1 | Joint angle changes are typically unipolar |

**Key Takeaway**: Polarity is essential for vision datasets because it captures the **direction** of brightness changes, which is crucial for understanding motion, edges, and scene dynamics. For audio and other modalities, single polarity is often sufficient because the important information is in the **magnitude** and **timing** of changes, not their direction.

