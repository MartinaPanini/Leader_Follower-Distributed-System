# Distributed Multi-Robot SLAM System

A distributed SLAM system implementing Collaborative Map Fusion with Graph Relaxation (CMF-GR) and Covariance Intersection for multi-robot state estimation and map merging.

## System Overview

This system demonstrates a distributed architecture where multiple robots:
- Perform independent state estimation using an Unscented Kalman Filter (UKF)
- Build topological maps through Dead Reckoning (odometry-only)
- Detect loop closures using visual sequence matching
- Fuse maps using Covariance Intersection and topological relaxation

## Architecture

### Time-Delayed Deployment
Robots follow the same ground truth path but start at different times, simulating a fleet deployment scenario.

### Trust No One Protocol
Each robot checks for loop closures against:
- **Self-Check**: Its own historical nodes (excluding recent ones)
- **Cross-Check**: Other robots' maps

## Main Files

### `main.m`
Main simulation script that orchestrates the entire system.

**Key Components**:
- **Configuration**: Parameters for UKF, map matching, and simulation
- **Initialization**: Robot fleet setup with individual noise profiles
- **Simulation Loop**: Per-robot state estimation, mapping, and loop closure
- **Metrics**: RMSE and NEES calculation for each robot
- **Visualization**: Trajectory plots and NEES evolution

**Robot Structure**:
```matlab
Robots(i).id           % Robot identifier
Robots(i).name         % Robot name
Robots(i).color        % Plot color
Robots(i).start_delay  % Time offset in frames
Robots(i).sigma_v      % Linear velocity noise
Robots(i).sigma_w      % Angular velocity noise
Robots(i).state        % Current pose [x; y; theta]
Robots(i).P            % Covariance matrix
Robots(i).Map          % Topological map
Robots(i).hist         % Historical data
```

### `ukf.m`
Unscented Kalman Filter for state propagation (Dead Reckoning mode).

**Function**: `[x_next, P_next] = ukf(x, P, v, w, Params)`

**Purpose**: Propagates robot state using odometry inputs without measurement updates. Landmarks and GPS are intentionally removed to simulate pure dead reckoning, where correction only occurs through loop closures.

**Process**:
1. Generate sigma points around current state
2. Propagate through motion model
3. Compute predicted mean and covariance
4. Add process noise Q

### `align_and_relax_map.m`
Collaborative Map Fusion with Graph Relaxation (CMF-GR).

**Function**: `[F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius, target_P)`

**Purpose**: Fuses the current robot state with a matched historical node using Covariance Intersection and propagates the correction backwards through the topological map.

**Steps**:
1. **Covariance Intersection**: Optimally fuses estimates using trace-based weighting
2. **State Fusion**: Computes fused state with proper angle wrapping
3. **Topological Relaxation**: Propagates correction backwards with linear decay

**Key Formula**:
```
ω = trace(P_target) / (trace(P_current) + trace(P_target))
P_fused = inv(ω * inv(P_current) + (1-ω) * inv(P_target))
```

### `perform_sequence_matching.m`
Visual or geometric matching for loop closure detection.

**Function**: `[is_match, score] = perform_sequence_matching(curr_id, cand_id, Features1, Features2, Params)`

**Purpose**: Determines if two views correspond to the same location using visual features or geometric proximity.

**Modes**:
- **Visual**: Compares feature descriptors over a temporal sequence
- **Geometric Fallback**: Uses strict distance threshold (< 3m)

### `update_map_rt.m`
Real-time map node management.

**Function**: `[Map, added] = update_map_rt(Map, pose, view_id, Params, source_name)`

**Purpose**: Adds nodes to the topological map based on distance and angle thresholds to avoid redundancy.

**Criteria**:
- Distance > 2.0m from last node, OR
- Angle change > 20° from last node

## Parameters

### UKF Parameters
- `Q = diag([0.01^2, 0.005^2])`: Process noise (v, w)
- `R = diag([0.1^2, 0.01^2])`: Measurement noise (unused in DR mode)
- `alpha = 1.0`: UKF spread parameter
- `beta = 2`: Prior distribution parameter
- `kappa = 0`: Secondary scaling parameter

### Odometry Noise
- `sigma_v = 0.01`: Linear velocity noise
- `sigma_w = 0.002`: Angular velocity noise

### Loop Closure
- `search_radius = 100m`: Candidate search distance
- `geometric_threshold = 3m`: Geometric match distance
- `match_thresh = 0.10`: Visual similarity threshold
- `recent_cutoff = 200 frames`: Self-check exclusion window
- `influence_radius = 50m`: Map relaxation propagation distance

## Metrics

### RMSE (Root Mean Square Error)
Euclidean distance error between estimated and ground truth positions.

### NEES (Normalized Estimation Error Squared)
Measures filter consistency. For 2D position:
```
NEES = e^T * P^-1 * e
```
Expected value: χ²(2) ≈ 2.0. Threshold: 5.99 for consistency.

## Data

### KITTI Dataset
Requires KITTI odometry dataset (sequence 00):
- `Dataset/sequences/00/image_0/`: Images
- `Dataset/poses/00.txt`: Ground truth poses
- `Dataset/features/kitti_00_features.mat`: Pre-extracted visual features

## Usage

```matlab
% Run simulation
main

% Expected output:
% - RMSE and NEES for each robot
% - Loop closure statistics
% - Trajectory plots
% - NEES evolution plots
```

## Algorithm Flow

1. **Initialization**: Setup N robots with time delays
2. **For each step k**:
   - **For each active robot r**:
     - Get noisy odometry from ground truth
     - **Predict**: UKF propagation (Dead Reckoning)
     - **Map Update**: Add node if distance/angle threshold met
     - **Loop Closure Detection**:
       - Find spatial candidates (100m radius)
       - Apply self-check constraint (exclude recent 200)
       - Perform sequence matching or geometric check
       - If match found → Execute CMF-GR
     - Save history
3. **Compute metrics**: RMSE, NEES, Loop Closure counts
4. **Visualize**: Trajectories, maps, NEES

## Key Insights

### Why Dead Reckoning?
Removing landmarks simulates realistic SLAM where external absolute positioning is unavailable. The robot drifts significantly (100-200m), making loop closure critical for localization.

### Why Covariance Intersection?
Traditional Kalman fusion assumes known correlations. CI handles unknown cross-correlations between robots, providing conservative but consistent estimates.

### Why Topological Relaxation?
Simple pose correction creates discontinuities. Topological relaxation smoothly propagates corrections through the map graph, maintaining spatial consistency.

## References

- Covariance Intersection: S. Julier & J. Uhlmann, "A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations"
- Graph Relaxation: Distributed Robotics & Perception Notes, Chapter 15
- UKF: S. Julier & J. Uhlmann, "Unscented Filtering and Nonlinear Estimation"