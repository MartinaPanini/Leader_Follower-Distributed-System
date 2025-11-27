function [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius, L_P)
% ALIGN_AND_RELAX_MAP - CMF-GR: Collaborative Map Fusion with Graph Relaxation
%
% This function implements the hierarchical fusion approach with TOPOLOGICAL relaxation:
% 1. Calculates rigid transformation T between Follower and Leader poses
% 2. Applies transformation to Follower state
% 3. Propagates correction BACKWARD through graph topology (not spatial distance)
%
% Inputs:
%   F_Map           - Follower map structure with .Nodes and .Count
%   F_state         - Current Follower state [x; y; theta] (3x1)
%   F_P             - Follower covariance matrix (3x3)
%   target_pose     - Leader pose to align to [x; y; theta] (3x1)
%   influence_radius - Radius for graph relaxation (meters, converted to hops)
%   L_P             - Leader covariance matrix (3x3)
%
% Outputs:
%   F_Map   - Updated Follower map
%   F_state - Corrected Follower state
%   F_P     - Updated Follower covariance

% --- STEP 4: Calcola errore relativo Pose_B vs Pose_A_matched ---
% Calculate the transform that maps F_state to target_pose
dth = angdiff(F_state(3), target_pose(3));
translation = target_pose(1:2) - F_state(1:2);

% --- STEP 5: Calcola T_alignment ---
% The pivot point is the current F_state (rendezvous point)
pivot = F_state(1:2);

% --- STEP 6: Esegui CMF_Graph_Relaxation (Map_B, Map_A, T_alignment) ---
% TOPOLOGICAL RELAXATION (Paper Eq. 10):
% Instead of using spatial distance, propagate correction backward through
% the graph structure (temporal/topological order).
% Weight decays with graph distance (hop count), not spatial distance.
%
% This is more robust in complex environments (e.g., U-turns, overlapping loops)
% where nodes can be spatially close but topologically far apart.

% Update Current State (Weight = 1 at rendezvous)
F_state(3) = angdiff(0, F_state(3) + dth);
F_state(1:2) = F_state(1:2) + translation;

% Update Map Nodes with TOPOLOGICAL Graph Relaxation
if F_Map.Count > 0
    % Convert influence_radius to topological hops
    % Approximate: assume ~2m per node on average
    influence_hops = ceil(influence_radius / 2.0);
    influence_hops = min(influence_hops, F_Map.Count - 1);  % Cap at map size

    % Current node index (most recent, at rendezvous)
    current_idx = F_Map.Count;

    % Propagate correction BACKWARD through graph topology
    for hop = 1:influence_hops
        % Index of node to correct (going backward in time)
        idx = current_idx - hop;

        if idx < 1
            break;  % Reached beginning of map
        end

        node_pose = F_Map.Nodes(idx).pose;

        % Weight Function: Linear decay with TOPOLOGICAL distance (hop count)
        % w = 1.0 at rendezvous (hop=0), w = 0.0 at influence_hops
        w = 1.0 - (hop / influence_hops);

        % Apply Weighted Correction
        % 1. Rotate around pivot by w * dth
        dth_w = w * dth;
        R_w = [cos(dth_w), -sin(dth_w); sin(dth_w), cos(dth_w)];

        rel_pos = node_pose(1:2) - pivot;
        rot_pos = R_w * rel_pos;

        % 2. Translate by w * translation
        trans_w = w * translation;

        new_pos = pivot + rot_pos + trans_w;
        new_th = angdiff(0, node_pose(3) + dth_w);

        F_Map.Nodes(idx).pose = [new_pos; new_th];
    end

    % Update last_node_pose (most recent node)
    F_Map.last_node_pose = F_Map.Nodes(F_Map.Count).pose;
end

% Update Covariance using Covariance Intersection
% This method handles unknown correlations between Leader and Follower
% Reference: DRP_Notes.pdf Cap. 15 - Distributed Data Fusion
%
% The old method (F_P = F_P * 0.5) was mathematically invalid:
% - Artificially reduced uncertainty
% - Caused filter inconsistency (NEES >> expected value)
% - Violated uncertainty propagation principles
%
% Covariance Intersection provides conservative, consistent fusion:

% Mixing parameter (0 < omega < 1)
% omega closer to 1 = trust Follower more
% omega closer to 0 = trust Leader more
% omega = 0.5 = equal weighting (conservative)
omega = 0.5;

% Estimate of Leader's uncertainty at rendezvous
% This should ideally be received from Leader, but we approximate it
% based on typical UKF performance
%P_Leader_est = diag([5^2, 5^2, 0.1^2]);  % [x_var, y_var, theta_var]

% Covariance Intersection formula:
% P_fused^-1 = omega * P_Follower^-1 + (1 - omega) * P_Leader^-1
try
    P_F_inv = inv(F_P);
    P_L_inv = inv(L_P);
    P_fused_inv = omega * P_F_inv + (1 - omega) * P_L_inv;
    F_P = inv(P_fused_inv);
catch
    % If inversion fails (singular matrix), fall back to addition
    warning('Covariance inversion failed, using conservative addition');
    F_P = F_P + L_P;
end

% Add uncertainty from visual matching/sequence matching
% This represents the imperfect nature of the rendezvous detection
R_match = diag([2^2, 2^2, 0.05^2]);  % Matching uncertainty
F_P = F_P + R_match;

% Ensure symmetry and positive definiteness
F_P = (F_P + F_P') / 2;
[U, S] = eig(F_P);
S = max(S, 1e-6);  % Ensure positive eigenvalues
F_P = U * S * U';

end

function d = angdiff(a, b)
% Compute angular difference (b - a) wrapped to [-pi, pi]
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
