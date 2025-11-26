function [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius)
% ALIGN_AND_RELAX_MAP - CMF-GR: Collaborative Map Fusion with Graph Relaxation
%
% This function implements the hierarchical fusion approach:
% 1. Calculates rigid transformation T between Follower and Leader poses
% 2. Applies transformation to all nodes in the Follower map
% 3. Applies relaxation formula to nearby nodes for smooth deformation
%
% Inputs:
%   F_Map           - Follower map structure with .Nodes and .Count
%   F_state         - Current Follower state [x; y; theta] (3x1)
%   F_P             - Follower covariance matrix (3x3)
%   target_pose     - Leader pose to align to [x; y; theta] (3x1)
%   influence_radius - Radius for graph relaxation (meters)
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
% Rotation matrix of the correction (not used directly but kept for clarity)
R_corr = [cos(dth), -sin(dth); sin(dth), cos(dth)];

% The pivot point is the current F_state (rendezvous point)
pivot = F_state(1:2);

% --- STEP 6: Esegui CMF_Graph_Relaxation (Map_B, Map_A, T_alignment) ---
% Instead of applying T rigidly to everything, we apply it with a weight.
% Weight = 1 at rendezvous (current node).
% Weight decays with distance (spatial distance).

% Update Current State (Weight = 1 at rendezvous)
F_state(3) = angdiff(0, F_state(3) + dth);
F_state(1:2) = F_state(1:2) + translation;

% Update Map Nodes with Graph Relaxation
if F_Map.Count > 0
    for i = 1:F_Map.Count
        node_pose = F_Map.Nodes(i).pose;

        % Distance from pivot (rendezvous)
        dist = norm(node_pose(1:2) - pivot);

        if dist < influence_radius
            % Weight Function (Linear Decay)
            % w = 1.0 at pivot, w = 0.0 at influence_radius
            w = 1.0 - (dist / influence_radius);
            if w < 0, w = 0; end

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

            F_Map.Nodes(i).pose = [new_pos; new_th];
        end
    end

    % Update last_node_pose if it was modified
    F_Map.last_node_pose = F_Map.Nodes(F_Map.Count).pose;
end

% Update Covariance
% Reduce uncertainty as we have fused with Leader
F_P = F_P * 0.5;

end

function d = angdiff(a, b)
% Compute angular difference (b - a) wrapped to [-pi, pi]
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
