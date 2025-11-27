function [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius, target_P)
% ALIGN_AND_RELAX_MAP - CMF-GR: Collaborative Map Fusion with Graph Relaxation
%
% Implements Covariance Intersection (CI) for fusion and Topological Relaxation.
%
% Inputs:
%   F_Map           - Follower map structure
%   F_state         - Current Follower state [x; y; theta] (3x1)
%   F_P             - Follower covariance matrix (3x3)
%   target_pose     - Target pose to align to [x; y; theta] (3x1)
%   influence_radius - Radius for graph relaxation (meters)
%   target_P        - Target covariance matrix (3x3)

% 1. Covariance Intersection (CI)
% Calculate optimal omega using trace heuristic
% omega = trace(P_B) / (trace(P_A) + trace(P_B))
% Here A = Follower, B = Target

tr_F = trace(F_P);
tr_T = trace(target_P);

% Avoid division by zero
if (tr_F + tr_T) < 1e-9
    omega = 0.5;
else
    omega = tr_T / (tr_F + tr_T);
end

% Compute Fused Covariance
% P_fused = inv( omega * inv(F_P) + (1-omega) * inv(target_P) )
try
    inv_F = inv(F_P);
    inv_T = inv(target_P);
    inv_fused = omega * inv_F + (1 - omega) * inv_T;
    P_fused = inv(inv_fused);

    % Compute Fused State
    % x_fused = P_fused * ( omega * inv(F_P) * x_F + (1-omega) * inv(target_P) * x_T )

    % Handle angle wrapping for theta
    % We fuse [x, y] normally. For theta, we fuse deviations from F_state(3).

    % State vectors
    x_F = F_state;
    x_T = target_pose;

    % Adjust x_T(3) to be close to x_F(3) to avoid wrap-around issues during weighted average
    x_T(3) = x_F(3) + angdiff(x_F(3), x_T(3));

    term_F = omega * inv_F * x_F;
    term_T = (1 - omega) * inv_T * x_T;

    x_fused = P_fused * (term_F + term_T);
    x_fused(3) = angdiff(0, x_fused(3)); % Wrap result

catch
    % Fallback if singular
    warning('CI Inversion failed. Using simple average.');
    P_fused = (F_P + target_P) / 2;
    x_fused = F_state;
    x_fused(1:2) = (F_state(1:2) + target_pose(1:2)) / 2;
    x_fused(3) = F_state(3) + 0.5 * angdiff(F_state(3), target_pose(3));
end

% 2. Calculate Correction Vector
correction = x_fused - F_state;
correction(3) = angdiff(F_state(3), x_fused(3));

dth = correction(3);
translation = correction(1:2);

% 3. Apply Correction to Current State
F_state = x_fused;
F_P = P_fused* 1.1;

% 4. Topological Graph Relaxation
% Propagate the correction backward

if F_Map.Count > 0
    pivot = F_state(1:2) - translation; % The original position before correction

    % Convert influence_radius to topological hops (approx 2m/node)
    influence_hops = ceil(influence_radius / 2.0);
    influence_hops = min(influence_hops, F_Map.Count - 1);

    current_idx = F_Map.Count;

    for hop = 1:influence_hops
        idx = current_idx - hop;
        if idx < 1, break; end

        node_pose = F_Map.Nodes(idx).pose;

        % Weight: Linear decay
        w = 1.0 - (hop / influence_hops);

        % Apply weighted correction
        % Rotate around pivot
        dth_w = w * dth;
        R_w = [cos(dth_w), -sin(dth_w); sin(dth_w), cos(dth_w)];

        rel_pos = node_pose(1:2) - pivot;
        rot_pos = R_w * rel_pos;

        trans_w = w * translation;

        new_pos = pivot + rot_pos + trans_w;
        new_th = angdiff(0, node_pose(3) + dth_w);

        F_Map.Nodes(idx).pose = [new_pos; new_th];
    end

    % Update last_node_pose
    F_Map.last_node_pose = F_Map.Nodes(F_Map.Count).pose;
end

end

function d = angdiff(a, b)
% Compute angular difference (b - a) wrapped to [-pi, pi]
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
