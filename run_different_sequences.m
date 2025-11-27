% RUN_DIFFERENT_SEQUENCES - Test CMF-GR with Leader and Follower on different paths
%
% This script simulates a realistic collaborative SLAM scenario where:
% - Leader follows one KITTI sequence
% - Follower follows a different KITTI sequence
% - They may have different trajectory lengths
% - Rendezvous happens when they are spatially close
% - Tests the full logic of collaborative map fusion

clear; clc; close all;

%% CONFIGURATION
LEADER_SEQ = '02';    % Leader's trajectory
FOLLOWER_SEQ = '05';  % Follower's trajectory

% For visualization - offset follower start position to avoid overlap
FOLLOWER_OFFSET = [50; 50; 0];  % [x; y; theta] offset in meters

fprintf('========================================\n');
fprintf('  DIFFERENT SEQUENCES TEST\n');
fprintf('========================================\n');
fprintf('Leader Sequence:   %s\n', LEADER_SEQ);
fprintf('Follower Sequence: %s\n', FOLLOWER_SEQ);
fprintf('Follower Offset:   [%.1f, %.1f, %.1f°]\n', ...
    FOLLOWER_OFFSET(1), FOLLOWER_OFFSET(2), rad2deg(FOLLOWER_OFFSET(3)));
fprintf('========================================\n\n');

%% LOAD DATA FOR BOTH SEQUENCES

% Leader Data
fprintf('Loading Leader data (Seq %s)...\n', LEADER_SEQ);
L_gt_file = fullfile('Dataset', 'poses', strcat(LEADER_SEQ, '.txt'));
L_feat_file = fullfile('Dataset', 'features', strcat('kitti_', LEADER_SEQ, '_features.mat'));

L_raw = load(L_gt_file);
L_GT.x = L_raw(:, 12);
L_GT.y = L_raw(:, 4);
L_GT.th = zeros(length(L_GT.x), 1);
for k = 1:length(L_GT.x)-1
    L_GT.th(k) = atan2(L_GT.y(k+1)-L_GT.y(k), L_GT.x(k+1)-L_GT.x(k));
end
L_GT.th(end) = L_GT.th(end-1);
L_GT.th = smoothdata(L_GT.th, 'gaussian', 10);

if isfile(L_feat_file)
    L_feat_data = load(L_feat_file);
    L_AllFeatures = L_feat_data.AllFeatures;
    use_L_features = true;
else
    L_AllFeatures = [];
    use_L_features = false;
end

% Follower Data
fprintf('Loading Follower data (Seq %s)...\n', FOLLOWER_SEQ);
F_gt_file = fullfile('Dataset', 'poses', strcat(FOLLOWER_SEQ, '.txt'));
F_feat_file = fullfile('Dataset', 'features', strcat('kitti_', FOLLOWER_SEQ, '_features.mat'));

F_raw = load(F_gt_file);
F_GT.x = F_raw(:, 12) + FOLLOWER_OFFSET(1);  % Apply offset
F_GT.y = F_raw(:, 4) + FOLLOWER_OFFSET(2);
F_GT.th = zeros(length(F_GT.x), 1);
for k = 1:length(F_GT.x)-1
    F_GT.th(k) = atan2(F_GT.y(k+1)-F_GT.y(k), F_GT.x(k+1)-F_GT.x(k));
end
F_GT.th(end) = F_GT.th(end-1);
F_GT.th = smoothdata(F_GT.th, 'gaussian', 10);
F_GT.th = F_GT.th + FOLLOWER_OFFSET(3);

if isfile(F_feat_file)
    F_feat_data = load(F_feat_file);
    F_AllFeatures = F_feat_data.AllFeatures;
    use_F_features = true;
else
    F_AllFeatures = [];
    use_F_features = false;
end

N_L = length(L_GT.x);
N_F = length(F_GT.x);
N_max = max(N_L, N_F);

fprintf('Leader steps:   %d\n', N_L);
fprintf('Follower steps: %d\n', N_F);
fprintf('Running for:    %d steps\n\n', N_max);

% Landmarks (union of both workspaces)
all_x = [L_GT.x; F_GT.x];
all_y = [L_GT.y; F_GT.y];
min_x = min(all_x); max_x = max(all_x);
min_y = min(all_y); max_y = max(all_y);
offset = 50;
Landmarks = [min_x-offset, min_y-offset; max_x+offset, min_y-offset;
    max_x+offset, max_y+offset; min_x-offset, max_y+offset];

%% PARAMETERS
dt = 0.1;
Param.sigma_v = 0.2;
Param.sigma_w = 0.05;
Param.rand_seed = 42;

MapParams.dist_thresh = 2.0;
MapParams.angle_thresh = deg2rad(20);
SeqParams.seq_len = 10;
SeqParams.match_thresh = 0.10;
Correction.gain = 0.8;
MAX_ANGLE_ERR = deg2rad(45);
MAX_DIST_JUMP = 25.0;  % Larger for different sequences

% Rendezvous detection threshold
RENDEZVOUS_DIST = 150.0;  % Meters - when L and F are this close, check for match

UkfParams.sigma_scale_odom = 0.05;
UkfParams.sigma_rot_odom   = 0.002;
UkfParams.sigma_range      = 0.1;
UkfParams.sigma_bearing    = 0.01;
UkfParams.max_sensor_range = 250;
UkfParams.Q = diag([0.05^2, 0.01^2]);
UkfParams.R = diag([UkfParams.sigma_range^2, UkfParams.sigma_bearing^2]);

n = 3;
alpha = 1.0;
beta = 2;
kappa = 0;
lambda = alpha^2 * (n + kappa) - n;
UkfParams.n = n;
UkfParams.lambda = lambda;
UkfParams.c = sqrt(n + lambda);

UkfParams.Wm = zeros(1, 2*n+1);
UkfParams.Wc = zeros(1, 2*n+1);
UkfParams.Wm(1) = lambda / (n + lambda);
UkfParams.Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);
UkfParams.Wm(2:end) = 1 / (2 * (n + lambda));
UkfParams.Wc(2:end) = 1 / (2 * (n + lambda));

%% INITIALIZATION

% Leader
L_state = [L_GT.x(1); L_GT.y(1); L_GT.th(1)];
L_P = eye(3) * 0.1;
L_Map.Nodes = [];
L_Map.last_node_pose = L_state;

% Follower (with small offset error)
F_state = [F_GT.x(1); F_GT.y(1); F_GT.th(1)] + randn(3,1) .* [5; 5; 0.05];
F_P = eye(3) * 0.1;
F_Map.Nodes = [];
F_Map.last_node_pose = F_state;

% History
L_hist = zeros(3, N_max);
F_hist = zeros(3, N_max);
L_hist(:, 1) = L_state;
F_hist(:, 1) = F_state;
L_P_hist = zeros(3, 3, N_max);
L_P_hist(:, :, 1) = L_P;

CorrectionData = [];
RendezvousData = [];  % Track when they meet

rng(Param.rand_seed);

%% SIMULATION

fprintf('Starting simulation...\n');
tic;

EmptyNode = struct('id', 0, 'pose', [0;0;0], 'view_id', 0);
L_Map.Nodes = repmat(EmptyNode, max(N_L, N_F), 1); L_Map.Count = 0;
F_Map.Nodes = repmat(EmptyNode, max(N_L, N_F), 1); F_Map.Count = 0;

for k = 1:N_max-1

    % --- LEADER UPDATE (if still active) ---
    if k < N_L
        dx_L = L_GT.x(k+1) - L_GT.x(k);
        dy_L = L_GT.y(k+1) - L_GT.y(k);
        dth_L = angdiff(L_GT.th(k), L_GT.th(k+1));
        v_L = sqrt(dx_L^2 + dy_L^2);
        w_L = dth_L;

        true_pose_L = [L_GT.x(k+1); L_GT.y(k+1); L_GT.th(k+1)];
        [L_state, L_P] = run_local_ukf(L_state, L_P, v_L, w_L, true_pose_L, Landmarks, UkfParams);
        [L_Map, L_node_added] = update_map_rt(L_Map, L_state, k, MapParams);

        L_hist(:, k+1) = L_state;
        L_P_hist(:, :, k+1) = L_P;
    else
        % Leader finished, maintain last state
        L_hist(:, k+1) = L_hist(:, k);
        L_P_hist(:, :, k+1) = L_P_hist(:, :, k);
        L_node_added = false;
    end

    % --- FOLLOWER UPDATE (if still active) ---
    if k < N_F
        dx_F = F_GT.x(k+1) - F_GT.x(k);
        dy_F = F_GT.y(k+1) - F_GT.y(k);
        dth_F = angdiff(F_GT.th(k), F_GT.th(k+1));

        % Add noise to follower
        noise_v = randn * Param.sigma_v * dt;
        noise_w = randn * Param.sigma_w * dt;
        v_F = sqrt(dx_F^2 + dy_F^2) + noise_v;
        w_F = dth_F + noise_w;

        true_pose_F = [F_GT.x(k+1); F_GT.y(k+1); F_GT.th(k+1)];
        [F_state, F_P] = run_local_ukf(F_state, F_P, v_F, w_F, true_pose_F, Landmarks, UkfParams);
        [F_Map, F_node_added] = update_map_rt(F_Map, F_state, k, MapParams);

        F_hist(:, k+1) = F_state;
    else
        % Follower finished, maintain last state
        F_hist(:, k+1) = F_hist(:, k);
        F_node_added = false;
    end

    % --- RENDEZVOUS DETECTION & CMF-GR ---
    % Check if Leader and Follower are spatially close
    dist_L_F = norm(L_state(1:2) - F_state(1:2));

    if dist_L_F < RENDEZVOUS_DIST && F_node_added && L_Map.Count > 0

        % Find closest Leader node to Follower position
        valid_L_nodes = L_Map.Nodes(1:L_Map.Count);
        Ln_coords = reshape([valid_L_nodes.pose], 3, [])';
        dists = sqrt((Ln_coords(:,1) - F_state(1)).^2 + (Ln_coords(:,2) - F_state(2)).^2);

        [min_dist, best_L_idx] = min(dists);

        if min_dist < RENDEZVOUS_DIST
            % Record rendezvous
            RendezvousData = [RendezvousData; k, dist_L_F];

            % Execute CMF-GR
            target_pose = L_Map.Nodes(best_L_idx).pose;
            [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(...
                F_Map, L_Map, F_state, target_pose, F_P, L_P, ...
                CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP);

            F_hist(:, k+1) = F_state;  % Update history with corrected state
        end
    end

    if mod(k, 200) == 0
        fprintf('.');
    end
end

fprintf('\n');
sim_time = toc;
fprintf('Simulation completed in %.2f seconds\n\n', sim_time);

%% COMPUTE METRICS

% Leader metrics (compared to Leader GT)
L_valid_idx = 1:N_L;
err_L = sqrt((L_hist(1, L_valid_idx)' - L_GT.x).^2 + (L_hist(2, L_valid_idx)' - L_GT.y).^2);
rmse_L = rms(err_L);

% Follower metrics (compared to Follower GT)
F_valid_idx = 1:N_F;
err_F = sqrt((F_hist(1, F_valid_idx)' - F_GT.x).^2 + (F_hist(2, F_valid_idx)' - F_GT.y).^2);
rmse_F = rms(err_F);
ate_F = mean(err_F);

% NEES for Leader
nees_vals = zeros(N_L, 1);
for k = 1:N_L
    err_vec = [L_GT.x(k) - L_hist(1,k); L_GT.y(k) - L_hist(2,k)];
    P_pos = L_P_hist(1:2, 1:2, k);
    try
        nees_vals(k) = err_vec' * inv(P_pos) * err_vec;
    catch
        nees_vals(k) = NaN;
    end
end
nees_vals_clean = nees_vals(~isnan(nees_vals));
nees_mean = mean(nees_vals_clean);
nees_std = std(nees_vals_clean);

num_rendezvous = size(RendezvousData, 1);
num_corrections = size(CorrectionData, 1);

fprintf('========================================\n');
fprintf('         RESULTS\n');
fprintf('========================================\n\n');

fprintf('ACCURACY:\n');
fprintf('  Leader RMSE:        %.3f m\n', rmse_L);
fprintf('  Follower RMSE:      %.3f m\n', rmse_F);
fprintf('  Follower ATE:       %.3f m\n\n', ate_F);

fprintf('CONSISTENCY:\n');
fprintf('  Leader NEES:        %.2f ± %.2f\n\n', nees_mean, nees_std);

fprintf('COLLABORATION:\n');
fprintf('  Rendezvous Events:  %d\n', num_rendezvous);
fprintf('  Corrections:        %d\n', num_corrections);
fprintf('  Correction Rate:    %.1f%%\n\n', (num_corrections/num_rendezvous)*100);

fprintf('========================================\n');

%% PLOTTING

% Plot 1: Both trajectories with ground truth
figure('Name', 'Different Sequences - Overview', 'Color', 'w', 'Position', [50, 100, 1200, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title(sprintf('Leader (Seq %s) + Follower (Seq %s) with CMF-GR', LEADER_SEQ, FOLLOWER_SEQ));

% Ground truths
plot(L_GT.x, L_GT.y, 'Color', [0.7 0.7 0.7], 'LineWidth', 3, 'DisplayName', 'Leader GT');
plot(F_GT.x, F_GT.y, 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'LineStyle', '--', 'DisplayName', 'Follower GT');

% Estimates
plot(L_hist(1,L_valid_idx), L_hist(2,L_valid_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'Leader (UKF)');
plot(F_hist(1,F_valid_idx), F_hist(2,F_valid_idx), 'r-', 'LineWidth', 2, 'DisplayName', 'Follower (UKF+CMF)');

% Landmarks
plot(Landmarks(:,1), Landmarks(:,2), 'ks', 'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', 'Landmarks');

% Corrections
if ~isempty(CorrectionData)
    X_links = [CorrectionData(:,1), CorrectionData(:,3), nan(size(CorrectionData,1), 1)]';
    Y_links = [CorrectionData(:,2), CorrectionData(:,4), nan(size(CorrectionData,1), 1)]';
    plot(X_links(:), Y_links(:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'CMF Corrections');
    plot(CorrectionData(:,3), CorrectionData(:,4), 'g.', 'MarkerSize', 12, 'HandleVisibility', 'off');
end

% Rendezvous points
if ~isempty(RendezvousData)
    for i = 1:size(RendezvousData, 1)
        k_rd = RendezvousData(i, 1);
        if i == 1
            plot(F_hist(1, k_rd), F_hist(2, k_rd), 'mo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Rendezvous');
        else
            plot(F_hist(1, k_rd), F_hist(2, k_rd), 'mo', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end
    end
end

legend('Location', 'best');

% Plot 2: Error over time
figure('Name', 'Position Error Over Time', 'Color', 'w', 'Position', [100, 150, 1000, 400]);
subplot(1,2,1);
plot(1:N_L, err_L, 'b-', 'LineWidth', 1.5);
hold on; grid on;
xlabel('Time Step'); ylabel('Position Error [m]');
title(sprintf('Leader (Seq %s) - RMSE: %.2fm', LEADER_SEQ, rmse_L));
yline(rmse_L, 'r--', 'RMSE', 'LineWidth', 1.5);

subplot(1,2,2);
plot(1:N_F, err_F, 'r-', 'LineWidth', 1.5);
hold on; grid on;
xlabel('Time Step'); ylabel('Position Error [m]');
title(sprintf('Follower (Seq %s) - RMSE: %.2fm', FOLLOWER_SEQ, rmse_F));
yline(ate_F, 'g--', 'ATE', 'LineWidth', 1.5);

% Mark rendezvous events
if ~isempty(RendezvousData)
    for i = 1:size(RendezvousData, 1)
        k_rd = RendezvousData(i, 1);
        if k_rd <= N_F
            xline(k_rd, 'm--', 'Alpha', 0.5);
        end
    end
end

%% HELPER FUNCTIONS

function [Map, added] = update_map_rt(Map, pose, view_id, Params)
added = false;
if Map.Count == 0
    Map.Count = 1; Map.Nodes(1).id = 1; Map.Nodes(1).pose = pose;
    Map.Nodes(1).view_id = view_id; Map.last_node_pose = pose; added = true; return;
end
dist = norm(pose(1:2) - Map.last_node_pose(1:2));
dth = abs(angdiff(Map.last_node_pose(3), pose(3)));
if dist > Params.dist_thresh || dth > Params.angle_thresh
    Map.Count = Map.Count + 1; idx = Map.Count;
    Map.Nodes(idx).id = idx; Map.Nodes(idx).pose = pose;
    Map.Nodes(idx).view_id = view_id; Map.last_node_pose = pose; added = true;
end
end

function [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, x_L, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP)
% CMF_GR: Collaborative Map Fusion / Global Registration
target_pose = x_L;

% Heading Check
dth_check = abs(angdiff(F_state(3), target_pose(3)));

% Check Euclidean distance
dist_check = norm(target_pose(1:2) - F_state(1:2));

if dth_check < MAX_ANGLE_ERR && dist_check < MAX_DIST_JUMP
    % Store pivot for correction data recording
    pivot = F_state(1:2);

    % Influence Radius
    influence_radius = 50.0;

    % Call align_and_relax_map to perform CMF-GR
    [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius);

    % Record Correction
    CorrectionData = [CorrectionData; pivot(1), pivot(2), target_pose(1), target_pose(2)];
end
end

function d = angdiff(a, b)
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
