% RUN_ALL_EXPERIMENTS - Batch process all KITTI sequences
%
% This script runs the collaborative SLAM simulation for all KITTI sequences
% and saves results (plots + metrics) to a 'results' folder.

clear; clc; close all;

POSES = {'00', '01', '02', '03', '04', '05', '06', '07', '08', '09'};

% Create results folder if it doesn't exist
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

fprintf('========================================\n');
fprintf('  BATCH PROCESSING ALL SEQUENCES\n');
fprintf('========================================\n\n');

for seq_idx = 1:length(POSES)
    seq_id = POSES{seq_idx};

    fprintf('\n>>> Processing Sequence %s (%d/%d)...\n', seq_id, seq_idx, length(POSES));

    try
        % Run simulation for this sequence
        [metrics, fig_handles] = run_sequence(seq_id);

        % Save metrics to text file
        metrics_file = fullfile(results_folder, sprintf('metrics_%s.txt', seq_id));
        save_metrics(metrics_file, seq_id, metrics);

        % Save plots
        save_plots(results_folder, seq_id, fig_handles);

        % Close figures to save memory
        close(fig_handles);

        fprintf('✓ Sequence %s completed successfully\n', seq_id);

    catch ME
        fprintf('✗ Error processing sequence %s: %s\n', seq_id, ME.message);
        close all; % Clean up any open figures
    end
end

fprintf('\n========================================\n');
fprintf('  BATCH PROCESSING COMPLETE\n');
fprintf('  Results saved in: %s/\n', results_folder);
fprintf('========================================\n');

%% HELPER FUNCTIONS

function [metrics, fig_handles] = run_sequence(seq_id)
% RUN_SEQUENCE - Execute simulation for a single KITTI sequence
%
% This is essentially the main.m logic extracted into a function

% Load data
filename_gt = fullfile('Dataset', 'poses', strcat(seq_id, '.txt'));
feature_file = fullfile('Dataset', 'features', strcat('kitti_', seq_id, '_features.mat'));

if ~isfile(filename_gt)
    error('Ground truth file not found: %s', filename_gt);
end

fprintf('Loading data for sequence %s...\n', seq_id);
raw_data = load(filename_gt);
GT.x = raw_data(:, 12);
GT.y = raw_data(:, 4);

GT.th = zeros(length(GT.x), 1);
for k = 1:length(GT.x)-1
    GT.th(k) = atan2(GT.y(k+1)-GT.y(k), GT.x(k+1)-GT.x(k));
end
GT.th(end) = GT.th(end-1);
GT.th = smoothdata(GT.th, 'gaussian', 10);

if isfile(feature_file)
    feat_data = load(feature_file);
    AllFeatures = feat_data.AllFeatures;
    use_visual_features = true;
else
    warning('Features not found for sequence %s. Using geometric fallback.', seq_id);
    use_visual_features = false;
    AllFeatures = [];
end

min_x = min(GT.x); max_x = max(GT.x);
min_y = min(GT.y); max_y = max(GT.y);
offset = 50;
Landmarks = [min_x-offset, min_y-offset; max_x+offset, min_y-offset;
    max_x+offset, max_y+offset; min_x-offset, max_y+offset];

% Parameters (from main.m)
dt = 0.1;
Param.sigma_v = 0.2;
Param.sigma_w = 0.05;
Param.rand_seed = 42;

MapParams.dist_thresh = 2.0;
MapParams.angle_thresh = deg2rad(20);
SeqParams.seq_len = 10;
SeqParams.match_thresh = 0.10;
Correction.gain = 0.8;
MAX_ANGLE_ERR = deg2rad(30);
MAX_DIST_JUMP = 8.0;

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

% Initialization
N = length(GT.x);

L_state = [GT.x(1); GT.y(1); GT.th(1)];
L_P = eye(3) * 0.1;
L_Map.Nodes = [];
L_Map.last_node_pose = L_state;

F_state = [0; 0; 0];
F_P = eye(3) * 0.1;
F_Map.Nodes = [];
F_Map.last_node_pose = F_state;
F_drift_state = F_state;

L_hist = zeros(3, N);
F_hist = zeros(3, N);
F_drift_hist = zeros(3, N);
L_P_hist = zeros(3, 3, N);

F_pure_state = [GT.x(1); GT.y(1); GT.th(1)];
F_pure_P = eye(3) * 0.1;
F_pure_hist = zeros(3, N);

CorrectionData = [];
rng(Param.rand_seed);

% Simulation loop
fprintf('Running simulation (%d steps)...\n', N);
tic;

EmptyNode = struct('id', 0, 'pose', [0;0;0], 'view_id', 0);
L_Map.Nodes = repmat(EmptyNode, N, 1); L_Map.Count = 0;
F_Map.Nodes = repmat(EmptyNode, N, 1); F_Map.Count = 0;

for k = 1:N-1
    dx = GT.x(k+1) - GT.x(k);
    dy = GT.y(k+1) - GT.y(k);
    dth_true = angdiff(GT.th(k), GT.th(k+1));
    v_cmd = sqrt(dx^2 + dy^2);
    w_cmd = dth_true;
    current_view_id = k;

    true_pose_k1 = [GT.x(k+1); GT.y(k+1); GT.th(k+1)];
    [L_state, L_P] = ukf(L_state, L_P, v_cmd, w_cmd, true_pose_k1, Landmarks, UkfParams);
    [L_Map, L_node_added] = update_map_rt(L_Map, L_state, current_view_id, MapParams);

    noise_v = randn * Param.sigma_v * dt;
    noise_w = randn * Param.sigma_w * dt;
    v_F = v_cmd + noise_v;
    w_F = w_cmd + noise_w;

    [F_state, F_P] = ukf(F_state, F_P, v_F, w_F, true_pose_k1, Landmarks, UkfParams);
    [F_Map, F_node_added] = update_map_rt(F_Map, F_state, current_view_id, MapParams);

    [F_pure_state, F_pure_P] = ukf(F_pure_state, F_pure_P, v_F, w_F, true_pose_k1, Landmarks, UkfParams);

    F_drift_state(1) = F_drift_state(1) + v_F * cos(F_drift_state(3));
    F_drift_state(2) = F_drift_state(2) + v_F * sin(F_drift_state(3));
    F_drift_state(3) = F_drift_state(3) + w_F;
    F_drift_hist(:, k) = F_drift_state;

    L_hist(:, k) = L_state;
    L_P_hist(:, :, k) = L_P;
    F_hist(:, k) = F_state;
    F_pure_hist(:, k) = F_pure_state;

    % Sequence matching
    is_match = false;
    best_L_idx = -1;

    if F_node_added && L_Map.Count > 0
        valid_L_nodes = L_Map.Nodes(1:L_Map.Count);
        Ln_coords = reshape([valid_L_nodes.pose], 3, [])';
        dists = sqrt((Ln_coords(:,1) - F_state(1)).^2 + (Ln_coords(:,2) - F_state(2)).^2);

        [vals, sorted_idxs] = sort(dists);
        valid_mask = vals < 15.0;
        candidate_indices = sorted_idxs(valid_mask);
        if length(candidate_indices) > 3, candidate_indices = candidate_indices(1:3); end

        best_score = 0;
        for i = 1:length(candidate_indices)
            idx_L = candidate_indices(i);
            cand_view_id = L_Map.Nodes(idx_L).view_id;

            if use_visual_features
                [match_bool, score] = perform_sequence_matching(...
                    current_view_id, cand_view_id, AllFeatures, AllFeatures, SeqParams);
            else
                match_bool = abs(current_view_id - cand_view_id) < 5; score = 1.0;
            end

            if match_bool && score > best_score
                best_score = score; best_L_idx = idx_L;
            end
        end

        if best_L_idx > 0
            is_match = true;
        end
    end

    if is_match
        target_pose = L_Map.Nodes(best_L_idx).pose;
        [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, target_pose, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP);
    end
end

L_hist(:, end) = L_state;
L_P_hist(:, :, end) = L_P;
F_hist(:, end) = F_state;
F_pure_hist(:, end) = F_pure_state;
F_drift_hist(:, end) = F_drift_state;

sim_time = toc;
fprintf('Simulation completed in %.2f sec.\n', sim_time);

% Compute metrics
metrics = compute_metrics(GT, L_hist, F_hist, F_pure_hist, F_drift_hist, L_P_hist, CorrectionData, N);
metrics.sim_time = sim_time;
metrics.seq_id = seq_id;

% Generate plots
fig_handles = generate_plots(GT, L_hist, F_hist, F_pure_hist, F_drift_hist, Landmarks, CorrectionData, SeqParams);
end

function metrics = compute_metrics(GT, L_hist, F_hist, F_pure_hist, F_drift_hist, L_P_hist, CorrectionData, N)
% Compute all metrics

% RMSE
err_L = sqrt((L_hist(1,:)' - GT.x).^2 + (L_hist(2,:)' - GT.y).^2);
err_F_drift = sqrt((F_drift_hist(1,:)' - GT.x).^2 + (F_drift_hist(2,:)' - GT.y).^2);
err_F_corr = sqrt((F_hist(1,:)' - GT.x).^2 + (F_hist(2,:)' - GT.y).^2);
err_F_pure = sqrt((F_pure_hist(1,:)' - GT.x).^2 + (F_pure_hist(2,:)' - GT.y).^2);

metrics.rmse_L = rms(err_L);
metrics.rmse_F_drift = rms(err_F_drift);
metrics.rmse_F_corr = rms(err_F_corr);
metrics.rmse_F_pure = rms(err_F_pure);

% NEES
nees_vals = zeros(N, 1);
for k = 1:N
    err_vec = [GT.x(k) - L_hist(1,k); GT.y(k) - L_hist(2,k)];
    P_pos = L_P_hist(1:2, 1:2, k);
    try
        nees_vals(k) = err_vec' * inv(P_pos) * err_vec;
    catch
        nees_vals(k) = NaN;
    end
end
nees_vals_clean = nees_vals(~isnan(nees_vals));
metrics.nees_mean = mean(nees_vals_clean);
metrics.nees_std = std(nees_vals_clean);

% ATE
metrics.ate_before = mean(err_F_pure);
metrics.ate_after = mean(err_F_corr);
metrics.ate_improvement = (metrics.ate_before - metrics.ate_after) / metrics.ate_before * 100;

% Network payload
metrics.num_corrections = size(CorrectionData, 1);
bytes_per_correction = 3 * 8 + 9 * 8;
metrics.total_payload_kb = (metrics.num_corrections * bytes_per_correction) / 1024;

% Overall improvement
metrics.overall_improvement = (metrics.rmse_F_drift - metrics.rmse_F_corr) / metrics.rmse_F_drift * 100;
end

function fig_handles = generate_plots(GT, L_hist, F_hist, F_pure_hist, F_drift_hist, Landmarks, CorrectionData, SeqParams)
% Generate plots and return figure handles

% Plot 1: Leader and Follower UKF
fig1 = figure('Visible', 'off');
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Leader and Follower with UKF (No Fusion)');
plot(GT.x, GT.y, 'Color', [0.7 0.7 0.7], 'LineWidth', 4, 'DisplayName', 'Ground Truth');
plot(Landmarks(:,1), Landmarks(:,2), 'ks', 'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', 'Landmarks');
plot(L_hist(1,:), L_hist(2,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Leader (UKF)');
plot(F_pure_hist(1,:), F_pure_hist(2,:), 'g-', 'LineWidth', 2, 'DisplayName', 'Follower (UKF Only)');
legend('Location', 'best');

% Plot 2: Follower with CMF
fig2 = figure('Visible', 'off');
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Follower with Collaborative Map Fusion (CMF-GR)');
plot(GT.x, GT.y, 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot(F_hist(1,:), F_hist(2,:), 'r-', 'LineWidth', 2, 'DisplayName', 'Follower (UKF + CMF)');
plot(L_hist(1,:), L_hist(2,:), 'b:', 'LineWidth', 1, 'DisplayName', 'Leader (Target)');
if ~isempty(CorrectionData)
    X_links = [CorrectionData(:,1), CorrectionData(:,3), nan(size(CorrectionData,1), 1)]';
    Y_links = [CorrectionData(:,2), CorrectionData(:,4), nan(size(CorrectionData,1), 1)]';
    plot(X_links(:), Y_links(:), 'g-', 'LineWidth', 1.0, 'DisplayName', 'Correction Links');
    plot(CorrectionData(:,3), CorrectionData(:,4), 'g.', 'MarkerSize', 8, 'HandleVisibility', 'off');
end
legend('Location', 'best');

fig_handles = [fig1, fig2];
end

function save_metrics(filepath, seq_id, metrics)
% Save metrics to text file

fid = fopen(filepath, 'w');

fprintf(fid, '================================================\n');
fprintf(fid, '  KITTI Sequence %s - Results\n', seq_id);
fprintf(fid, '================================================\n\n');
fprintf(fid, 'Simulation Time: %.2f seconds\n\n', metrics.sim_time);

fprintf(fid, '1. ACCURACY - RMSE Position (x,y)\n');
fprintf(fid, '   ----------------------------------------\n');
fprintf(fid, '   Leader (UKF vs GT):        %.3f m\n', metrics.rmse_L);
fprintf(fid, '   Follower Odometry:         %.3f m\n', metrics.rmse_F_drift);
fprintf(fid, '   Follower UKF Only:         %.3f m\n', metrics.rmse_F_pure);
fprintf(fid, '   Follower UKF + CMF:        %.3f m\n', metrics.rmse_F_corr);
fprintf(fid, '   UKF vs Odometry:           %.1f%% better\n\n', (metrics.rmse_F_drift - metrics.rmse_F_pure)/metrics.rmse_F_drift * 100);

fprintf(fid, '2. CONSISTENCY - NEES\n');
fprintf(fid, '   ----------------------------------------\n');
fprintf(fid, '   Mean NEES:                 %.2f ± %.2f\n', metrics.nees_mean, metrics.nees_std);
fprintf(fid, '   Expected (2-DOF):          ~2.00\n');
fprintf(fid, '   Chi-squared 95%% bound:     < 5.99\n\n');

fprintf(fid, '3. FUSION - ATE (Absolute Trajectory Error)\n');
fprintf(fid, '   ----------------------------------------\n');
fprintf(fid, '   ATE Before Fusion:         %.3f m\n', metrics.ate_before);
fprintf(fid, '   ATE After CMF-GR:          %.3f m\n', metrics.ate_after);
fprintf(fid, '   Improvement:               %.1f%%\n\n', metrics.ate_improvement);

fprintf(fid, '4. NETWORK - Data Payload\n');
fprintf(fid, '   ----------------------------------------\n');
fprintf(fid, '   Num. Corrections:          %d\n', metrics.num_corrections);
fprintf(fid, '   Total Data Exchanged:      %.2f KB\n\n', metrics.total_payload_kb);

fprintf(fid, '================================================\n');
fprintf(fid, 'OVERALL IMPROVEMENT: %.1f%%\n', metrics.overall_improvement);
fprintf(fid, '================================================\n');

fclose(fid);
end

function save_plots(results_folder, seq_id, fig_handles)
% Save figures as PNG

plot_names = {'ukf_comparison', 'cmf_fusion'};

for i = 1:length(fig_handles)
    filename = fullfile(results_folder, sprintf('%s_%s.png', seq_id, plot_names{i}));
    saveas(fig_handles(i), filename);
end
end

function d = angdiff(a, b)
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
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
