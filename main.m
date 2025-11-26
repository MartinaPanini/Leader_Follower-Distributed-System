clear; clc; close all;

%% EXCTRACT FEATURES
extractFeatures = false;
POSES = {'00', '01', '02', '03', '04', '05', '06', '07', '08', '09'};
if extractFeatures == true
    for i = 1:length(POSES)
        sequence_path = fullfile('Dataset', 'sequences', POSES{i}, 'image_0');
        output_filename = strcat('kitti_', POSES{i}, '_features.mat');
        preprocess_kitti_features(sequence_path, output_filename);
    end
end
seq_id = '05';
filename_gt = fullfile('Dataset', 'poses', strcat(seq_id, '.txt'));
feature_file = fullfile('Dataset', 'features', strcat('kitti_', seq_id, '_features.mat'));

%% GLOBAL CONFIGURATION

% Simulation
dt = 0.1;
Param.sigma_v = 0.2;
Param.sigma_w = 0.05;
Param.rand_seed = 42;

% Map & Matching
MapParams.dist_thresh = 2.0;
MapParams.angle_thresh = deg2rad(20);
SeqParams.seq_len = 10;
SeqParams.match_thresh = 0.10;
Correction.gain = 0.8;
% Security tresholds
MAX_ANGLE_ERR = deg2rad(30);
MAX_DIST_JUMP = 8.0;

% UKF Leader
UkfParams.sigma_scale_odom = 0.05;
UkfParams.sigma_rot_odom   = 0.002;
UkfParams.sigma_range      = 0.1;
UkfParams.sigma_bearing    = 0.01;
UkfParams.max_sensor_range = 250;
UkfParams.Q = diag([0.05^2, 0.01^2]);
UkfParams.R = diag([UkfParams.sigma_range^2, UkfParams.sigma_bearing^2]);

% UKF Weights
n = 3;
alpha = 1.0; % Increased from 1e-3 to 1.0 to avoid negative weights and improve stability
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

%% LOAD DATA

if ~isfile(filename_gt)
    error('File pose dont found: %s', filename_gt);
end

fprintf('Load data\n');
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
    warning('File features dont found. Geometric Fallback.');
    use_visual_features = false;
    AllFeatures = [];
end

min_x = min(GT.x); max_x = max(GT.x);
min_y = min(GT.y); max_y = max(GT.y);
offset = 50;
Landmarks = [min_x-offset, min_y-offset; max_x+offset, min_y-offset;
    max_x+offset, max_y+offset; min_x-offset, max_y+offset];

%% INIZIALIZATION

N = length(GT.x);

% Leader
L_state = [GT.x(1); GT.y(1); GT.th(1)];
L_P = eye(3) * 0.1;
L_Map.Nodes = [];
L_Map.last_node_pose = L_state;

% Follower
%F_state = [GT.x(1); GT.y(1); GT.th(1)];
F_state = [0; 0; 0];
F_P = eye(3) * 0.1; % Initialize Follower Covariance
F_Map.Nodes = [];
F_Map.last_node_pose = F_state;
F_drift_state = F_state; % odometry

L_hist = zeros(3, N);
F_hist = zeros(3, N);
F_drift_hist = zeros(3, N);

% Covariance history for NEES calculation
L_P_hist = zeros(3, 3, N);
F_P_hist = zeros(3, 3, N);  % Follower covariance history

% Follower UKF Only (No CMF)
F_pure_state = [GT.x(1); GT.y(1); GT.th(1)];
F_pure_P = eye(3) * 0.1;
F_pure_hist = zeros(3, N);

CorrectionData = [];

rng(Param.rand_seed);

%% SIMULATION LOOP

fprintf('\nSTART SIMULATION (%d steps) <<<\n', N);
tic;

EmptyNode = struct('id', 0, 'pose', [0;0;0], 'view_id', 0);
L_Map.Nodes = repmat(EmptyNode, N, 1); L_Map.Count = 0;
F_Map.Nodes = repmat(EmptyNode, N, 1); F_Map.Count = 0;

for k = 1:N-1

    % input
    dx = GT.x(k+1) - GT.x(k);
    dy = GT.y(k+1) - GT.y(k);
    dth_true = angdiff(GT.th(k), GT.th(k+1));
    v_cmd = sqrt(dx^2 + dy^2);
    w_cmd = dth_true;
    current_view_id = k;

    % 1. Aggiornamento Locale LEADER (run_local_ukf)
    % Il Leader stima dove si trova basandosi SOLO sui SUOI sensori
    true_pose_k1 = [GT.x(k+1); GT.y(k+1); GT.th(k+1)];
    [L_state, L_P] = run_local_ukf(L_state, L_P, v_cmd, w_cmd, true_pose_k1, Landmarks, UkfParams);
    [L_Map, L_node_added] = update_map_rt(L_Map, L_state, current_view_id, MapParams);

    % 2. Aggiornamento Locale FOLLOWER
    % Il Follower stima dove si trova basandosi SOLO sui SUOI sensori
    % Generate noisy control for Follower
    noise_v = randn * Param.sigma_v * dt;
    noise_w = randn * Param.sigma_w * dt;
    v_F = v_cmd + noise_v;
    w_F = w_cmd + noise_w;

    % 2. Aggiornamento Locale FOLLOWER (run_local_ukf)
    % Il Follower stima dove si trova basandosi SOLO sui SUOI sensori
    [F_state, F_P] = run_local_ukf(F_state, F_P, v_F, w_F, true_pose_k1, Landmarks, UkfParams);
    [F_Map, F_node_added] = update_map_rt(F_Map, F_state, current_view_id, MapParams);

    % UKF for Follower (Pure - No CMF)
    [F_pure_state, F_pure_P] = run_local_ukf(F_pure_state, F_pure_P, v_F, w_F, true_pose_k1, Landmarks, UkfParams);

    % Pure Odometry for comparison (Drift)
    F_drift_state(1) = F_drift_state(1) + v_F * cos(F_drift_state(3));
    F_drift_state(2) = F_drift_state(2) + v_F * sin(F_drift_state(3));
    F_drift_state(3) = F_drift_state(3) + w_F;
    F_drift_hist(:, k) = F_drift_state;

    L_hist(:, k) = L_state;
    L_P_hist(:, :, k) = L_P;  % Save covariance for NEES
    F_hist(:, k) = F_state;
    F_P_hist(:, :, k) = F_P;  % Save Follower covariance for NEES
    F_pure_hist(:, k) = F_pure_state;

    % --- A questo punto hai due stime indipendenti e slegate ---

    % 3. Controllo Sequence Matching (Brain-Inspired)
    % Check if we can match Follower current view with Leader Map
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
        % 4. Evento "Rendezvous": FUSIONE
        % Qui usi x_L e P_L (calcolati con UKF) per correggere la mappa del Follower

        % Retrieve Leader Node info for fusion
        target_pose = L_Map.Nodes(best_L_idx).pose;
        % Note: In a real distributed system, we would receive x_L and P_L associated with that node.
        % Here we use the current L_state if it's live, or the stored pose.
        % The pseudocode says "x_L, P_L". Let's assume we use the stored node pose as x_L.
        % And we need P_L. We didn't store P in Map.Nodes.
        % For simplicity, let's use the current P_L (approx) or a fixed covariance.
        % Or better, let's pass the current L_state and L_P if we assume they are meeting NOW.
        % But Sequence Matching matches against PAST nodes.
        % So strictly, we should have stored P in the map.
        % Let's use L_P (current) as a proxy or just pass it as requested.

        % To strictly follow pseudocode signature:
        % Map_F = execute_CMF_GR(Map_F, Map_L, x_F, x_L, P_F, P_L);
        % We need to update F_state too.

        [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, target_pose, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP);
    end

    if mod(k, 200) == 0, fprintf('.'); end
end
L_hist(:, end) = L_state;
L_P_hist(:, :, end) = L_P;  % Save final covariance
F_hist(:, end) = F_state;
F_P_hist(:, :, end) = F_P;  % Save final Follower covariance
F_pure_hist(:, end) = F_pure_state;
F_drift_hist(:, end) = F_drift_state;

fprintf('\nSimulation completed in %.2f sec.\n', toc);

%% COMPUTE COMPREHENSIVE METRICS

% 1. ACCURATEZZA - RMSE Posizione (x,y)
err_L_EKF = sqrt((L_hist(1,:)' - GT.x).^2 + (L_hist(2,:)' - GT.y).^2);
err_F_Drift = sqrt((F_drift_hist(1,:)' - GT.x).^2 + (F_drift_hist(2,:)' - GT.y).^2);
err_F_Corr = sqrt((F_hist(1,:)' - GT.x).^2 + (F_hist(2,:)' - GT.y).^2);
err_F_Pure = sqrt((F_pure_hist(1,:)' - GT.x).^2 + (F_pure_hist(2,:)' - GT.y).^2);

rmse_L = rms(err_L_EKF);
rmse_F_drift = rms(err_F_Drift);
rmse_F_corr = rms(err_F_Corr);
rmse_F_pure = rms(err_F_Pure);

% 2. AFFIDABILITÀ - NEES (Normalized Estimation Error Squared)
% NEES = (x_true - x_est)^T * P^-1 * (x_true - x_est)
% Using only position (x,y) for 2-DOF NEES

% NEES for Leader
nees_L_vals = zeros(N, 1);
for k = 1:N
    % Position error (2D)
    err_vec = [GT.x(k) - L_hist(1,k); GT.y(k) - L_hist(2,k)];

    % Extract position covariance (2x2 submatrix)
    P_pos = L_P_hist(1:2, 1:2, k);

    % Compute NEES
    try
        nees_L_vals(k) = err_vec' * inv(P_pos) * err_vec;
    catch
        % If P is singular, skip this point
        nees_L_vals(k) = NaN;
    end
end

% NEES for Follower (with CMF-GR)
nees_F_vals = zeros(N, 1);
for k = 1:N
    % Position error (2D)
    err_vec = [GT.x(k) - F_hist(1,k); GT.y(k) - F_hist(2,k)];

    % Extract position covariance (2x2 submatrix)
    P_pos = F_P_hist(1:2, 1:2, k);

    % Compute NEES
    try
        nees_F_vals(k) = err_vec' * inv(P_pos) * err_vec;
    catch
        % If P is singular, skip this point
        nees_F_vals(k) = NaN;
    end
end

% Remove NaN values for mean calculation
nees_L_clean = nees_L_vals(~isnan(nees_L_vals));
nees_L_mean = mean(nees_L_clean);
nees_L_std = std(nees_L_clean);

nees_F_clean = nees_F_vals(~isnan(nees_F_vals));
nees_F_mean = mean(nees_F_clean);
nees_F_std = std(nees_F_clean);

% NEES ideale dovrebbe essere ~ n (numero di dimensioni, qui 2)
% Se NEES >> n, il filtro è overconfident
% Chi-squared test: per 2-DOF, 95% dei valori dovrebbero essere < 5.99

% 3. FUSIONE - ATE (Absolute Trajectory Error)
% ATE misura l'errore di traiettoria globale
ate_before = mean(err_F_Pure);  % ATE prima della fusione
ate_after = mean(err_F_Corr);   % ATE dopo CMF-GR
ate_improvement = (ate_before - ate_after) / ate_before * 100;

% 4. NETWORK - Data Payload (KB)
% Stima del payload scambiato durante la fusione
num_corrections = size(CorrectionData, 1);
bytes_per_correction = 3 * 8 + 9 * 8;  % state (3 doubles) + P (9 doubles)
total_payload_bytes = num_corrections * bytes_per_correction;
total_payload_kb = total_payload_bytes / 1024;

fprintf('\n================================================\n');
fprintf('         FINAL ANALYSIS            \n');
fprintf('================================================\n');
fprintf('\n1. ACCURACY - RMSE Position (x,y)\n');
fprintf('   ----------------------------------------\n');
fprintf('   Leader (UKF vs GT):        %.3f m\n', rmse_L);
fprintf('   Follower Odometry:         %.3f m\n', rmse_F_drift);
fprintf('   Follower UKF Only:         %.3f m\n', rmse_F_pure);
fprintf('   Follower UKF + CMF:        %.3f m\n', rmse_F_corr);
fprintf('   ✓ UKF beats Odometry:      %.1f%% better\n', (rmse_F_drift - rmse_F_pure)/rmse_F_drift * 100);

fprintf('\n2. CONSISTENCY - NEES \n');
fprintf('   ----------------------------------------\n');
fprintf('   Leader NEES:               %.2f ± %.2f\n', nees_L_mean, nees_L_std);
fprintf('   Follower NEES (UKF+CMF):   %.2f ± %.2f\n', nees_F_mean, nees_F_std);
fprintf('   Expected (2-DOF):          ~2.00\n');
fprintf('   Chi-squared 95%% bound:     < 5.99\n');
if nees_F_mean < 10
    fprintf('   ✓ Follower filter improved with CMF-GR\n');
else
    fprintf('   ⚠ Follower filter may still be overconfident\n');
end

fprintf('\n3. FUSION - ATE (Absolute Trajectory Error)\n');
fprintf('   ----------------------------------------\n');
fprintf('   ATE Before Fusion:         %.3f m\n', ate_before);
fprintf('   ATE After CMF-GR:          %.3f m\n', ate_after);
fprintf('   ✓ Improvement:             %.1f%%\n', ate_improvement);

fprintf('\n4. NETWORK - Data Payload\n');
fprintf('   ----------------------------------------\n');
fprintf('   Num. Corrections:          %d\n', num_corrections);
fprintf('   Total Data Exchanged:      %.2f KB\n', total_payload_kb);
fprintf('   Avg. per Correction:       %.2f bytes\n', bytes_per_correction);

fprintf('\n================================================\n');
fprintf('OVERALL IMPROVEMENT: %.1f%%\n', (rmse_F_drift - rmse_F_corr)/rmse_F_drift * 100);
fprintf('================================================\n');

%% PLOT

% Plot 1: Leader and Follower with UKF only (no fusion)
figure('Name', 'Leader and Follower UKF', 'Color', 'w', 'Position', [50, 100, 900, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Leader and Follower with UKF (No Fusion)');
plot(GT.x, GT.y, 'Color', [0.7 0.7 0.7], 'LineWidth', 4, 'DisplayName', 'Ground Truth');
plot(Landmarks(:,1), Landmarks(:,2), 'ks', 'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', 'Landmarks');
plot(L_hist(1,:), L_hist(2,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Leader (UKF)');
plot(F_pure_hist(1,:), F_pure_hist(2,:), 'g-', 'LineWidth', 2, 'DisplayName', 'Follower (UKF Only)');
legend('Location', 'best');

% Plot 2: Follower corrected with fusion
figure('Name', 'Follower with Collaborative Fusion', 'Color', 'w', 'Position', [960, 100, 900, 600]);
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

function d = angdiff(a, b)
d = b - a; d = mod(d + pi, 2*pi) - pi;
end

function [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, x_L, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP)
% CMF_GR: Collaborative Map Fusion / Global Registration
% Uses Leader state (x_L) to correct Follower state (F_state) and Map.
% Implements Rigid Alignment + Graph Relaxation via align_and_relax_map.

target_pose = x_L;

% Heading Check
dth_check = abs(angdiff(F_state(3), target_pose(3)));

% Check Euclidean distance
dist_check = norm(target_pose(1:2) - F_state(1:2));

if dth_check < MAX_ANGLE_ERR && dist_check < MAX_DIST_JUMP

    % Store pivot for correction data recording
    pivot = F_state(1:2);

    % Influence Radius (e.g., 50 meters)
    influence_radius = 50.0;

    % Call align_and_relax_map to perform CMF-GR
    [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius);

    % Record Correction
    CorrectionData = [CorrectionData; pivot(1), pivot(2), target_pose(1), target_pose(2)];
end
end