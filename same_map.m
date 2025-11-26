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
seq_id = '09';
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
F_state = [GT.x(1); GT.y(1); GT.th(1)];
F_P = eye(3) * 0.1; % Initialize Follower Covariance
F_Map.Nodes = [];
F_Map.last_node_pose = F_state;
F_drift_state = [GT.x(1); GT.y(1); GT.th(1)]; % odometry

L_hist = zeros(3, N);
F_hist = zeros(3, N);
F_drift_hist = zeros(3, N);

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

    % 1. Aggiornamento Locale LEADER (ex run_leader_ekf)
    % Il Leader stima dove si trova basandosi SOLO sui SUOI sensori
    [L_state_pred, L_P_pred] = ukf_predict_step(L_state, L_P, v_cmd, w_cmd, UkfParams.Q, UkfParams);
    true_pose_k1 = [GT.x(k+1); GT.y(k+1); GT.th(k+1)];
    [L_state, L_P] = ukf_update_step(L_state_pred, L_P_pred, true_pose_k1, Landmarks, UkfParams);
    [L_Map, L_node_added] = update_map_rt(L_Map, L_state, current_view_id, MapParams);

    % 2. Aggiornamento Locale FOLLOWER
    % Il Follower stima dove si trova basandosi SOLO sui SUOI sensori
    % Generate noisy control for Follower
    noise_v = randn * Param.sigma_v * dt;
    noise_w = randn * Param.sigma_w * dt;
    v_F = v_cmd + noise_v;
    w_F = w_cmd + noise_w;

    % UKF for Follower (Prediction + Update)
    [F_state_pred, F_P_pred] = ukf_predict_step(F_state, F_P, v_F, w_F, UkfParams.Q, UkfParams);
    % Assuming Follower also sees landmarks (based on "z_F_sensors" in pseudocode)
    % If Follower is blind, we skip update or pass empty landmarks.
    % For now, let's assume it sees landmarks like Leader but from its own perspective (which is GT path effectively in this simulation)
    % No, measurements are consistent with TRUE pose (GT).
    [F_state, F_P] = ukf_update_step(F_state_pred, F_P_pred, true_pose_k1, Landmarks, UkfParams);

    [F_Map, F_node_added] = update_map_rt(F_Map, F_state, current_view_id, MapParams);

    % UKF for Follower (Pure - No CMF)
    [F_pure_state_pred, F_pure_P_pred] = ukf_predict_step(F_pure_state, F_pure_P, v_F, w_F, UkfParams.Q, UkfParams);
    [F_pure_state, F_pure_P] = ukf_update_step(F_pure_state_pred, F_pure_P_pred, true_pose_k1, Landmarks, UkfParams);

    % Pure Odometry for comparison (Drift)
    F_drift_state(1) = F_drift_state(1) + v_F * cos(F_drift_state(3));
    F_drift_state(2) = F_drift_state(2) + v_F * sin(F_drift_state(3));
    F_drift_state(3) = F_drift_state(3) + w_F;
    F_drift_hist(:, k) = F_drift_state;

    L_hist(:, k) = L_state;
    F_hist(:, k) = F_state;
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
F_hist(:, end) = F_state;
F_pure_hist(:, end) = F_pure_state;
F_drift_hist(:, end) = F_drift_state;

fprintf('\nSimulation completed in %.2f sec.\n', toc);

%% COMPUTE ERROR RMSE

err_L_EKF = sqrt((L_hist(1,:)' - GT.x).^2 + (L_hist(2,:)' - GT.y).^2);
err_F_Drift = sqrt((F_drift_hist(1,:)' - GT.x).^2 + (F_drift_hist(2,:)' - GT.y).^2);
err_F_Corr = sqrt((F_hist(1,:)' - GT.x).^2 + (F_hist(2,:)' - GT.y).^2);

rmse_L = rms(err_L_EKF);
rmse_F_drift = rms(err_F_Drift);
rmse_F_corr = rms(err_F_Corr);

err_F_Pure = sqrt((F_pure_hist(1,:)' - GT.x).^2 + (F_pure_hist(2,:)' - GT.y).^2);
rmse_F_pure = rms(err_F_Pure);

fprintf('\n------------------------------------------------\n');
fprintf('       FINAL ANALYSIS            \n');
fprintf('------------------------------------------------\n');
fprintf('RMSE Leader (UKF vs GT):       %.3f m\n', rmse_L);
fprintf('RMSE Follower (NOT CORR):       %.3f m\n', rmse_F_drift);
fprintf('RMSE Follower (UKF ONLY):       %.3f m\n', rmse_F_pure);
fprintf('RMSE Follower (CORRECT):      %.3f m\n', rmse_F_corr);
fprintf('------------------------------------------------\n');
fprintf('IMPROVEMENT:        %.1f %%\n', ...
    (rmse_F_drift - rmse_F_corr)/rmse_F_drift * 100);
fprintf('------------------------------------------------\n');

%% PLOT

% ODOMETRY VS GT
figure('Name', 'Drift vs Ground Truth', 'Color', 'w', 'Position', [50, 100, 900, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Trajectories with and without corrections');
plot(GT.x, GT.y, 'Color', [0.7 0.7 0.7], 'LineWidth', 4, 'DisplayName', 'Ground Truth');
plot(Landmarks(:,1), Landmarks(:,2), 'ks', 'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', 'Landmarks');
plot(L_hist(1,:), L_hist(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Leader (UKF Estimate)');
plot(F_drift_hist(1,:), F_drift_hist(2,:), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Follower (Odometry)');
legend('Location', 'best');

% Collaborative correction
figure('Name', 'Collaborative correction', 'Color', 'w', 'Position', [960, 100, 900, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title(sprintf('Collaborative correction Score > %.2f)', SeqParams.match_thresh));
plot(GT.x, GT.y, 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot(F_pure_hist(1,:), F_pure_hist(2,:), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Follower (UKF Only)');
h_fcorr = plot(F_hist(1,:), F_hist(2,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Follower (Collaborative SLAM)');
plot(L_hist(1,:), L_hist(2,:), 'b:', 'LineWidth', 1, 'DisplayName', 'Leader Map (Target)');
if ~isempty(CorrectionData)
    X_links = [CorrectionData(:,1), CorrectionData(:,3), nan(size(CorrectionData,1), 1)]';
    Y_links = [CorrectionData(:,2), CorrectionData(:,4), nan(size(CorrectionData,1), 1)]';
    plot(X_links(:), Y_links(:), 'g-', 'LineWidth', 1.0, 'DisplayName', 'Correction Links (Snap)');
    plot(CorrectionData(:,3), CorrectionData(:,4), 'g.', 'MarkerSize', 8, 'HandleVisibility', 'off');
end
% 7. (Opzionale) Unisci i nodi di Map_B in Map_A per la visualizzazione globale
% We visualize both maps/trajectories on the same plot to show the alignment.
legend([h_fcorr, findobj(gca, 'DisplayName', 'Correction Links (Snap)'), findobj(gca, 'DisplayName', 'Leader Map (Target)')], 'Location', 'best');


%% HELPER FUNCTIONS
function [x_next, P_next] = ukf_predict_step(x, P, v, w, Q, Params)
n = Params.n;
lambda = Params.lambda;
c = Params.c;
Wm = Params.Wm;
Wc = Params.Wc;

% Generate Sigma Points
P = (P + P') / 2;
try
    S = chol(P + 1e-9*eye(n), 'lower');
catch
    [U, Val] = eig(P);
    S = U * sqrt(max(Val, 0)) * U';
end

X = zeros(n, 2*n+1);
X(:, 1) = x;
for i = 1:n
    X(:, i+1) = x + c * S(:, i);
    X(:, i+1+n) = x - c * S(:, i);
end

% Propagate Sigma Points
X_pred = zeros(n, 2*n+1);
for i = 1:2*n+1
    theta = X(3, i);
    X_pred(1, i) = X(1, i) + v * cos(theta);
    X_pred(2, i) = X(2, i) + v * sin(theta);
    X_pred(3, i) = X(3, i) + w;
end

% Compute Predicted Mean
x_next = zeros(n, 1);
for i = 1:2*n+1
    x_next = x_next + Wm(i) * X_pred(:, i);
end
sin_sum = 0; cos_sum = 0;
for i = 1:2*n+1
    sin_sum = sin_sum + Wm(i) * sin(X_pred(3, i));
    cos_sum = cos_sum + Wm(i) * cos(X_pred(3, i));
end
x_next(3) = atan2(sin_sum, cos_sum);

% Compute Predicted Covariance
P_next = zeros(n, n);
for i = 1:2*n+1
    diff = X_pred(:, i) - x_next;
    diff(3) = angdiff(x_next(3), X_pred(3, i));
    P_next = P_next + Wc(i) * (diff * diff');
end

% Add Process Noise
theta_pred = x_next(3);
G = [cos(theta_pred), 0; sin(theta_pred), 0; 0, 1];
P_next = P_next + G * Q * G';
end

function [x_upd, P_upd] = ukf_update_step(x, P, true_pose, Landmarks, Params)
n = Params.n;
c = Params.c;
Wm = Params.Wm;
Wc = Params.Wc;
R = Params.R;

x_upd = x; P_upd = P;

% Re-generate Sigma Points
P = (P + P') / 2;
try
    S = chol(P + 1e-9*eye(n), 'lower');
catch
    [U, Val] = eig(P);
    S = U * sqrt(max(Val, 0)) * U';
end

X_sigma = zeros(n, 2*n+1);
X_sigma(:, 1) = x;
for i = 1:n
    X_sigma(:, i+1) = x + c * S(:, i);
    X_sigma(:, i+1+n) = x - c * S(:, i);
end

for i = 1:size(Landmarks, 1)
    lm = Landmarks(i,:)';
    dist_true = norm(lm - true_pose(1:2));

    if dist_true < Params.max_sensor_range
        % Generate Measurement
        z_r = dist_true + randn * Params.sigma_range;
        z_b = angdiff(true_pose(3), atan2(lm(2)-true_pose(2), lm(1)-true_pose(1))) + randn * Params.sigma_bearing;
        z = [z_r; z_b];

        % Predict Measurements for each Sigma Point
        Z_sigma = zeros(2, 2*n+1);
        for j = 1:2*n+1
            dx_s = lm(1) - X_sigma(1, j);
            dy_s = lm(2) - X_sigma(2, j);
            Z_sigma(1, j) = sqrt(dx_s^2 + dy_s^2);
            Z_sigma(2, j) = angdiff(X_sigma(3, j), atan2(dy_s, dx_s));
        end

        % Predicted Measurement Mean
        z_mean = zeros(2, 1);
        for j = 1:2*n+1
            z_mean = z_mean + Wm(j) * Z_sigma(:, j);
        end
        sin_sum_z = 0; cos_sum_z = 0;
        for j = 1:2*n+1
            sin_sum_z = sin_sum_z + Wm(j) * sin(Z_sigma(2, j));
            cos_sum_z = cos_sum_z + Wm(j) * cos(Z_sigma(2, j));
        end
        z_mean(2) = atan2(sin_sum_z, cos_sum_z);

        % Predicted Measurement Covariance and Cross-Covariance
        S_z = zeros(2, 2);
        P_xz = zeros(n, 2);

        for j = 1:2*n+1
            z_diff = Z_sigma(:, j) - z_mean;
            z_diff(2) = angdiff(z_mean(2), Z_sigma(2, j));

            x_diff = X_sigma(:, j) - x_upd; % Note: x_upd is mean
            x_diff(3) = angdiff(x_upd(3), X_sigma(3, j));

            S_z = S_z + Wc(j) * (z_diff * z_diff');
            P_xz = P_xz + Wc(j) * (x_diff * z_diff');
        end

        S_z = S_z + R;

        % Kalman Gain
        K = P_xz / S_z;

        % Update State and Covariance
        innovation = z - z_mean;
        innovation(2) = angdiff(z_mean(2), z(2));

        x_upd = x_upd + K * innovation;
        x_upd(3) = angdiff(0, x_upd(3));

        P_upd = P_upd - K * S_z * K';

        % Re-generate Sigma Points for next landmark
        P_upd = (P_upd + P_upd') / 2;
        try
            S = chol(P_upd + 1e-9*eye(n), 'lower');
        catch
            [U, Val] = eig(P_upd);
            S = U * sqrt(max(Val, 0)) * U';
        end
        X_sigma(:, 1) = x_upd;
        for j = 1:n
            X_sigma(:, j+1) = x_upd + c * S(:, j);
            X_sigma(:, j+1+n) = x_upd - c * S(:, j);
        end
    end
end
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

function d = angdiff(a, b)
d = b - a; d = mod(d + pi, 2*pi) - pi;
end

function [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, x_L, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP)
% CMF_GR: Collaborative Map Fusion / Global Registration
% Uses Leader state (x_L) to correct Follower state (F_state) and Map.
% Implements Rigid Alignment + Graph Relaxation.

target_pose = x_L;

% Heading Check
dth_check = abs(angdiff(F_state(3), target_pose(3)));

% Check Euclidean distance
dist_check = norm(target_pose(1:2) - F_state(1:2));

if dth_check < MAX_ANGLE_ERR && dist_check < MAX_DIST_JUMP

    % --- STEP 4: Calcola errore relativo Pose_B vs Pose_A_matched ---
    % Calculate the transform that maps F_state to x_L
    % T_LF = x_L (-) F_state
    % We want a transform [dx, dy, dtheta] such that applying it to F_state yields x_L.
    % Ideally, we should rotate F_state to align orientation, then translate.

    dth = angdiff(F_state(3), target_pose(3));

    % --- STEP 5: Calcola T_alignment ---
    % Rotation matrix of the correction
    R_corr = [cos(dth), -sin(dth); sin(dth), cos(dth)];

    % Translation correction
    % If we rotate F_state(1:2) around itself, position doesn't change.
    % But we want to align the frames.
    % Let's assume the correction is a rigid body transform of the whole map.
    % The pivot point is the current F_state (rendezvous).
    % We rotate the map around F_state, then translate to match x_L.

    pivot = F_state(1:2);
    translation = target_pose(1:2) - F_state(1:2);

    % --- STEP 6: Esegui CMF_Graph_Relaxation (Map_B, Map_A, T_alignment) ---
    % Instead of applying T rigidly to everything, we apply it with a weight.
    % Weight = 1 at rendezvous (current node).
    % Weight decays with distance (graph distance or spatial distance).

    % Influence Radius (e.g., 50 meters or 50 nodes)
    influence_radius = 50.0;

    % Update Current State (Weight = 1)
    F_state(3) = angdiff(0, F_state(3) + dth);
    F_state(1:2) = F_state(1:2) + translation;

    % Update Map Nodes
    if F_Map.Count > 0
        for i = 1:F_Map.Count
            node_pose = F_Map.Nodes(i).pose;

            % Distance from pivot (rendezvous)
            dist = norm(node_pose(1:2) - pivot);

            if dist < influence_radius
                % Weight Function (Linear Decay)
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

    % Record Correction
    CorrectionData = [CorrectionData; pivot(1), pivot(2), target_pose(1), target_pose(2)];

    % Update Covariance
    % Reduce uncertainty as we have fused with Leader
    F_P = F_P * 0.5;
end
end