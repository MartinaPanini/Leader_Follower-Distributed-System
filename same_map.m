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
seq_id = ['00']; 
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

% EKF Leader
EkfParams.sigma_scale_odom = 0.05; 
EkfParams.sigma_rot_odom   = 0.002; 
EkfParams.sigma_range      = 0.1;       
EkfParams.sigma_bearing    = 0.01;    
EkfParams.max_sensor_range = 150;
EkfParams.Q = diag([0.05^2, 0.01^2]); 
EkfParams.R = diag([EkfParams.sigma_range^2, EkfParams.sigma_bearing^2]);

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
F_Map.Nodes = [];
F_Map.last_node_pose = F_state;
F_drift_state = [GT.x(1); GT.y(1); GT.th(1)]; % odometry

L_hist = zeros(3, N);
F_hist = zeros(3, N);       
F_drift_hist = zeros(3, N); 

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
    
    % leader EKF
    [L_state_pred, L_P_pred] = ekf_predict_step(L_state, L_P, v_cmd, w_cmd, EkfParams.Q);
    true_pose_k1 = [GT.x(k+1); GT.y(k+1); GT.th(k+1)];
    [L_state, L_P] = ekf_update_step(L_state_pred, L_P_pred, true_pose_k1, Landmarks, EkfParams);
    [L_Map, L_node_added] = update_map_rt(L_Map, L_state, current_view_id, MapParams);
    
    % Follower odometry
    noise_v = randn * Param.sigma_v * dt;
    noise_w = randn * Param.sigma_w * dt;
    
    v_F = v_cmd + noise_v; 
    w_F = w_cmd + noise_w;
 
    F_state(1) = F_state(1) + v_F * cos(F_state(3));
    F_state(2) = F_state(2) + v_F * sin(F_state(3));
    F_state(3) = F_state(3) + w_F;
    
    F_drift_state(1) = F_drift_state(1) + v_F * cos(F_drift_state(3));
    F_drift_state(2) = F_drift_state(2) + v_F * sin(F_drift_state(3));
    F_drift_state(3) = F_drift_state(3) + w_F;

    [F_Map, F_node_added] = update_map_rt(F_Map, F_state, current_view_id, MapParams);

    L_hist(:, k) = L_state;
    F_hist(:, k) = F_state;
    F_drift_hist(:, k) = F_drift_state;
    
    % Sequence Matching
    if F_node_added && L_Map.Count > 0
        
        valid_L_nodes = L_Map.Nodes(1:L_Map.Count);
        Ln_coords = reshape([valid_L_nodes.pose], 3, [])';
        dists = sqrt((Ln_coords(:,1) - F_state(1)).^2 + (Ln_coords(:,2) - F_state(2)).^2);
        
        [vals, sorted_idxs] = sort(dists);
        valid_mask = vals < 15.0;
        candidate_indices = sorted_idxs(valid_mask);
        if length(candidate_indices) > 3, candidate_indices = candidate_indices(1:3); end
        
        best_score = 0; best_L_idx = -1;
        
        for i = 1:length(candidate_indices)
            idx_L = candidate_indices(i);
            cand_view_id = L_Map.Nodes(idx_L).view_id;
            
            if use_visual_features
                [is_match, score] = perform_sequence_matching(...
                    current_view_id, cand_view_id, AllFeatures, AllFeatures, SeqParams);
            else
                is_match = abs(current_view_id - cand_view_id) < 5; score = 1.0;
            end
            
            if is_match && score > best_score
                best_score = score; best_L_idx = idx_L;
            end
        end
        
        if best_L_idx > 0
            target_pose = L_Map.Nodes(best_L_idx).pose;
            % Heading Check
            dth_check = abs(angdiff(F_state(3), target_pose(3)));
            
            % Check Mahalanobis distance
            dist_check = norm(target_pose(1:2) - F_state(1:2));     
            
            if dth_check < MAX_ANGLE_ERR && dist_check < MAX_DIST_JUMP
                
                % --- SE IL CHECK PASSA, APPLICA LA CORREZIONE ---
                
                old_pos = F_state(1:2);
                
                CorrectionData = [CorrectionData; old_pos(1), old_pos(2), target_pose(1), target_pose(2)];
                
                dx = target_pose(1) - F_state(1);
                dy = target_pose(2) - F_state(2);
                dth = angdiff(F_state(3), target_pose(3));
                
                F_state(1) = F_state(1) + Correction.gain * dx;
                F_state(2) = F_state(2) + Correction.gain * dy;
                F_state(3) = F_state(3) + Correction.gain * dth;
                
                % Aggiorna mappa e storia post-correzione
                F_Map.Nodes(F_Map.Count).pose = F_state; 
                F_Map.last_node_pose = F_state;
                F_hist(:, k) = F_state; 
        end
    end
    
    if mod(k, 200) == 0, fprintf('.'); end
    end
end
L_hist(:, end) = L_state;
F_hist(:, end) = F_state;
F_drift_hist(:, end) = F_drift_state;

fprintf('\nSimulation completed in %.2f sec.\n', toc);

%% COMPUTE ERROR RMSE

err_L_EKF = sqrt((L_hist(1,:)' - GT.x).^2 + (L_hist(2,:)' - GT.y).^2);
err_F_Drift = sqrt((F_drift_hist(1,:)' - GT.x).^2 + (F_drift_hist(2,:)' - GT.y).^2);
err_F_Corr = sqrt((F_hist(1,:)' - GT.x).^2 + (F_hist(2,:)' - GT.y).^2);

rmse_L = rms(err_L_EKF);
rmse_F_drift = rms(err_F_Drift);
rmse_F_corr = rms(err_F_Corr);

fprintf('\n------------------------------------------------\n');
fprintf('       FINAL ANALYSIS            \n');
fprintf('------------------------------------------------\n');
fprintf('RMSE Leader (EKF vs GT):       %.3f m\n', rmse_L);
fprintf('RMSE Follower (NOT CORR):       %.3f m\n', rmse_F_drift);
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
plot(L_hist(1,:), L_hist(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Leader (EKF Estimate)');
plot(F_drift_hist(1,:), F_drift_hist(2,:), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Follower (Odometry)');
legend('Location', 'best');

% Collaborative correction
figure('Name', 'Collaborative correction', 'Color', 'w', 'Position', [960, 100, 900, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title(sprintf('Collaborative correction Score > %.2f)', SeqParams.match_thresh));
plot(GT.x, GT.y, 'Color', [0.8 0.8 0.8], 'LineWidth', 3, 'DisplayName', 'Ground Truth');
h_fcorr = plot(F_hist(1,:), F_hist(2,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Follower (Collaborative SLAM)');
plot(L_hist(1,:), L_hist(2,:), 'b:', 'LineWidth', 1, 'DisplayName', 'Leader Map (Target)');
if ~isempty(CorrectionData)
    X_links = [CorrectionData(:,1), CorrectionData(:,3), nan(size(CorrectionData,1), 1)]';
    Y_links = [CorrectionData(:,2), CorrectionData(:,4), nan(size(CorrectionData,1), 1)]';
    plot(X_links(:), Y_links(:), 'g-', 'LineWidth', 1.0, 'DisplayName', 'Correction Links (Snap)');
    plot(CorrectionData(:,3), CorrectionData(:,4), 'g.', 'MarkerSize', 8, 'HandleVisibility', 'off');
end
legend([h_fcorr, findobj(gca, 'DisplayName', 'Correction Links (Snap)'), findobj(gca, 'DisplayName', 'Leader Map (Target)')], 'Location', 'best');


%% HELPER FUNCTIONS 
function [x_next, P_next] = ekf_predict_step(x, P, v, w, Q)
    theta = x(3);
    F = [1, 0, -v*sin(theta); 0, 1, v*cos(theta); 0, 0, 1];
    G = [cos(theta), 0; sin(theta), 0; 0, 1];
    x_next = x + [v*cos(theta); v*sin(theta); w];
    P_next = F*P*F' + G*Q*G';
end

function [x_upd, P_upd] = ekf_update_step(x, P, true_pose, Landmarks, Params)
    x_upd = x; P_upd = P;
    for i = 1:size(Landmarks, 1)
        lm = Landmarks(i,:)';
        dist_true = norm(lm - true_pose(1:2));
        if dist_true < Params.max_sensor_range
            z_r = dist_true + randn * Params.sigma_range;
            z_b = angdiff(true_pose(3), atan2(lm(2)-true_pose(2), lm(1)-true_pose(1))) + randn * Params.sigma_bearing;
            z = [z_r; z_b];
            dx = lm(1) - x(1); dy = lm(2) - x(2);
            range = sqrt(dx^2 + dy^2); bearing = angdiff(x(3), atan2(dy, dx));
            z_hat = [range; bearing];
            H = [-dx/range, -dy/range, 0; dy/range^2, -dx/range^2, -1];
            S = H*P*H' + Params.R; K = P*H' / S;
            inn = z - z_hat; inn(2) = angdiff(0, inn(2));
            x_upd = x_upd + K * inn; P_upd = (eye(3) - K*H) * P_upd;
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