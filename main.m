clear; clc; close all;

%% FEATURE EXTRACTION (Optional)
extractFeatures = false;
POSES = {'00', '01', '02', '03', '04', '05', '06', '07', '08', '09'};
if extractFeatures
    for i = 1:length(POSES)
        sequence_path = fullfile('Dataset', 'sequences', POSES{i}, 'image_0');
        output_filename = strcat('kitti_', POSES{i}, '_features.mat');
        preprocess_kitti_features(sequence_path, output_filename);
    end
end

seq_id = '00';
filename_gt = fullfile('Dataset', 'poses', strcat(seq_id, '.txt'));
feature_file = fullfile('Dataset', 'features', strcat('kitti_', seq_id, '_features.mat'));

%% PARAMETERS

dt = 0.1;
Param.sigma_v = 0.01;
Param.sigma_w = 0.002;
Param.rand_seed = 100;

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
UkfParams.max_sensor_range = 350;
UkfParams.Q = diag([0.01^2, 0.005^2]);
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

%% LOAD DATA

if ~isfile(filename_gt)
    error('Ground truth file not found: %s', filename_gt);
end

fprintf('Loading data...\n');
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
    warning('Features not found. Using geometric fallback.');
    use_visual_features = false;
    AllFeatures = [];
end

min_x = min(GT.x); max_x = max(GT.x);
min_y = min(GT.y); max_y = max(GT.y);

%% INITIALIZATION

N_STEPS = length(GT.x);
N_ROBOTS = 2;
Robots = repmat(struct, N_ROBOTS, 1);

Robots(1).id = 1;
Robots(1).name = 'Leader';
Robots(1).color = 'b';
Robots(1).start_delay = 0;
Robots(1).sigma_v = 0.0;
Robots(1).sigma_w = 0.0;
Robots(1).state = [GT.x(1); GT.y(1); GT.th(1)];
Robots(1).P = eye(3) * 0.1;
Robots(1).Map.Nodes = [];
Robots(1).Map.Count = 0;
Robots(1).Map.last_node_pose = Robots(1).state;
Robots(1).hist.state = zeros(3, N_STEPS);
Robots(1).hist.P = zeros(3, 3, N_STEPS);
Robots(1).hist.nees = nan(1, N_STEPS);

Robots(2).id = 2;
Robots(2).name = 'Follower';
Robots(2).color = 'r';
Robots(2).start_delay = 200;
Robots(2).sigma_v = Param.sigma_v;
Robots(2).sigma_w = Param.sigma_w;
Robots(2).state = [0; 0; 0];
Robots(2).P = eye(3) * 0.1;
Robots(2).Map.Nodes = [];
Robots(2).Map.Count = 0;
Robots(2).Map.last_node_pose = Robots(2).state;
Robots(2).hist.state = zeros(3, N_STEPS);
Robots(2).hist.P = zeros(3, 3, N_STEPS);
Robots(2).hist.nees = nan(1, N_STEPS);

EmptyNode = struct('id', 0, 'pose', [0;0;0], 'view_id', 0, 'source', '');
for r = 1:N_ROBOTS
    Robots(r).Map.Nodes = repmat(EmptyNode, N_STEPS, 1);
end

CorrectionData = [];
LoopClosureStats = zeros(N_ROBOTS, 1);
rng(Param.rand_seed);

%% SIMULATION LOOP

fprintf('\nSTART SIMULATION (%d steps) <<<\n', N_STEPS);
tic;

for k = 1:N_STEPS-1
    for r = 1:N_ROBOTS
        seq_idx = k - Robots(r).start_delay;

        if seq_idx < 1
            continue;
        elseif seq_idx == 1
            Robots(r).state = [GT.x(1); GT.y(1); GT.th(1)];
            Robots(r).Map.last_node_pose = Robots(r).state;
            Robots(r).hist.state(:, k) = Robots(r).state;
            Robots(r).hist.P(:, :, k) = Robots(r).P;
            continue;
        elseif seq_idx >= N_STEPS
            continue;
        end

        dx = GT.x(seq_idx+1) - GT.x(seq_idx);
        dy = GT.y(seq_idx+1) - GT.y(seq_idx);
        dth_true = angdiff(GT.th(seq_idx), GT.th(seq_idx+1));

        v_cmd = sqrt(dx^2 + dy^2);
        w_cmd = dth_true;

        noise_v = randn * Robots(r).sigma_v * dt;
        noise_w = randn * Robots(r).sigma_w * dt;

        v_r = v_cmd + noise_v;
        w_r = w_cmd + noise_w;

        true_pose_next = [GT.x(seq_idx+1); GT.y(seq_idx+1); GT.th(seq_idx+1)];

        [Robots(r).state, Robots(r).P] = ukf(Robots(r).state, Robots(r).P, v_r, w_r, UkfParams);

        [Robots(r).Map, node_added] = update_map_rt(Robots(r).Map, Robots(r).state, seq_idx, MapParams, Robots(r).name);

        e_k = true_pose_next - Robots(r).state;
        e_k(3) = angdiff(Robots(r).state(3), true_pose_next(3));
        try
            nees_k = e_k(1:2)' * inv(Robots(r).P(1:2,1:2)) * e_k(1:2);
            Robots(r).hist.nees(k) = nees_k;
        catch
            Robots(r).hist.nees(k) = NaN;
        end

        Robots(r).hist.state(:, k) = Robots(r).state;
        Robots(r).hist.P(:, :, k) = Robots(r).P;

        if node_added
            for j = 1:N_ROBOTS
                if Robots(j).Map.Count == 0, continue; end


                % Optimization: In a real system, use spatial hashing or KD-tree.
                % Here: Brute force distance check (acceptable for N < 10000)

                valid_nodes = Robots(j).Map.Nodes(1:Robots(j).Map.Count);
                poses_j = [valid_nodes.pose];

                % Calculate distances to current robot state
                dists = sqrt((poses_j(1,:) - Robots(r).state(1)).^2 + ...
                    (poses_j(2,:) - Robots(r).state(2)).^2);

                % Filter by distance threshold
                [vals, sorted_idxs] = sort(dists);
                % INCREASED RADIUS: 15m -> 500m to handle large drift (Dead Reckoning)
                % This allows the robot to find candidates even if its estimated position is far off.
                candidates = sorted_idxs(vals < 100.0);

                % Self-Check Constraint: Exclude recent nodes to avoid trivial matches
                if r == j
                    % Exclude last 200 nodes/frames
                    recent_cutoff = Robots(r).Map.Count - 200;
                    candidates = candidates(candidates < recent_cutoff);
                end

                % Limit candidates to top 3
                if length(candidates) > 3, candidates = candidates(1:3); end

                best_score = 0;
                best_node_idx = -1;

                for c_idx = 1:length(candidates)
                    idx_map_j = candidates(c_idx);
                    cand_node = Robots(j).Map.Nodes(idx_map_j);

                    % Sequence Matching
                    if use_visual_features
                        [is_match, score] = perform_sequence_matching(...
                            seq_idx, cand_node.view_id, AllFeatures, AllFeatures, SeqParams);
                    else
                        % Geometric fallback (if no features)
                        % For simulation without features, we assume match if close enough
                        % But strictly, we should check view_id difference for self-check
                        if r == j && abs(seq_idx - cand_node.view_id) < 50
                            is_match = false; score = 0;
                        else
                            is_match = true; score = 1.0;
                        end
                    end

                    if is_match && score > best_score
                        best_score = score;
                        best_node_idx = idx_map_j;
                    end
                end

                % Rendezvous Event
                if best_node_idx > 0

                    target_pose = Robots(j).Map.Nodes(best_node_idx).pose;

                    % Retrieve Target Covariance (P)
                    % We need P associated with the target node.
                    % Since we store P history by step k, we need the step k corresponding to the node.
                    % Robots(j).Map.Nodes(idx).view_id stores the sequence index (local time).
                    % But Robots(j).hist.P is indexed by Global Step k.
                    % We need to map view_id back to global k?
                    % Actually, Robots(j).hist.P is indexed by k (1..N_STEPS).
                    % And view_id was set to seq_idx (local time).
                    % Wait, seq_idx = k - delay.
                    % So global_k = view_id + delay.

                    target_view_id = Robots(j).Map.Nodes(best_node_idx).view_id;
                    target_global_k = target_view_id + Robots(j).start_delay;

                    if target_global_k > 0 && target_global_k <= N_STEPS
                        target_P = Robots(j).hist.P(:, :, target_global_k);
                    else
                        % Fallback if history not available (should not happen for valid nodes)
                        target_P = eye(3) * 0.1;
                    end

                    % Execute CMF-GR (Map Fusion)
                    % Influence Radius: 50 meters
                    [Robots(r).Map, Robots(r).state, Robots(r).P] = align_and_relax_map(...
                        Robots(r).Map, Robots(r).state, Robots(r).P, ...
                        target_pose, 50.0, target_P);

                    % Store for Visualization
                    CorrectionData = [CorrectionData;
                        Robots(r).state(1), Robots(r).state(2), target_pose(1), target_pose(2)];

                    LoopClosureStats(r) = LoopClosureStats(r) + 1;
                end
            end
        end
    end
    if mod(k, 200) == 0
        fprintf('.');
    end
end

fprintf('\nSimulation completed in %.2f sec.\n', toc);

%% COMPUTE METRICS
fprintf('\n================================================\n');
fprintf('         FINAL ANALYSIS            \n');
fprintf('================================================\n');

for r = 1:N_ROBOTS

    valid_mask = Robots(r).hist.state(1,:) ~= 0;

    % Extract estimated path
    est_x = Robots(r).hist.state(1, valid_mask)';
    est_y = Robots(r).hist.state(2, valid_mask)';

    indices = find(valid_mask);
    gt_indices = indices - Robots(r).start_delay + 1;

    % Filter out-of-bounds GT indices (just in case)
    valid_gt = gt_indices <= length(GT.x) & gt_indices > 0;
    indices = indices(valid_gt);
    gt_indices = gt_indices(valid_gt);

    est_x = Robots(r).hist.state(1, indices)';
    est_y = Robots(r).hist.state(2, indices)';

    gt_x = GT.x(gt_indices);
    gt_y = GT.y(gt_indices);

    err_dist = sqrt((est_x - gt_x).^2 + (est_y - gt_y).^2);
    rmse = rms(err_dist);

    % 2. NEES Calculation
    % Already computed in loop: Robots(r).hist.nees(k)
    nees_vals = Robots(r).hist.nees(indices);
    nees_mean = mean(nees_vals, 'omitnan');
    nees_std = std(nees_vals, 'omitnan');

    fprintf('\nROBOT %d (%s):\n', Robots(r).id, Robots(r).name);
    fprintf('   RMSE Position:       %.3f m\n', rmse);
    fprintf('   NEES (Avg ± Std):    %.2f ± %.2f\n', nees_mean, nees_std);
    fprintf('   Loop Closures:       %d\n', LoopClosureStats(r));
    %
    if nees_mean < 5.99
        fprintf('   ✓ Filter Consistent (NEES < 5.99)\n');
    else
        fprintf('   ⚠ Filter Inconsistent/Overconfident\n');
    end
end
fprintf('================================================\n');

%% PLOT RESULTS

figure('Name', 'Multi-Robot Trajectories', 'Color', 'w', 'Position', [50, 100, 900, 600]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Distributed Multi-Robot System');

% Plot Ground Truth
plot(GT.x, GT.y, 'Color', [0.8 0.8 0.8], 'LineWidth', 4, 'DisplayName', 'Ground Truth');

for r = 1:N_ROBOTS
    % Plot Trajectory
    % Filter out zeros (uninitialized steps)
    valid_idx = Robots(r).hist.state(1,:) ~= 0;
    plot(Robots(r).hist.state(1, valid_idx), Robots(r).hist.state(2, valid_idx), ...
        '-', 'Color', Robots(r).color, 'LineWidth', 2, 'DisplayName', Robots(r).name);

    % Plot Nodes
    if Robots(r).Map.Count > 0
        nodes = Robots(r).Map.Nodes(1:Robots(r).Map.Count);
        poses = [nodes.pose];
        plot(poses(1,:), poses(2,:), 'o', 'Color', Robots(r).color, 'MarkerFaceColor', Robots(r).color, ...
            'MarkerSize', 4, 'HandleVisibility', 'off');
    end
end

legend('Location', 'best');

% NEES Plot
figure('Name', 'NEES', 'Color', 'w', 'Position', [50, 750, 900, 400]);
hold on; grid on;
for r = 1:N_ROBOTS
    plot(1:N_STEPS, Robots(r).hist.nees, '-', 'Color', Robots(r).color, 'DisplayName', [Robots(r).name ' NEES']);
end
xlabel('Global Step k');
ylabel('NEES');
title('NEES Evolution');
legend('Location', 'best');


function [Map, added] = update_map_rt(Map, pose, view_id, Params, source_name)
added = false;
if Map.Count == 0
    Map.Count = 1; Map.Nodes(1).id = 1; Map.Nodes(1).pose = pose;
    Map.Nodes(1).view_id = view_id; Map.Nodes(1).source = source_name;
    Map.last_node_pose = pose; added = true; return;
end
dist = norm(pose(1:2) - Map.last_node_pose(1:2));
dth = abs(angdiff(Map.last_node_pose(3), pose(3)));
if dist > Params.dist_thresh || dth > Params.angle_thresh
    Map.Count = Map.Count + 1; idx = Map.Count;
    Map.Nodes(idx).id = idx; Map.Nodes(idx).pose = pose;
    Map.Nodes(idx).view_id = view_id; Map.Nodes(idx).source = source_name;
    Map.last_node_pose = pose; added = true;
end
end

function d = angdiff(a, b)
d = b - a; d = mod(d + pi, 2*pi) - pi;
end

function [F_Map, F_state, F_P, CorrectionData] = execute_CMF_GR(F_Map, L_Map, F_state, x_L, F_P, L_P, CorrectionData, Correction, MAX_ANGLE_ERR, MAX_DIST_JUMP)

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
    [F_Map, F_state, F_P] = align_and_relax_map(F_Map, F_state, F_P, target_pose, influence_radius, L_P);

    % Record Correction
    CorrectionData = [CorrectionData; pivot(1), pivot(2), target_pose(1), target_pose(2)];
end
end