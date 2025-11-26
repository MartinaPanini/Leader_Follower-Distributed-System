function [est_x, est_y, est_th, P_history] = run_local_ukf(gt_x, gt_y, gt_th, Landmarks, params)
    
    % Configuration Parameters
    Q = diag([params.sigma_v^2, params.sigma_w^2]); % Process Noise Covariance (v, omega)
    R = diag([params.sigma_range^2, params.sigma_bearing^2]); % Measurement Noise Covariance (range, bearing)
    dt = 1; % Unit time step

    N = length(gt_x);
    
    % UKF Parameters
    n = 3; % State dimension (x, y, theta)
    alpha = 1e-3;
    beta = 2;
    kappa = 0;
    lambda = alpha^2 * (n + kappa) - n;
    
    % Weights
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);
    Wm(1) = lambda / (n + lambda);
    Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);
    Wm(2:end) = 1 / (2 * (n + lambda));
    Wc(2:end) = 1 / (2 * (n + lambda));
    
    % Initialize Estimated State and Covariance
    x_est = zeros(3, N);
    x_est(:,1) = [gt_x(1); gt_y(1); gt_th(1)];
    
    P = eye(3) * 0.1;
    P_history = zeros(3, 3, N);
    P_history(:,:,1) = P;

    % Temporal Loop
    for k = 1:N-1
        
        %% 1. PREDICTION STEP
        % Control Input (Simulated from Ground Truth)
        dx = gt_x(k+1) - gt_x(k);
        dy = gt_y(k+1) - gt_y(k);
        dth = angdiff(gt_th(k), gt_th(k+1));
        
        dist = sqrt(dx^2 + dy^2);
        
        % Noisy Control
        v_noisy = dist + randn * params.sigma_scale_odom * dist; 
        w_noisy = dth + randn * params.sigma_rot_odom;
        
        % Generate Sigma Points
        % P might not be strictly positive definite due to numerical errors, so we ensure symmetry and add small epsilon
        P = (P + P') / 2;
        try
            S = chol(P + 1e-9*eye(n), 'lower');
        catch
            [U, Val] = eig(P);
            S = U * sqrt(max(Val, 0)) * U';
        end
        
        c = sqrt(n + lambda);
        X = zeros(n, 2*n+1);
        x_k = x_est(:, k);
        X(:, 1) = x_k;
        for i = 1:n
            X(:, i+1) = x_k + c * S(:, i);
            X(:, i+1+n) = x_k - c * S(:, i);
        end
        
        % Propagate Sigma Points through Motion Model
        X_pred = zeros(n, 2*n+1);
        for i = 1:2*n+1
            theta = X(3, i);
            X_pred(1, i) = X(1, i) + v_noisy * cos(theta);
            X_pred(2, i) = X(2, i) + v_noisy * sin(theta);
            X_pred(3, i) = X(3, i) + w_noisy;
        end
        
        % Compute Predicted Mean
        x_pred_mean = zeros(n, 1);
        for i = 1:2*n+1
            x_pred_mean = x_pred_mean + Wm(i) * X_pred(:, i);
        end
        % Angle averaging requires care, but for small spread standard average is often used in simple UKF. 
        % However, correct way is sum of sines and cosines or iterative. 
        % For simplicity and consistency with standard UKF implementations for robots:
        % We can re-normalize the angle after averaging or use a more robust mean for angles.
        % Let's use the standard weighted sum but normalize the result.
        % A better approach for angles:
        sin_sum = 0; cos_sum = 0;
        for i = 1:2*n+1
            sin_sum = sin_sum + Wm(i) * sin(X_pred(3, i));
            cos_sum = cos_sum + Wm(i) * cos(X_pred(3, i));
        end
        x_pred_mean(3) = atan2(sin_sum, cos_sum);

        
        % Compute Predicted Covariance
        P_pred = zeros(n, n);
        for i = 1:2*n+1
            diff = X_pred(:, i) - x_pred_mean;
            diff(3) = angdiff(x_pred_mean(3), X_pred(3, i));
            P_pred = P_pred + Wc(i) * (diff * diff');
        end
        
        % Add Process Noise (Approximation: transform Q to state space or add directly if additive)
        % The original EKF mapped noise via Jacobian G. 
        % Here we are treating noise as additive to the state evolution for simplicity in standard UKF formulation,
        % or we should augment the state. 
        % Given the EKF used G*Q*G', it implies noise enters through control.
        % For non-augmented UKF, we usually add projected noise covariance.
        % Let's approximate by adding G*Q*G' evaluated at mean, or just add a diagonal noise representing state uncertainty growth.
        % However, strictly speaking, if noise is non-additive (it is inside cos/sin), we should use Augmented UKF.
        % But the user request asks for "Sigma Points ... around current state", implying non-augmented.
        % So we add the noise covariance term similar to EKF: G*Q*G'.
        theta_pred = x_pred_mean(3);
        G = [cos(theta_pred), 0; sin(theta_pred), 0; 0, 1];
        P_pred = P_pred + G * Q * G';

        
        %% 2. UPDATE STEP
        true_pose = [gt_x(k+1); gt_y(k+1); gt_th(k+1)];
        
        % Re-generate Sigma Points from Predicted State
        P_pred = (P_pred + P_pred') / 2;
        try
            S_pred = chol(P_pred + 1e-9*eye(n), 'lower');
        catch
            [U, Val] = eig(P_pred);
            S_pred = U * sqrt(max(Val, 0)) * U';
        end
        
        X_pred_sigma = zeros(n, 2*n+1);
        X_pred_sigma(:, 1) = x_pred_mean;
        for i = 1:n
            X_pred_sigma(:, i+1) = x_pred_mean + c * S_pred(:, i);
            X_pred_sigma(:, i+1+n) = x_pred_mean - c * S_pred(:, i);
        end
        
        for lm_idx = 1:size(Landmarks, 1)
            lm_x = Landmarks(lm_idx, 1);
            lm_y = Landmarks(lm_idx, 2);
            
            real_dist = sqrt((true_pose(1) - lm_x)^2 + (true_pose(2) - lm_y)^2);
            
            if real_dist < params.max_sensor_range
                % Generate Measurement
                real_bearing = atan2(lm_y - true_pose(2), lm_x - true_pose(1)) - true_pose(3);
                z_dist = real_dist + randn * params.sigma_range;
                z_bearing = angdiff(0, real_bearing) + randn * params.sigma_bearing;
                z = [z_dist; z_bearing];
                
                % Predict Measurements for each Sigma Point
                Z_sigma = zeros(2, 2*n+1);
                for i = 1:2*n+1
                    dx_s = lm_x - X_pred_sigma(1, i);
                    dy_s = lm_y - X_pred_sigma(2, i);
                    Z_sigma(1, i) = sqrt(dx_s^2 + dy_s^2);
                    Z_sigma(2, i) = angdiff(X_pred_sigma(3, i), atan2(dy_s, dx_s));
                end
                
                % Predicted Measurement Mean
                z_mean = zeros(2, 1);
                for i = 1:2*n+1
                    z_mean = z_mean + Wm(i) * Z_sigma(:, i);
                end
                % Angle averaging for bearing
                sin_sum_z = 0; cos_sum_z = 0;
                for i = 1:2*n+1
                    sin_sum_z = sin_sum_z + Wm(i) * sin(Z_sigma(2, i));
                    cos_sum_z = cos_sum_z + Wm(i) * cos(Z_sigma(2, i));
                end
                z_mean(2) = atan2(sin_sum_z, cos_sum_z);
                
                % Predicted Measurement Covariance and Cross-Covariance
                S_z = zeros(2, 2);
                P_xz = zeros(n, 2);
                
                for i = 1:2*n+1
                    z_diff = Z_sigma(:, i) - z_mean;
                    z_diff(2) = angdiff(z_mean(2), Z_sigma(2, i));
                    
                    x_diff = X_pred_sigma(:, i) - x_pred_mean;
                    x_diff(3) = angdiff(x_pred_mean(3), X_pred_sigma(3, i));
                    
                    S_z = S_z + Wc(i) * (z_diff * z_diff');
                    P_xz = P_xz + Wc(i) * (x_diff * z_diff');
                end
                
                S_z = S_z + R;
                
                % Kalman Gain
                K = P_xz / S_z;
                
                % Update State and Covariance
                innovation = z - z_mean;
                innovation(2) = angdiff(z_mean(2), z(2)); % Be careful with sign here. z - z_mean.
                % Actually angdiff(a, b) is b - a. So angdiff(z_mean, z) is z - z_mean wrapped. Correct.
                
                x_pred_mean = x_pred_mean + K * innovation;
                x_pred_mean(3) = angdiff(0, x_pred_mean(3)); % Normalize angle
                
                P_pred = P_pred - K * S_z * K';
                
                % Re-generate Sigma Points for next landmark update (optional but recommended if updates are sequential)
                % Or just keep using the same sigma points? 
                % Standard EKF/UKF updates sequentially. For UKF, ideally we regenerate sigma points after each update 
                % because the distribution has changed.
                P_pred = (P_pred + P_pred') / 2;
                try
                    S_pred = chol(P_pred + 1e-9*eye(n), 'lower');
                catch
                    [U, Val] = eig(P_pred);
                    S_pred = U * sqrt(max(Val, 0)) * U';
                end
                X_pred_sigma(:, 1) = x_pred_mean;
                for i = 1:n
                    X_pred_sigma(:, i+1) = x_pred_mean + c * S_pred(:, i);
                    X_pred_sigma(:, i+1+n) = x_pred_mean - c * S_pred(:, i);
                end
            end
        end
        
        x_est(:, k+1) = x_pred_mean;
        P = P_pred;
        P_history(:,:,k+1) = P;
    end
    
    est_x = x_est(1, :)';
    est_y = x_est(2, :)';
    est_th = x_est(3, :)';
end