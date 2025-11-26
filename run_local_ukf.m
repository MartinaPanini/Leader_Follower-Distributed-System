function [x_next, P_next] = run_local_ukf(x, P, v, w, true_pose, Landmarks, Params)
% RUN_LOCAL_UKF - Unscented Kalman Filter for robot localization
%
% Implements UKF with Unscented Transform for prediction and update steps.
% This function performs one complete prediction + update cycle.
%
% Inputs:
%   x         - Current state [x; y; theta] (3x1)
%   P         - Current covariance matrix (3x3)
%   v         - Linear velocity control input
%   w         - Angular velocity control input
%   true_pose - True pose for generating measurements [x; y; theta] (3x1)
%   Landmarks - Landmark positions (Nx2 matrix)
%   Params    - UKF parameters structure containing:
%               .n, .lambda, .c, .Wm, .Wc, .Q, .R, .max_sensor_range,
%               .sigma_range, .sigma_bearing
%
% Outputs:
%   x_next - Updated state estimate (3x1)
%   P_next - Updated covariance matrix (3x3)

%% PREDICTION STEP
[x_pred, P_pred] = ukf_predict(x, P, v, w, Params);

%% UPDATE STEP
[x_next, P_next] = ukf_update(x_pred, P_pred, true_pose, Landmarks, Params);

end

%% HELPER FUNCTIONS

function [x_next, P_next] = ukf_predict(x, P, v, w, Params)
% UKF Prediction step using Unscented Transform

n = Params.n;
lambda = Params.lambda;
c = Params.c;
Wm = Params.Wm;
Wc = Params.Wc;
Q = Params.Q;

% Generate Sigma Points
P = (P + P') / 2;  % Ensure symmetry
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

% Propagate Sigma Points through Motion Model
X_pred = zeros(n, 2*n+1);
for i = 1:2*n+1
    theta = X(3, i);
    X_pred(1, i) = X(1, i) + v * cos(theta);
    X_pred(2, i) = X(2, i) + v * sin(theta);
    X_pred(3, i) = X(3, i) + w;
end

% Compute Predicted Mean (with angle wrapping)
x_next = zeros(n, 1);
for i = 1:2*n+1
    x_next = x_next + Wm(i) * X_pred(:, i);
end
% Angle averaging using circular statistics
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

function [x_upd, P_upd] = ukf_update(x, P, true_pose, Landmarks, Params)
% UKF Update step using Unscented Transform

n = Params.n;
c = Params.c;
Wm = Params.Wm;
Wc = Params.Wc;
R = Params.R;

x_upd = x;
P_upd = P;

% Re-generate Sigma Points from predicted state
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

% Process each landmark
for i = 1:size(Landmarks, 1)
    lm = Landmarks(i,:)';
    dist_true = norm(lm - true_pose(1:2));

    if dist_true < Params.max_sensor_range
        % Generate noisy measurement
        z_r = dist_true + randn * Params.sigma_range;
        z_b = angdiff(true_pose(3), atan2(lm(2)-true_pose(2), lm(1)-true_pose(1))) + randn * Params.sigma_bearing;
        z = [z_r; z_b];

        % Predict measurements for each Sigma Point
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

        % Measurement Covariance and Cross-Covariance
        S_z = zeros(2, 2);
        P_xz = zeros(n, 2);

        for j = 1:2*n+1
            z_diff = Z_sigma(:, j) - z_mean;
            z_diff(2) = angdiff(z_mean(2), Z_sigma(2, j));

            x_diff = X_sigma(:, j) - x_upd;
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

function d = angdiff(a, b)
% Compute angular difference (b - a) wrapped to [-pi, pi]
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
