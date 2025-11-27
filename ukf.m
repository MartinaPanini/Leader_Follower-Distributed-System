function [x_next, P_next] = ukf(x, P, v, w, Params)
% UKF - State Propagator (Dead Reckoning Only)
%
% Implements UKF Prediction step only. No measurement update.
%
% Inputs:
%   x         - Current state [x; y; theta] (3x1)
%   P         - Current covariance matrix (3x3)
%   v         - Linear velocity control input
%   w         - Angular velocity control input
%   Params    - UKF parameters structure
%
% Outputs:
%   x_next - Updated state estimate (3x1)
%   P_next - Updated covariance matrix (3x3)

%% PREDICTION STEP
[x_next, P_next] = ukf_predict(x, P, v, w, Params);

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

function d = angdiff(a, b)
% Compute angular difference (b - a) wrapped to [-pi, pi]
d = b - a;
d = mod(d + pi, 2*pi) - pi;
end
