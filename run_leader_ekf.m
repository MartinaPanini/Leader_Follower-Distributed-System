function [est_x, est_y, est_th, P_history] = run_leader_ekf(gt_x, gt_y, gt_th, Landmarks, params)
    
    % Parametri di configurazione
    Q = diag([params.sigma_v^2, params.sigma_w^2]); % Covarianza Processo (v, omega)
    R = diag([params.sigma_range^2, params.sigma_bearing^2]); % Covarianza Misura (range, bearing)
    dt = 1; % Assumiamo step unitario tra i frame del dataset KITTI per semplicità

    N = length(gt_x);
    
    % Inizializzazione Stato Stimato (x_hat) e Covarianza (P)
    x_est = zeros(3, N);
    x_est(:,1) = [gt_x(1); gt_y(1); gt_th(1)]; % Condizione iniziale nota
    
    P = eye(3) * 0.1; % Incertezza iniziale piccola
    P_history = zeros(3, 3, N);
    P_history(:,:,1) = P;

    % Loop temporale
    for k = 1:N-1
        
        %% 1. PREDICTION STEP (Propriocezione / Odometria)
        % Calcoliamo l'input di controllo u(k) basandoci sulla Ground Truth
        % Simuliamo che il robot "misuri" quanto si è mosso
        
        % Calcolo velocità lineare (v) e angolare (w) ideali dallo step k al k+1
        dx = gt_x(k+1) - gt_x(k);
        dy = gt_y(k+1) - gt_y(k);
        dth = angdiff(gt_th(k), gt_th(k+1));
        
        dist = sqrt(dx^2 + dy^2);
        
        % Aggiungiamo rumore al controllo (simula errore encoder/IMU)
        v_noisy = dist + randn * params.sigma_scale_odom * dist; 
        w_noisy = dth + randn * params.sigma_rot_odom;
        
        % Stato precedente
        theta_k = x_est(3, k);
        
        % Modello Cinematico (f(x,u))
        % x(k+1) = x(k) + v*cos(theta)*dt
        pred_x = x_est(1, k) + v_noisy * cos(theta_k);
        pred_y = x_est(2, k) + v_noisy * sin(theta_k);
        pred_th = x_est(3, k) + w_noisy;
        
        % Jacobiano del modello di movimento (F_k) rispetto allo stato
        % df/dx = [1 0 -v*sin(th); 0 1 v*cos(th); 0 0 1]
        F = eye(3);
        F(1,3) = -v_noisy * sin(theta_k);
        F(2,3) =  v_noisy * cos(theta_k);
        
        % Jacobiano del modello rispetto al rumore (G_k) approssimato
        % Mappiamo il rumore di v e w sullo stato
        G = [cos(theta_k), 0;
             sin(theta_k), 0;
             0,            1];
         
        % Predizione della Covarianza (P_k-)
        % P- = F*P*F' + G*Q*G'
        P_pred = F * P * F' + G * Q * G';
        
        % Stato predetto (A priori)
        x_k_minus = [pred_x; pred_y; pred_th];
        
        %% 2. UPDATE STEP (Esterocezione / Landmarks)
        % Controlla se ci sono landmark visibili
        
        % Posizione reale attuale (per simulare la misura del sensore)
        true_pose = [gt_x(k+1); gt_y(k+1); gt_th(k+1)];
        
        measured = false;
        
        for i = 1:size(Landmarks, 1)
            lm_x = Landmarks(i, 1);
            lm_y = Landmarks(i, 2);
            
            % Distanza reale dal robot al landmark
            real_dist = sqrt((true_pose(1) - lm_x)^2 + (true_pose(2) - lm_y)^2);
            
            % Se il landmark è nel raggio del sensore
            if real_dist < params.max_sensor_range
                measured = true;
                
                % --- Generazione Misura Simulata (z) ---
                % z = h(x_true) + noise
                real_bearing = atan2(lm_y - true_pose(2), lm_x - true_pose(1)) - true_pose(3);
                
                z_dist = real_dist + randn * params.sigma_range;
                z_bearing = angdiff(0, real_bearing) + randn * params.sigma_bearing;
                z = [z_dist; z_bearing];
                
                % --- Predizione Misura (h(x_minus)) ---
                dx_est = lm_x - x_k_minus(1);
                dy_est = lm_y - x_k_minus(2);
                range_est = sqrt(dx_est^2 + dy_est^2);
                bearing_est = angdiff(x_k_minus(3), atan2(dy_est, dx_est));
                
                z_hat = [range_est; bearing_est];
                
                % --- Jacobiano della Misura (H_k) ---
                % Derivate parziali di range e bearing rispetto a x, y, theta
                % r2 = dx^2 + dy^2
                r2 = range_est^2;
                H = [ -(dx_est)/range_est, -(dy_est)/range_est, 0;
                       (dy_est)/r2,       -(dx_est)/r2,       -1 ];
                   
                % --- Kalman Gain (W_k) ---
                % S = H*P*H' + R
                S = H * P_pred * H' + R;
                W = P_pred * H' / S;  % W = P*H'*inv(S)
                
                % --- Aggiornamento Stato ---
                innovation = z - z_hat;
                innovation(2) = angdiff(0, innovation(2)); % Normalizza angolo
                
                x_k_minus = x_k_minus + W * innovation;
                
                % --- Aggiornamento Covarianza ---
                % P = (I - W*H) * P_pred
                P_pred = (eye(3) - W * H) * P_pred;
            end
        end
        
        % Salva i risultati per il passo successivo
        x_est(:, k+1) = x_k_minus;
        P = P_pred;
        P_history(:,:,k+1) = P;
    end
    
    % Estrai output
    est_x = x_est(1, :)';
    est_y = x_est(2, :)';
    est_th = x_est(3, :)';
end