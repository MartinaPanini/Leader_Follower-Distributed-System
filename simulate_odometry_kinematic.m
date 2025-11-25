function [ox, oy, oth] = simulate_odometry_kinematic(gx, gy, gth, dt, sigma_v, sigma_w, rand_seed)
% SIMULATE_ODOMETRY_KINEMATIC
% Genera odometria basata sul modello cinematico unicycle.
%
% INPUT:
% gx, gy, gth : Vettori Ground Truth (Leader o Follower)
% dt          : Passo temporale (es. 0.1s o 1s). Se i dati sono sequenziali senza tempo, usa 1.
% sigma_v     : Deviazione standard rumore velocità lineare [m/s] o [m/step]
% sigma_w     : Deviazione standard rumore velocità angolare [rad/s] o [rad/step]
% rand_seed   : Seed per riproducibilità

    % Setup iniziale
    N = length(gx);
    ox = zeros(N, 1); 
    oy = zeros(N, 1); 
    oth = zeros(N, 1);
    
    % Inizializzazione: Il robot parte sapendo dove si trova (o con errore 0)
    ox(1) = gx(1); 
    oy(1) = gy(1); 
    oth(1) = gth(1);
    
    rng(rand_seed); 
    
    for k = 2:N
        %% 1. INVERSE KINEMATICS (Derivare i controlli ideali dalla GT)
        % Calcoliamo quanto il robot si è "realmente" mosso per dedurre v e w.
        
        dx = gx(k) - gx(k-1);
        dy = gy(k) - gy(k-1);
        
        % Velocità lineare ideale (magnitudine dello spostamento / dt)
        % Nota: Assumiamo che il robot si muova prevalentemente in avanti.
        % Se dx,dy sono nel frame globale, la distanza euclidea è corretta per v.
        v_ideal = sqrt(dx^2 + dy^2) / dt;
        
        % Velocità angolare ideale
        dth_true = angdiff(gth(k-1), gth(k));
        w_ideal = dth_true / dt;
        
        %% 2. NOISE INJECTION (Errori sugli attuatori/sensori)
        % Qui l'errore è fisico: il robot crede di andare a v, ma va a v + rumore.
        % Puoi aggiungere anche un termine proporzionale (bias) se vuoi complicare la vita.
        
        v_noisy = v_ideal + randn * sigma_v;
        w_noisy = w_ideal + randn * sigma_w;
        
        %% 3. FORWARD KINEMATICS (Integrazione Unicycle)
        % Modello: 
        % x_new = x_old + v * cos(theta_old) * dt
        % y_new = y_old + v * sin(theta_old) * dt
        % th_new = th_old + w * dt
        
        % Nota cruciale: Usiamo oth(k-1) (l'orientamento STIMATO precedente), 
        % non quello vero. È questo che crea l'accumulo realistico di drift!
        % Se c'è errore su theta, proietteremo la velocità lineare nella direzione sbagliata.
        
        ox(k) = ox(k-1) + v_noisy * cos(oth(k-1)) * dt;
        oy(k) = oy(k-1) + v_noisy * sin(oth(k-1)) * dt;
        oth(k) = oth(k-1) + w_noisy * dt;
    end
end