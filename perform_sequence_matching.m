function [is_match, score] = perform_sequence_matching(curr_F_id, cand_L_id, FeaturesF, FeaturesL, Params)
% PERFORM_SEQUENCE_MATCHING
% Implementa il controllo di sequenza temporale (Eq. 6 del paper).
%
% INPUT:
% curr_F_id : Indice immagine corrente del Follower (ViewID)
% cand_L_id : Indice immagine candidata del Leader (ViewID)
% FeaturesF : Cell array feature follower
% FeaturesL : Cell array feature leader
% Params    : struct con .seq_len (es. 5) e .match_thresh (es. 0.15)

    is_match = false;
    score = 0;
    
    K = floor(Params.seq_len / 2); % Metà finestra (es. 2 se len=5)
    
    % 1. Controllo limiti indice (non possiamo matchare l'inizio o la fine assoluta)
    if curr_F_id <= K || curr_F_id > length(FeaturesF) - K || ...
       cand_L_id <= K || cand_L_id > length(FeaturesL) - K
        return;
    end
    
    % 2. Calcolo Score sulla sequenza
    % Sommiamo i punteggi di similarità lungo la diagonale temporale
    accumulated_score = 0;
    
    for delta = -K:K
        % Recupera i descrittori grezzi (matrici binarie)
        desc_F = FeaturesF{curr_F_id + delta};
        desc_L = FeaturesL{cand_L_id + delta};
        %------------------------ MODIFICA DI TEST ------------------
        %safety_idx = min(length(FeaturesL), cand_L_id + delta + 2); 
        %desc_L = FeaturesL{safety_idx};
        
        if isempty(desc_F) || isempty(desc_L)
            continue; 
        end
        
        % Matching visuale (Distanza di Hamming per descrittori binari)
        % MaxRatio 0.8 è standard per filtrare match ambigui (Lowe's ratio test)
        indexPairs = matchFeatures(binaryFeatures(desc_F), binaryFeatures(desc_L), ...
            'MatchThreshold', 50, ...
            'MaxRatio', 0.8, ...
            'Unique', false);
        
        num_matches = size(indexPairs, 1);
        
        % Similarità S(I_B, I_A) normalizzata sul numero di feature del Follower
        % Evita divisione per zero
        if size(desc_F, 1) > 0
            sim_score = num_matches / size(desc_F, 1);
        else
            sim_score = 0;
        end
        
        accumulated_score = accumulated_score + sim_score;
    end
    
    % 3. Media (Eq. 6)
    avg_score = accumulated_score / Params.seq_len;
    
    % 4. Decisione
    if avg_score > Params.match_thresh
        is_match = true;
        score = avg_score;
    end
end