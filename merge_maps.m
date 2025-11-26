function Global_Map = merge_maps(L_Map, F_Map, CorrectionData)
% MERGE_MAPS - Create unified topological graph from Leader and Follower maps
%
% This function physically merges the two separate maps into a single
% global topological graph, representing the true collaborative SLAM result.
%
% Inputs:
%   L_Map          - Leader's map structure (.Nodes, .Count)
%   F_Map          - Follower's map (already aligned via CMF-GR)
%   CorrectionData - Matrix of correction events [F_x, F_y, L_x, L_y]
%
% Output:
%   Global_Map - Unified map structure with:
%                .Nodes - All nodes from both robots
%                .Count - Total number of nodes
%                .Edges - Connections between nodes (rendezvous links)

%% Initialize Global Map
Global_Map.Nodes = [];
Global_Map.Count = 0;
Global_Map.Edges = [];

% If no corrections happened, return Leader map only
if isempty(CorrectionData) || F_Map.Count == 0
    Global_Map = L_Map;
    fprintf('No fusion occurred. Global map = Leader map (%d nodes)\n', L_Map.Count);
    return;
end

%% Add All Leader Nodes
fprintf('Merging maps...\n');
for i = 1:L_Map.Count
    Global_Map.Count = Global_Map.Count + 1;
    Global_Map.Nodes(Global_Map.Count).id = Global_Map.Count;
    Global_Map.Nodes(Global_Map.Count).pose = L_Map.Nodes(i).pose;
    Global_Map.Nodes(Global_Map.Count).source = 'Leader';
    Global_Map.Nodes(Global_Map.Count).view_id = L_Map.Nodes(i).view_id;
    Global_Map.Nodes(Global_Map.Count).original_id = i;  % Track original ID
end

num_leader_nodes = Global_Map.Count;

%% Add Follower Nodes (excluding near-duplicates)
% Nodes very close to Leader nodes (rendezvous points) are considered duplicates
duplicate_threshold = 3.0;  % meters
num_duplicates = 0;
num_follower_unique = 0;

for i = 1:F_Map.Count
    f_pose = F_Map.Nodes(i).pose;

    % Check if this Follower node is too close to any Leader node
    is_duplicate = false;
    closest_L_id = -1;
    min_dist = inf;

    for j = 1:L_Map.Count
        l_pose = L_Map.Nodes(j).pose;
        dist = norm(f_pose(1:2) - l_pose(1:2));

        if dist < min_dist
            min_dist = dist;
            closest_L_id = j;  % Global Map ID for this Leader node
        end

        if dist < duplicate_threshold
            is_duplicate = true;
            break;
        end
    end

    if ~is_duplicate
        % Add as unique Follower node
        Global_Map.Count = Global_Map.Count + 1;
        Global_Map.Nodes(Global_Map.Count).id = Global_Map.Count;
        Global_Map.Nodes(Global_Map.Count).pose = f_pose;
        Global_Map.Nodes(Global_Map.Count).source = 'Follower';
        Global_Map.Nodes(Global_Map.Count).view_id = F_Map.Nodes(i).view_id;
        Global_Map.Nodes(Global_Map.Count).original_id = i;
        num_follower_unique = num_follower_unique + 1;
    else
        % This is a rendezvous point - create edge linking to Leader node
        edge.from_source = 'Follower';
        edge.from_original_id = i;
        edge.to_source = 'Leader';
        edge.to_global_id = closest_L_id;
        edge.type = 'rendezvous';
        edge.distance = min_dist;
        Global_Map.Edges = [Global_Map.Edges; edge];
        num_duplicates = num_duplicates + 1;
    end
end

%% Summary
fprintf('  Leader nodes:          %d\n', num_leader_nodes);
fprintf('  Follower unique nodes: %d\n', num_follower_unique);
fprintf('  Rendezvous links:      %d\n', num_duplicates);
fprintf('  Total global nodes:    %d\n', Global_Map.Count);
fprintf('Global map created successfully.\n\n');

end
