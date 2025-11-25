function Map = build_experience_map(AgentData, dist_thresh, angle_thresh)

    % --- Initialization ---
    Map.Nodes.x = [];
    Map.Nodes.y = [];
    Map.Nodes.theta = [];
    Map.Nodes.id = [];       % Unique Node ID
    Map.Nodes.view_id = [];  % Associated Visual Place ID (for Loop Closure)
    
    % Links represent relative odometric constraints between nodes
    Map.Links.from = [];
    Map.Links.to = [];
    Map.Links.dx = [];     % Relative X in 'from' frame
    Map.Links.dy = [];     % Relative Y in 'from' frame
    Map.Links.dtheta = []; % Relative Theta

    % --- Add Initial Node ---
    current_node_idx = 1;
    
    Map.Nodes.id(end+1) = current_node_idx;
    Map.Nodes.x(end+1) = AgentData.odom_x(1);
    Map.Nodes.y(end+1) = AgentData.odom_y(1);
    Map.Nodes.theta(end+1) = AgentData.odom_theta(1);
    Map.Nodes.view_id(end+1) = AgentData.ViewID(1);
    
    last_raw_idx = 1; % Index of the raw data corresponding to the last node
    num_samples = length(AgentData.odom_x);

    % --- Graph Construction Loop ---
    for k = 2:num_samples
        
        % State of the last created node
        node_x = AgentData.odom_x(last_raw_idx);
        node_y = AgentData.odom_y(last_raw_idx);
        node_th = AgentData.odom_theta(last_raw_idx);
        
        % Current candidate state
        curr_x = AgentData.odom_x(k);
        curr_y = AgentData.odom_y(k);
        curr_th = AgentData.odom_theta(k);
        
        % Compute accumulated distance and rotation
        dist = sqrt((curr_x - node_x)^2 + (curr_y - node_y)^2);
        dth = abs(angdiff(node_th, curr_th));
        
        % Threshold Check (Sparsification Logic)
        if dist > dist_thresh || dth > angle_thresh
            
            new_node_idx = current_node_idx + 1;
            
            % A. Create New Node (Vertex)
            Map.Nodes.id(end+1) = new_node_idx;
            Map.Nodes.x(end+1) = curr_x;
            Map.Nodes.y(end+1) = curr_y;
            Map.Nodes.theta(end+1) = curr_th;
            Map.Nodes.view_id(end+1) = AgentData.ViewID(k);
            
            % B. Create Link (Edge) - Relative Constraint
            % We transform the global displacement into the local frame of the previous node.
            % This simulates the robot's internal odometry measurement between nodes.
            
            % Global displacement
            gx = curr_x - node_x;
            gy = curr_y - node_y;
            
            % Rotation matrix (Inverse of node_th)
            c = cos(node_th);
            s = sin(node_th);
            
            % Project to local frame
            rel_dx = gx * c + gy * s;
            rel_dy = -gx * s + gy * c;
            rel_dth = angdiff(node_th, curr_th);
            
            % Store the constraint
            Map.Links.from(end+1) = current_node_idx;
            Map.Links.to(end+1) = new_node_idx;
            Map.Links.dx(end+1) = rel_dx;
            Map.Links.dy(end+1) = rel_dy;
            Map.Links.dtheta(end+1) = rel_dth;
            
            % Update indices
            current_node_idx = new_node_idx;
            last_raw_idx = k;
        end
    end
end