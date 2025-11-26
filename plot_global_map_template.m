% PLOT_GLOBAL_MAP - Visualize the unified global topological graph
%
% Add this at the end of main.m plots section

% Plot 3: Global Map Visualization
figure('Name', 'Global Topological Map', 'Color', 'w', 'Position', [100, 50, 1000, 700]);
hold on; grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]');
title('Global Map: Unified Topological Graph (Leader + Follower)');

% Plot Ground Truth for reference
plot(GT.x, GT.y, 'Color', [0.9 0.9 0.9], 'LineWidth', 2, 'DisplayName', 'Ground Truth');

% Plot all nodes colored by source
for i = 1:Global_Map.Count
    node = Global_Map.Nodes(i);

    if strcmp(node.source, 'Leader')
        % Leader nodes in blue
        plot(node.pose(1), node.pose(2), 'bo', 'MarkerSize', 6, ...
            'MarkerFaceColor', 'b', 'HandleVisibility', 'off');
    else
        % Follower nodes in red
        plot(node.pose(1), node.pose(2), 'ro', 'MarkerSize', 6, ...
            'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
    end
end

% Add representative markers for legend
plot(nan, nan, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', ...
    sprintf('Leader Nodes (%d)', sum(strcmp({Global_Map.Nodes.source}, 'Leader'))));
plot(nan, nan, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', ...
    sprintf('Follower Nodes (%d unique)', sum(strcmp({Global_Map.Nodes.source}, 'Follower'))));

% Draw rendezvous edges
if ~isempty(Global_Map.Edges)
    for i = 1:length(Global_Map.Edges)
        edge = Global_Map.Edges(i);

        % Find the Follower node position
        f_idx = edge.from_original_id;
        if f_idx <= F_Map.Count
            f_pos = F_Map.Nodes(f_idx).pose(1:2);

            % Find the Leader node position
            l_idx = edge.to_global_id;
            if l_idx <= L_Map.Count
                l_pos = L_Map.Nodes(l_idx).pose(1:2);

                % Draw link
                if i == 1
                    plot([f_pos(1), l_pos(1)], [f_pos(2), l_pos(2)], 'g-', ...
                        'LineWidth', 2, 'DisplayName', sprintf('Rendezvous Links (%d)', length(Global_Map.Edges)));
                else
                    plot([f_pos(1), l_pos(1)], [f_pos(2), l_pos(2)], 'g-', ...
                        'LineWidth', 2, 'HandleVisibility', 'off');
                end

                % Mark rendezvous points
                plot(f_pos(1), f_pos(2), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
    end
end

legend('Location', 'best');

% Add statistics text box
stats_text = sprintf(['Global Map Statistics:\n' ...
    'Total Nodes: %d\n' ...
    'Leader: %d nodes\n' ...
    'Follower: %d unique nodes\n' ...
    'Rendezvous Links: %d'], ...
    Global_Map.Count, ...
    sum(strcmp({Global_Map.Nodes.source}, 'Leader')), ...
    sum(strcmp({Global_Map.Nodes.source}, 'Follower')), ...
    length(Global_Map.Edges));

annotation('textbox', [0.15, 0.75, 0.2, 0.15], 'String', stats_text, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'FontSize', 10);
