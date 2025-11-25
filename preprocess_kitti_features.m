
function preprocess_kitti_features(sequence_path, output_filename)
    if ~isfolder(sequence_path)
        error('Folder doesnt found in: %s', sequence_path);
    end
    files = dir(fullfile(sequence_path, '*.png'));
    num_images = length(files);
    if num_images == 0
        error('Any image .png found in folder.');
    end
    
    fprintf('Start extracting ORB on %d images for %s.\n', num_images, output_filename);
    
    AllFeatures = cell(num_images, 1);
    
    parfor k = 1:num_images
        
        img_name = fullfile(sequence_path, files(k).name);
        img = imread(img_name);
       
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        points = detectORBFeatures(img, 'ScaleFactor', 1.2, 'NumLevels', 8);
        points = points.selectStrongest(500);
        [features, valid_points] = extractFeatures(img, points);
    
        if isa(features, 'binaryFeatures')
            AllFeatures{k} = features.Features;
        else
            AllFeatures{k} = features;
        end    
    end
    
    fprintf('Saving in %s \n', output_filename);
    output_path = 'Dataset/features';
    if ~isfolder(output_path)
        mkdir(output_path);
    end
    output_file_path = fullfile(output_path, output_filename);
    save(output_file_path, 'AllFeatures', '-v7.3'); 
    fprintf('Pre-process finished.\n\n');

end