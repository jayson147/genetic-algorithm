% Load and define map with obstacles
map = im2bw(imread('random_map.bmp')); 
mapSize = size(map); % Actual size of the map

% Check if the map supports the desired endpoint coordinates
if mapSize(1) >= 500 && mapSize(2) >= 500

    % Define problem parameters
    startPoint = [1, 1];
    endPoint = [500, 500];
else
    error('Map size is too small for the desired endpoint.');
end

% Initialize parameters
populationSize = 500;
numGenerations = 100;
crossoverRate = 0.9;
mutationRate = 0.05;
noOfPointsInSolution = 8;


% Initialize population
population = initializePopulation(populationSize, noOfPointsInSolution, mapSize);


% Setting tournamentSize to 10% of the populationSize
tournamentSize = max(5, floor(0.2 * populationSize));  


% User input for selection method
disp('Choose Selection Method: 0 for RWS, 1 for Tournament, 2 for Rank');
selectionMethodInput = input('Enter your choice (0/1/2): ');

% Set selectionMethod based on input
switch selectionMethodInput
    case 0
        selectionMethod = 'RWS';
    case 1
        selectionMethod = 'Tournament';
        % tournamentSize is already set, no need for additional input
    case 2
        selectionMethod = 'Rank';
    otherwise
        disp('Invalid selection. Defaulting to RWS.');
        selectionMethod = 'RWS';
end


% User input for crossover method
disp('Choose Crossover Method: 0 for Single-Point Crossover, 1 for Uniform Crossover');
crossoverMethodInput = input('Enter your choice (0/1): ');

% Set crossoverMethod based on input
if crossoverMethodInput == 0
    crossoverMethod = 'SinglePoint';
else
    crossoverMethod = 'Uniform';
end

% User input for mutation method

disp('Choose Mutation Method: 0 for Random Resetting, 1 for Gaussian');
mutationMethodInput = input('Enter your choice (0/1): ');

% Set mutationMethod based on input
if mutationMethodInput == 0
    mutationMethod = 'RandomResetting';
else
    mutationMethod = 'Gaussian';
end

% Start measuring time
tic;

% Genetic Algorithm main loop
for gen = 1:numGenerations

    % Evaluate Fitness
    fitness = evaluatePopulation(population, map); % fitness evaluation

    % Selection
    parents = selectParents(population, fitness, selectionMethod, tournamentSize); % the selection method

    % Crossover and Mutation
    children = crossoverAndMutate(parents, crossoverRate, mutationRate, mapSize, crossoverMethod, mutationMethod);


    % Create new generation
    population = createNewGeneration(population, children, fitness); % generation update

end

% End measuring time
executionTime = toc;

% Find the best path and its fitness
[bestFitness, bestIndex] = min(fitness);
bestPath = population(bestIndex, :, :);



% Display Results
bestPath = findBestPath(population, fitness); % function to find the best path


startPoint = [1, 1];

endPoint = [size(map,1), size(map,2)]; 

% Include start and end points
bestPathWithEndpoints = [startPoint; bestPath; endPoint];


totalDistance = sum(sqrt(sum(diff(bestPath).^2, 2)));% Calculate total Euclidean distance

% Display the results
disp(['GA Execution Time: ', num2str(executionTime), ' seconds']);
disp(['Total Euclidean Distance of the Optimal Path: ', num2str(totalDistance)]);


% Call the plot function with the best path
bestPathWithEndpoints = [[1, 1]; bestPath; [mapSize(1), mapSize(2)]];

plotPath(bestPathWithEndpoints, map);


function population = initializePopulation(populationSize, noOfPointsInSolution, mapSize)

    population = zeros(populationSize, noOfPointsInSolution, 2);

    for i = 1:populationSize
        % Start point
        population(i, 1, :) = [1, 1];  % Assuming top-left is [1,1]
        % End point
        population(i, end, :) = [mapSize(1), mapSize(2)];  % Assuming bottom-right is [mapSize(1),mapSize(2)]
        
        % Generate random intermediate points
        for j = 2:(noOfPointsInSolution - 1)
            newPoint = [randi(mapSize(1)), randi(mapSize(2))];
            newPoint = clampPointToMap(newPoint, mapSize); % Clamp the point to map boundaries
            population(i, j, :) = newPoint;
        end

    end
end



function selected = rouletteWheelSelection(population, fitness)

    % Normalize fitness scores
    normalizedFitness = fitness / sum(fitness);
    
    % Calculate cumulative sum
    cumulativeSum = cumsum(normalizedFitness);
    
    % Select individuals
    selected = zeros(size(population));
    for i = 1:size(population, 1)
        r = rand();
        for j = 1:length(cumulativeSum)
            if r <= cumulativeSum(j)
                selected(i, :, :) = population(j, :, :);
                break;
            end
        end
    end
end

function selected = tournamentSelection(population, fitness, tournamentSize)
    % Initialize selected population
    selected = zeros(size(population));
    
    % Perform tournaments
    for i = 1:size(population, 1)
        % Randomly select tournament participants
        indices = randi([1 size(population, 1)], 1, tournamentSize);
        [~, bestIdx] = max(fitness(indices));
        
        % Select the best individual from the tournament
        selected(i, :, :) = population(indices(bestIdx), :, :);
    end
end

function selected = rankBasedSelection(population, fitness)
    % Rank individuals based on fitness
    [~, sortedIndices] = sort(fitness, 'descend');
    
    % Calculate selection probabilities based on rank
    totalRanks = sum(1:length(fitness));
    selectionProbabilities = (length(fitness):-1:1) / totalRanks;
    
    % Select individuals
    selected = zeros(size(population));
    for i = 1:size(population, 1)
        r = rand();
        cumulativeSum = cumsum(selectionProbabilities);
        for j = 1:length(cumulativeSum)
            if r <= cumulativeSum(j)
                selected(i, :, :) = population(sortedIndices(j), :, :);
                break;
            end
        end
    end
end


function parents = selectParents(population, fitness, selectionMethod, tournamentSize)

    switch selectionMethod
        case 'RWS'
            parents = rouletteWheelSelection(population, fitness);
        case 'Tournament'
            parents = tournamentSelection(population, fitness, tournamentSize);
        case 'Rank'
            parents = rankBasedSelection(population, fitness);
        otherwise
            error('Invalid selection method.');
    end
end

function offspring = singlePointCrossover(parent1, parent2)
    numPoints = size(parent1, 1);
    
    % Ensure there are enough points for crossover
    if numPoints > 3
        % Randomly choose a crossover point, avoiding the first and last point
        crossoverPoint = randi([2, numPoints - 2]);
        
        % Perform crossover
        offspring = [parent1(1:crossoverPoint, :); parent2(crossoverPoint+1:end, :)];
    else
        % If not enough points, just copy one of the parents
        offspring = parent1;
    end
end



function offspring = uniformCrossover(parent1, parent2)
    % Initialize offspring
    offspring = zeros(size(parent1));
    
    % Assuming the first two columns are x and y coordinates
    % Start point fixed at [1, 1]
    offspring(1, 1:2) = [1, 1];

    % End point fixed at [500, 500]
    offspring(end, 1:2) = [500, 500];

    % Randomly choose genes from either parent, excluding the first and last points
    for i = 2:size(parent1, 1) - 1
        if rand() < 0.5
            offspring(i, 1:2) = parent1(i, 1:2);
        else
            offspring(i, 1:2) = parent2(i, 1:2);
        end
    end
end

function mutated = randomResettingMutation(individual, mutationRate, mapSize, map)
    mutated = individual;
    for i = 2:size(individual, 1) - 1 
        if rand() < mutationRate
            newPoint = [randi([2, mapSize(1)-1]), randi([2, mapSize(2)-1])];
            newPoint = clampPointToMap(newPoint, mapSize); % Clamp the point to map boundaries
            if ~checkCollision(mutated(i-1,:), newPoint, map) && ...
               ~checkCollision(newPoint, mutated(i+1,:), map)
                mutated(i, :) = newPoint;
            end
        end
    end
end


function mutated = gaussianMutation(individual, mutationRate, sigma, mapSize)
    mutated = individual;
    for i = 2:size(individual, 1) - 1  % Avoid mutating start and end points
        if rand() < mutationRate
            newPoint = mutated(i, :) + sigma * randn(1, 2);
            newPoint = clampPointToMap(newPoint, mapSize); % Clamp the point to map boundaries
            mutated(i, :) = newPoint;
        end
    end
end



function children = crossoverAndMutate(parents, crossoverRate, mutationRate, mapSize, crossoverMethod, mutationMethod)
    children = zeros(size(parents));
    
    for i = 1:2:size(parents, 1)

        % Select parents
        parent1 = parents(i, :, :);
        parent2 = parents(min(i+1, size(parents, 1)), :, :);

        % Crossover
        if rand() < crossoverRate
            switch crossoverMethod
                case 'SinglePoint'
                    child1 = singlePointCrossover(parent1, parent2);
                    child2 = singlePointCrossover(parent2, parent1);
                case 'Uniform'
                    child1 = uniformCrossover(parent1, parent2);
                    child2 = uniformCrossover(parent2, parent1);
                otherwise
                    error('Invalid crossover method.');
            end
        else
            child1 = parent1;
            child2 = parent2;
        end

        % Mutation

        if strcmp(mutationMethod, 'RandomResetting')
            children(i, :, :) = randomResettingMutation(child1, mutationRate, mapSize);
            if i+1 <= size(parents, 1)
                children(i+1, :, :) = randomResettingMutation(child2, mutationRate, mapSize);
            end
        elseif strcmp(mutationMethod, 'Gaussian')
            children(i, :, :) = gaussianMutation(child1, mutationRate);
            if i+1 <= size(parents, 1)
                children(i+1, :, :) = gaussianMutation(child2, mutationRate);
            end
        else
            error('Invalid mutation method.');
        end
    end
end


function fitness = evaluatePopulation(population, map)

    populationSize = size(population, 1);
    fitness = zeros(populationSize, 1);

    for i = 1:populationSize
        path = squeeze(population(i, :, :));

        % Calculate path length
        pathLength = sum(sqrt(sum(diff(path).^2, 2)));

        % Penalty for obstacles
        obstaclePenalty = 0;
        for j = 1:size(path, 1) - 1
            if checkCollision(path(j,:), path(j+1,:), map)
                obstaclePenalty = obstaclePenalty + 1e6; % Increase the magnitude of the penalty


            end
        end

        % Combine path length and obstacle penalty
        fitness(i) = pathLength + obstaclePenalty;
    end

end



function collision = checkCollision(point1, point2, map)
    collision = false;
    linePoints = bresenhamLine(point1, point2); % Bresenham's line algorithm
    
    for k = 1:size(linePoints, 1)
        x = linePoints(k, 1);
        y = linePoints(k, 2);

        % Check if the indices are valid for the map
        if x >= 1 && x <= size(map, 1) && y >= 1 && y <= size(map, 2)
            if map(x, y) == 1 % 1 represents obstacle
                collision = true;
                break;
            end
        else
            % Handling invalid indices
            collision = true; 
            break;

   
        end
    end
end


% Additional helper function for Bresenham's line algorithm

function points = bresenhamLine(p1, p2)
    % Bresenham's Line Algorithm to generate the points of a line in a grid
    % p1 and p2 are the endpoints of the line [x1, y1] and [x2, y2]

    x1 = p1(1); y1 = p1(2);
    x2 = p2(1); y2 = p2(2);
    
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    steep = dy > dx;

    if steep
        [x1, y1] = deal(y1, x1);
        [x2, y2] = deal(y2, x2);
        [dx, dy] = deal(dy, dx);
    end

    if x1 > x2
        [x1, x2] = deal(x2, x1);
        [y1, y2] = deal(y2, y1);
    end

    derr = 2*dy;
    err = 0;
    y = y1;
    points = zeros(dx + 1, 2);
    for x = x1:x2
        if steep
            points(x-x1+1, :) = [y, x];
        else
            points(x-x1+1, :) = [x, y];
        end
        err = err + derr;
        if err > dx
            y = y + (y2 > y1) * 2 - 1;  % Move y one step
            err = err - 2*dx;
        end
    end
end

function newPopulation = createNewGeneration(currentPopulation, children, fitness)
    populationSize = size(currentPopulation, 1);

    % Sort the current population based on fitness
    % Lower fitness is better, so sort in ascending order
    [~, sortedIndices] = sort(fitness);
    sortedPopulation = currentPopulation(sortedIndices, :, :);

    % Elitism: Select the top individuals from the current population
    numElite = floor(0.1 * populationSize); % for example, 10% of the population

    % Preserving the top 'numElite' individuals unchanged
    eliteIndividuals = sortedPopulation(1:numElite, :, :);


    % Fill the rest of the new population with children
    numChildren = populationSize - numElite;
    newPopulation = cat(1, eliteIndividuals, children(1:numChildren, :, :));
end

function bestPath = findBestPath(population, fitness)

    % Find the index of the best individual (lowest fitness)
    [~, bestIdx] = min(fitness);

    % Retrieve the best path from the population
    bestPath = squeeze(population(bestIdx, :, :));
end

function clampedPoint = clampPointToMap(point, mapSize)
    clampedPoint = max(min(point, mapSize), 1);
end


function plotPath(path, map)
    % Plot the map
    imshow(map, 'InitialMagnification', 'fit');
    hold on;
    
    % Plot the path
    plot(path(:,2), path(:,1), 'r', 'LineWidth', 2);
    
    % Plot start and end points
    plot(path(1,2), path(1,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);  % start point
    plot(path(end,2), path(end,1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);  % end point
    
    hold off;
end




















