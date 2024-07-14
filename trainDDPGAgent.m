% Create a new file named trainDDPGAgent.m
% Define the environment
env = SimpleOceanEnv;

% Define observation and action spaces
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Create a critic network
statePath = featureInputLayer(obsInfo.Dimension(1), 'Name', 'state');
actionPath = featureInputLayer(actInfo.Dimension(1), 'Name', 'action');
commonPath = concatenationLayer(1, 2, 'Name', 'concat');
commonPath = [commonPath
              fullyConnectedLayer(24, 'Name', 'fc1')
              reluLayer('Name', 'relu1')
              fullyConnectedLayer(24, 'Name', 'fc2')
              reluLayer('Name', 'relu2')
              fullyConnectedLayer(1, 'Name', 'fc3')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, 'state', 'concat/in1');
criticNetwork = connectLayers(criticNetwork, 'action', 'concat/in2');

criticOpts = rlRepresentationOptions('LearnRate', 1e-03, 'GradientThreshold', 1);

critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, ...
    'Observation', {'state'}, 'Action', {'action'}, criticOpts);

% Create an actor network
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'fc3')
    tanhLayer('Name', 'tanh')]; % Tanh activation for continuous actions
actorOpts = rlRepresentationOptions('LearnRate', 1e-04, 'GradientThreshold', 1);

actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    'Observation', {'state'}, actorOpts);

% Define the DDPG agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime', 1, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 64, ...
    'ExperienceBufferLength', 1e6);

agent = rlDDPGAgent(actor, critic, agentOpts);

% Set training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', 100, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the agent
trainStats = train(agent, env, trainOpts);

% Simulate the trained agent
simOptions = rlSimulationOptions('MaxSteps', 100);
experience = sim(env, agent, simOptions);
