classdef SimpleOceanEnv < rl.env.MATLABEnvironment
    % SimpleOceanEnv: Defines a simple grid environment with ocean currents

    properties
        % Environment properties
        GridSize = [5, 5]; % Grid size [rows, columns]
        StartState = [1, 1]; % Starting position [row, column]
        GoalState = [5, 5]; % Goal position [row, column]
        ObstacleState = [3, 3]; % Obstacle position [row, column]
        CurrentForce = [0, 1]; % Ocean current force [row change, column change]
        State % Current state [row, column]
    end

    properties(Access = protected)
        % Internal properties
        IsDone % Indicator for terminal state
    end

    methods
        % Constructor method
        function this = SimpleOceanEnv()
            % Initialize the environment
            ObservationInfo = rlNumericSpec([2 1], 'LowerLimit', [1; 1], 'UpperLimit', [5; 5]);
            ActionInfo = rlNumericSpec([2 1], 'LowerLimit', [-1; -1], 'UpperLimit', [1; 1]); % Continuous actions
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            reset(this);
        end

        % Reset environment to initial state
        function InitialObservation = reset(this)
            this.State = this.StartState;
            this.IsDone = false;
            InitialObservation = this.State';
        end

        % Step function to update environment based on action
        function [NextObservation, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            if this.IsDone
                error('Episode already terminated.');
            end

            % Update state based on action
            this.State = this.State + round(Action');
            this.State = min(max(this.State, [1, 1]), this.GridSize);

            % Apply ocean current
            this.State = this.State + this.CurrentForce;
            this.State = min(max(this.State, [1, 1]), this.GridSize);

            % Check for obstacle collision
            if isequal(this.State, this.ObstacleState)
                Reward = -10;
                this.IsDone = true;
            % Check for goal state
            elseif isequal(this.State, this.GoalState)
                Reward = 10;
                this.IsDone = true;
            else
                Reward = -1; % Small penalty for each step
            end

            NextObservation = this.State';
            IsDone = this.IsDone;
        end
    end
end
