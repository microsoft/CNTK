%% 
% Create training and test sets for the simple CNTK demo. Plot the results
% Create 2-dimensional data for the testing the CNTK Toolkit

N = 10000;
x = 2*(rand(N,1) - 0.5);            % Uniform from -1 to 1
y = 2*(rand(N,1) - 0.5);            % Uniform from -1 to 1

label = 0.25*sin(2*pi*0.5*x) > y;   % Sinusoidal decision boundary

%%
% Plot the training data
figure(1);
plot(x(label), y(label), 'rx', ...
    x(~label), y(~label), 'bo');
xlabel('X axis');
ylabel('Y axis');
title('Simple Data Training Plot');

print -dpng SimpleDemoData

%%
% Dump the data to a file. Label needs to be an integer so we can't just
% dump the entire array at once.
fp = fopen('SimpleDataTrain.txt','w');
for i=1:N
    fprintf(fp, '%g %g %d\n', x(i), y(i), label(i));
end
fclose(fp);

%%
% Create a uniform grid of test data.  This is easier to plot than the
% training data.
testDelta = 0.01;
testMax = 1;
testPoints = [-testMax:testDelta:testMax];
testN = length(testPoints);
[testX, testY] = meshgrid(testPoints, testPoints);
fp = fopen('SimpleDataTest.txt','w');
for i=1:length(testX(:))
    fprintf(fp, '%g %g %d\n', testX(i), testY(i), 0);
end
fclose(fp);

%% 
% 
% Run CNTK here.
% The rest of the command in this file plot the results.

%%
load SimpleDataTrain.txt
load SimpleDataTest.txt
load SimpleOutput.ScaledLogLikelihood

%%
figure(2);
if 0
    % Plot each test point.
    pos = SimpleOutput(:,1)>SimpleOutput(:,2);
	plot(SimpleDataTest(pos,1), SimpleDataTest(pos,2), 'rx', ...
        SimpleDataTest(~pos,1), SimpleDataTest(~pos,2), 'bo')
else
    data=reshape(SimpleOutput(:,1), testN, testN);
    imagesc(testPoints, testPoints, data);
    axis xy
    colorbar
    title('Output 2 from DNN');
    [m,i] = min(data.^2);
    hold on; plot(testPoints, testPoints(i), '--'); hold off
end
print -dpng SimpleDemoOutput

%%
% Capture the training error rate information from the log file.
fp = fopen('Demo_Simple_Demo_Simple_Demo_Output.log', 'r');
if fp
    clear trainingError
    while true
        theLine = fgets(fp);
        if isempty(theLine) || theLine(1) == -1
            break;
        end
        % Look for the message at the end of each epoch.
        if strncmp(theLine, 'Finished ', length('Finished '))
            try
                % Pick out the epoch number and training error
                strToks = regexprep(theLine, ...
                    '.*Epoch\[(\d*)].*EvalErr Per Sample = (.*) +Ave Learn.*', ...
                    '$1 $2');
                numericToks = str2num(strToks);
                trainingError(numericToks(1)) = numericToks(2);
            catch e
                % Ignore lines we can't read
            end
        end
    end
    fclose(fp);
else
    fprintf('Can not find the demo training log\n');
end

%% 
figure(3);
loglog(trainingError)
title('Performance of Simple Demo');
xlabel('Epoch Number');
ylabel('Average Error Rate (training data)');

print -dpng SimpleDemoErrorRate

