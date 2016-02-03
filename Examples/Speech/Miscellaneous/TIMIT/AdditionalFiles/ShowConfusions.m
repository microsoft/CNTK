function ShowConfusions(confusionData, squeeze)
% function ShowConfusions(confusionData)
% Average the three-state confusion data into monophone confusions.  Then
% display the data.  A graphical interface lets you interrogate the data,
% by moving the mouse, and clicking at various points.  The phonetic labels
% are shown on the graph.
confusionSmall = ( ...
    confusionData(1:3:end,1:3:end) + confusionData(2:3:end, 1:3:end) + confusionData(3:3:end, 1:3:end) + ...
    confusionData(1:3:end,2:3:end) + confusionData(2:3:end, 2:3:end) + confusionData(3:3:end, 2:3:end) + ...
    confusionData(1:3:end,3:3:end) + confusionData(2:3:end, 3:3:end) + confusionData(3:3:end, 3:3:end))/9;

if nargin < 2
    squeeze = 1;
end

imagesc(confusionSmall .^ squeeze)
axis ij
axis square
ylabel('True Label');
xlabel('CNTK Prediction');

%%
stateList = ReadStateList();

h = [];
fprintf('Select a point with the mouse, type return to end...\n');
while true
    [x,y] = ginput(1);
    if isempty(x) || isempty(y)
        break;
    end
    
    if ~isempty(h)
        delete(h);
        h = [];
    end
    try
        trueLabel = stateList{(round(x)-1)*3+1};
    catch
        trueLabel = 'Unknown'; 
    end
    try
        likelihoodLabel = stateList{(round(y)-1)*3+1};
    catch
        likelihoodLabel = 'Unknown';
    end
    h = text(40, -2, sprintf('%s -> %s', trueLabel, likelihoodLabel));
    % h = text(40, -2, sprintf('%g -> %g', x, y));
end

function stateList = ReadStateList(stateListFile)
% Read in the state list file. This file contains an ordered list of
% states, each corresponding to one label (and one output in the CNTK
% network.)
if nargin < 1
    stateListFile = 'TimitStateList.txt';
end
% Read in the state list file.
fp = fopen(stateListFile);
nStates = 183;              % Preordained
stateList = cell(nStates, 1);
stateIndex = 1;
while true
    theLine = fgets(fp);
    if isempty(theLine) || theLine(1) == -1
        break;
    end
    f = find(theLine == '_');
    if ~isempty(f)
        label = theLine(1:f(1)-1);
    else
        label = theLine(1:end-1);
    end
    stateList{stateIndex} = label;
    stateIndex = stateIndex + 1;
end
fclose(fp);
