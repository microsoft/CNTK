function confusionData = ComputeConfusion(mlfFile)
% function confusionData = ComputeConfusion(mlfFile)
% Compute all the confusions for one experiment. Read in the TIMIT MLF file
% so we know which utterances we have.  For each utterance, read in the
% CNTK output and compute the confusion matrix.  Sum them all together.
if nargin < 1
    mlfFile = 'TimitLabels.mlf';
end

%% 
% Parse the Timit MLF file because it tells us the true phonetic labels 
% for each segment of each utterance.  
fp = fopen(mlfFile,'r');

segmentLabels = [];
scale=1e-7;
numberOfUtterances = 0;
confusionData = 0;
while 1
    theLine = fgets(fp);
    if isempty(theLine) || theLine(1) == -1
        break;
    end
    if strncmp(theLine, '#!MLF!#', 7)
        continue;               % Ignore the header
    end
    if theLine(1) == '"'        % Look for file name indication
        numberOfUtterances = numberOfUtterances + 1;
        fileName = strtok(theLine);
        fileName = fileName(2:end-1);
        segmentLabels = [];
    end
    if theLine(1) >= '0' && theLine(1) <= '9'
        % Got a speech segment with times and phoneme label. Parse it.
        c = textscan(theLine, '%d %d %s ');
        b = double(c{1}(1)); e = double(c{2}); l = c{3}{1};
        if isempty(segmentLabels)
            clear segmentLabels;
            segmentLabels(1000) = struct('begin', b, 'end', e, 'label', l);
            segmentCount = 0;
        end
        segmentCount = segmentCount + 1;
        % Add a new entry in the list of segments.
        segmentLabels(segmentCount) = struct('begin', b*scale, 'end', e*scale, 'label', l);
    end
    if theLine(1) == '.'
        % Found the end of the speech transcription.  Process the new data.
        c = ComputeConfusionOnce(fileName, segmentLabels(1:segmentCount));
        confusionData = confusionData + c;
        segmentLabels = [];
    end
end
fclose(fp);

function Confusions = ComputeConfusionOnce(utteranceName, segmentLabels)
% function Confusions = ComputeConfusionOnce(utteranceName, labelData)
% Compute the confusion matrix for one TIMIT utterance.  This routine takes
% the segment data (from the TIMIT label file) and a feature-file name.  It
% transforms the feature file into a CNTK output file.  It reads in the
% CNTK output file, and tabulates a confusion matrix.  We do this one
% segment at a time, since TIMIT segments are variable length, and the CNTK
% output is sampled at regular intervals (10ms).
likelihoodName = strrep(strrep(utteranceName, 'Features/', 'Output/'), ...
    'fbank_zda', 'log');
try
    [likelihood,~] = htkread(likelihoodName);
catch me
    fprintf('Can''t read %s using htkread.  Ignoring.\n', likelihoodName);
    Confusions = 0;
    return
end

nStates = 183;              % Preordained.
frameRate = 100;            % Preordained
Confusions = zeros(nStates, nStates);
for i=1:size(segmentLabels, 2)
    % Go through each entry in the MLF file for one utterance.  Each entry
    % lists the beginning and each of each speech state.
    % Compare the true label with the winner of the maximum likelihood from
    % CNTK.
    beginIndex = max(1, round(segmentLabels(i).begin*frameRate));
    endIndex = min(size(likelihood,1), round(segmentLabels(i).end*frameRate));
    curIndices = beginIndex:endIndex;
    [~,winners] = max(likelihood(curIndices,:),[], 2);
    correctLabel = FindLabelNumber(segmentLabels(i).label);
    for w=winners(:)'                 % increment one at a time
        Confusions(correctLabel, w) = Confusions(correctLabel, w) + 1;
    end
end

function labelNumber = FindLabelNumber(labelName)
% For each label name, turn the name into an index. The labels are listed,
% in order, in the TimitStateList file.
persistent stateList
if isempty(stateList)
    stateList = ReadStateList('TimitStateList.txt');
end
for labelNumber=1:size(stateList,1)
    if strcmp(labelName, stateList{labelNumber})
        return;
    end
end
labelNumber = [];

function stateList = ReadStateList(stateListFile)
% Read in the state list file. This file contains an ordered list of
% states, each corresponding to one label (and one output in the CNTK
% network.)
fp = fopen(stateListFile);
nStates = 183;              % Preordained
stateList = cell(nStates, 1);
stateIndex = 1;
while true
    theLine = fgets(fp);
    if isempty(theLine) || theLine(1) == -1
        break;
    end
    stateList{stateIndex} = theLine(1:end-1);
    stateIndex = stateIndex + 1;
end
fclose(fp);
        

function [ DATA, HTKCode ] = htkread( Filename )
% [ DATA, HTKCode ] = htkread( Filename )
%
% Read DATA from possibly compressed HTK format file.
%
% Filename (string) - Name of the file to read from
% DATA (nSamp x NUMCOFS) - Output data array
% HTKCode - HTKCode describing file contents
%
% Compression is handled using the algorithm in 5.10 of the HTKBook.
% CRC is not implemented.
%
% Mark Hasegawa-Johnson
% July 3, 2002
% Based on function mfcc_read written by Alexis Bernard
% Found at: https://raw.githubusercontent.com/ronw/matlab_htk/master/htkread.m
%

fid=fopen(Filename,'r','b');
if fid<0,
    error(sprintf('Unable to read from file %s',Filename));
end

% Read number of frames
nSamp = fread(fid,1,'int32');

% Read sampPeriod
sampPeriod = fread(fid,1,'int32');

% Read sampSize
sampSize = fread(fid,1,'int16');

% Read HTK Code
HTKCode = fread(fid,1,'int16');

%%%%%%%%%%%%%%%%%
% Read the data
if bitget(HTKCode, 11),
    DIM=sampSize/2;
    nSamp = nSamp-4;
    %disp(sprintf('htkread: Reading %d frames, dim %d, compressed, from %s',nSamp,DIM,Filename)); 

    % Read the compression parameters
    A = fread(fid,[1 DIM],'float');
B = fread(fid,[1 DIM],'float');
    
    % Read and uncompress the data
    DATA = fread(fid, [DIM nSamp], 'int16')';
    DATA = (repmat(B, [nSamp 1]) + DATA) ./ repmat(A, [nSamp 1]);

    
else
    DIM=sampSize/4;
    %disp(sprintf('htkread: Reading %d frames, dim %d, uncompressed, from %s',nSamp,DIM,Filename)); 

    % If not compressed: Read floating point data
    DATA = fread(fid, [DIM nSamp], 'float')';
end

fclose(fid);

