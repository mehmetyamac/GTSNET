function net = RDN_network(net,nch,inname,outname)

G0 = 30; %32 Initial feature size
D = 4; %12; % Number of residual Dense Blocks (RDBs)
C = 3;  % Number of convolution in each RDB
G = 12; %16 Number of filter channels in each RDB convolution layer
k = 3; % Filter size

% Initial convolution
block = dagnn.Conv('size', [k k nch G0], 'pad', floor(k/2), 'hasBias', true);
net.addLayer('convFm1',block,inname,'Fm1',{'wm1','bm1'});
inname0 = 'Fm1';
% Second convolution
block = dagnn.Conv('size', [k k G0 G0], 'pad', floor(k/2), 'hasBias', true);
net.addLayer('convF0',block,inname0,'F0',{'w0','b0'});

% RDBs
prevout = {};
dinname = 'F0';
for d = 1:D
   doutname = ['F' num2str(d)];
   net = residualDenseBlock(net,d,G0,G,k,C,dinname,doutname);
   prevout = [prevout doutname]; 
   dinname = doutname;
end
% Concatenation
net.addLayer('ConcatAll', dagnn.Concat(), prevout, 'CRN');
% Convolution
block = dagnn.Conv('size', [1 1 D*G0 G0], 'pad', 0, 'hasBias', true);
net.addLayer('convSum1',block,'CRN','FRN',{'wrn1','brn1'});
% Convolution
block = dagnn.Conv('size', [k k G0 G0], 'pad', floor(k/2), 'hasBias', true);
net.addLayer('convSum3',block,'FRN','FGF',{'wgf','bgf'});
% Summation
block = dagnn.Sum();
net.addLayer('SumFinal',block,{inname0 'FGF'},'FDF');
% Final convolution
block = dagnn.Conv('size', [k k G0 nch], 'pad', floor(k/2), 'hasBias', true);
net.addLayer('ConvFinal',block,'FDF','FDF_Sum',{'wlast','blast'});
% Add
block = dagnn.Sum();
net.addLayer('SumOut',block,{'FDF_Sum' inname},outname);