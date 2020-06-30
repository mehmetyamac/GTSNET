function net = residualDenseBlock(net,d,nch,f,k,C,inname,outname)

leak = 0; % Relu

nch0 = nch;
inname0 = inname;
prevout = {inname};

for c=1:C
    % Convolution
    block = convBlock(k,k,nch,f);
    name = ['conv_' num2str(c) '_rdb_' num2str(d)];
    coutname = ['c' num2str(c) '_rdb_' num2str(d)];
    weights = {['w' num2str(c) '_rdb_' num2str(d)], ['b' num2str(c) '_rdb_' num2str(d)]};
    net.addLayer(name,block,inname,coutname,weights);
    % Relu
    block = dagnn.ReLU('leak',leak);
    name = ['relu_' num2str(c) '_rdb_' num2str(d)];
    routname = ['r' num2str(c) '_rdb_' num2str(d)];
    net.addLayer(name,block,coutname,routname);
    % Concatenation
    block = dagnn.Concat();
    name = ['conc_' num2str(c) '_rdb_' num2str(d)];
    coutname = ['ct' num2str(c) '_rdb_' num2str(d)];
    net.addLayer(name,block,[routname, prevout],coutname);
    % Update names
    prevout = {coutname}; %[prevout coutname];
    inname = coutname;
    % Update channels
    nch = nch+f;
end
% Final convolution
block = convBlock(1,1,nch,nch0);
name = ['convSum_rdb_' num2str(d)];
coutname = ['csum_rdb_' num2str(d)];
weights = {['wsumrdb_' num2str(d)], ['bsumrdb_' num2str(d)]};
net.addLayer(name,block,inname,coutname,weights);
% Summation
block = dagnn.Sum();
name = ['Sum_rdb_' num2str(d)];
net.addLayer(name,block,{inname0 coutname},outname);

end


function convObj = convBlock(h,w,c,f)
convObj = dagnn.Conv('size', [h w c f], 'pad', floor([h w h w]./2), 'hasBias', true, 'opts', {});
end