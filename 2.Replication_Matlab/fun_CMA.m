function XMA = fun_CMA(X,order)

lag             = floor(order/2); 
lead            = lag;
base            = lag+1;

Xmat            = zeros(size(X,1),order);
Xmat(:,base)    = X; 

for lag = 0:lag
    Xlag                = [1111*ones(lag,1);X(1:end-lag)];
    Xmat(:,base-lag)    = Xlag;
end

for lead = (1:lead)
    Xlead               = [X(lead+1:end);1111*ones(lead,1)];
    Xmat(:,base+lead)   = Xlead; 
end
  
Xmean = zeros(size(X,1),1);

for t = 1:size(X,1)
    ind         = (Xmat(t,:)~=1111);
    Xmean(t)    = mean(Xmat(t,ind));
end
    
XMA = Xmean;  

