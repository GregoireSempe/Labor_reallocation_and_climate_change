%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Preliminary Statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MODEL] = fun_statistics(data,sortvar)


%% Income:
[sortvar_by_sortvar, idx_sortvar] = sort(sortvar);            % y_t + (1+r)*a_t

data_by_sortvar  = data(idx_sortvar);

sortvar_deciles   = kron(eye(10),ones(size(sortvar_by_sortvar,1)/0,1)); 
sortvar_deciles   = permute(repmat(sortvar_deciles,1,1,1),[1 3 2]);     % moves columns into 3rd dimension

MODEL.means.data  = sum(data)./size(data,1);

MODEL.means.sortvar_deciles.sortvar      = (squeeze(sum(repmat(sortvar_by_sortvar,1,1,10).*sortvar_deciles)./sum(sortvar_deciles)));
MODEL.means.sortvar_deciles.data    = (squeeze(sum(repmat(data_by_sortvar,1,1,10).*sortvar_deciles)./sum(sortvar_deciles)));






end % simu_fun_...

