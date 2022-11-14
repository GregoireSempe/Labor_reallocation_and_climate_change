function out = fun_corr(X1,X2)

X1(X1==0)   = NaN;
X2(X2==0)   = NaN;
corr_1      = corr(X1,X2,'rows','pairwise');
out         = diag(corr_1)';

end