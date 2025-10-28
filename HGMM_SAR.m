function [theta_hat se e_HGMM]=HGMM_SAR(Y,X,W)
% Huber GMM estimator in Liu, Xu, Lee, and Mei (2025)
%This estimator is robust to outliers and conditional heteroskedasticity
%X: the 1st column is 1
% Use [G*X X] as IV matrix
%P_n: zero diagonals
options = optimset('Display','off', 'MaxFunEvals', 1e4);

[n,K]=size(X);
psi=@(x,M)M*(x>=M)+x.*(x<M).*(x>-M)-M*(x<(-M));
psi_prime=@(t,M) abs(t)<=M;
Z=[W*Y X];
  
    
options = optimset('Display','off');


  theta_BIV=Huber_bestIV(Y, X, W);

  %residual and the critical value M
  e=Y-Z*theta_BIV;
  M=1.345*median(abs(e-median(e)))/norminv(0.75);
  
  %P and Q in GMM
  G=W/((eye(n)-theta_BIV(1)*W));
  P_GMM=G-diag(diag(G));
  BIV=G*(X*theta_BIV(2:(1+K))); 
  Q_GMM=[BIV X];  
  
  %Non-iid: Eq(13) in Lin and Lee
  psi_e2=psi(e.^2,M);
  V=zeros(2+K);   %V=Omega in Lin and Lee (2010)
  V(1,1)=psi_e2'*(P_GMM.*(P_GMM+P_GMM'))*psi_e2/n;
  V(2:(2+K),2:(2+K))=Q_GMM'*diag(psi_e2)*Q_GMM/n;
  V_inv=inv(V);
  
  resi=@(theta)psi(Y-Z*theta,M);
  g=@(theta)[resi(theta)'*P_GMM*resi(theta); Q_GMM'*resi(theta)];
  GMM=@(theta)g(theta)'*V_inv*g(theta)/n;
  theta_hat=fminunc(GMM,theta_BIV, options);   
  
  %3rd stage GMM
    e=Y-Z*theta_hat;
    M=1.345*median(abs(e-median(e)))/norminv(0.75);
    
    G=W*((eye(n)-theta_hat(1)*W)\eye(n));
    P_GMM=G-diag(diag(G));
    BIV=G*(X*theta_hat(2:(1+K)));
    Q_GMM=[BIV X];
      
     %Non-iid: Eq(13) in Lin and Lee
     psi_e2=(psi(e,M)).^2;
     V=zeros(2+K);   %V=Omega in Lin and Lee (2010)
     V(1,1)=psi_e2'*(P_GMM.*(P_GMM+P_GMM'))*psi_e2/n;
     V(2:(2+K),2:(2+K))=Q_GMM'*diag(psi_e2)*Q_GMM/n;
     V_inv=inv(V);
     
     resi=@(theta)psi(Y-Z*theta,M);
     g=@(theta)[resi(theta)'*P_GMM*resi(theta); Q_GMM'*resi(theta)];
     GMM=@(theta)g(theta)'*V_inv*g(theta)/n;
     %theta_hat=fminunc(GMM,theta_hat, options);  

gs = GlobalSearch;
problem = createOptimProblem('fmincon','x0',theta_hat,'objective',GMM);
theta_hat = run(gs,problem);



     %calculate standard deviation 
    e_HGMM=Y-Z*theta_hat;
    psi_e2=(psi(e_HGMM,M)).^2;
    V_hat=zeros(2+K);   %V=Omega in Lin and Lee (2010)
    V_hat(1,1)=psi_e2'*(P_GMM.*(P_GMM+P_GMM'))*psi_e2;
    V_hat(2:(2+K),2:(2+K))=Q_GMM'*diag(psi_e2)*Q_GMM;
    Gamma_hat=zeros(rank(Z)+1,rank(Z));
    Gamma_hat(1,1)=psi_prime(e_HGMM,M)'*((P_GMM+P_GMM').*G)*(psi(e_HGMM,M).*e_HGMM);
    Gamma_hat(2:rank(Z)+1,1)=(repmat(psi_prime(e_HGMM,M),1,rank(Q_GMM)).*Q_GMM)'*W*Y;
    Gamma_hat(2:rank(Z)+1,2:rank(Z))=(repmat(psi_prime(e_HGMM,M),1,rank(Q_GMM)).*Q_GMM)'*X;
    Sigma2=eye(rank(Z))/(Gamma_hat'*(eye(rank(Z)+1)/V_hat)*Gamma_hat);
    se=sqrt(diag(Sigma2));    %standard error

end